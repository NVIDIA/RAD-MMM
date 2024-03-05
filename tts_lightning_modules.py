# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import argparse
import json
import os
import hashlib
import torch
import torchvision
import numpy as np
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from radam import RAdam
from models.radmmm import RADMMM
from data import AudioDataset, DataCollate
from common import update_params, to_image, SequenceLength
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from data_modules import BaseAudioDataModule
import inspect
from typing import Optional, Dict, Any
from tts_text_processing.text_processing import TextProcessing
import torch.distributed as dist
from common import Encoder, ConvAttention, ResidualLSTMConv
from common import get_mask_from_lengths, expand_over_time, SequenceLength, LengthRegulator
from utils import freeze, unfreeze, save_current_code
import torch.nn as nn
from alignment import mas_width1 as mas
from maskedbatchnorm1d import MaskedBatchNorm1d
from scipy.io.wavfile import write as write_wav
from vocoders.vocoder_utils import get_vocoder
from vocoders.vocoder_utils import get_audio_for_mels


class TTSModel(LightningModule):
    def __init__(self, decoder: torch.nn.Module, decoder_loss: torch.nn.Module,
                 text_encoder: Optional[torch.nn.Module]=None,
                 f0_predictor: Optional[torch.nn.Module]=None,
                 f0_predictor_loss: Optional[torch.nn.Module]=None,
                 energy_predictor: Optional[torch.nn.Module]=None,
                 energy_predictor_loss: Optional[torch.nn.Module]=None,
                 voiced_predictor: Optional[torch.nn.Module]=None,
                 voiced_predictor_loss: Optional[torch.nn.Module]=None,
                 duration_predictor: Optional[torch.nn.Module]=None,
                 duration_predictor_loss: Optional[torch.nn.Module]=None,
                 speaker_embed_regularization_loss: Optional[torch.nn.Module]=None,
                 accent_embed_regularization_loss: Optional[torch.nn.Module]=None,
                 speaker_accent_cross_regularization_loss: Optional[torch.nn.Module]=None,
                 optim_algo="RAdam", learning_rate=1e-3,
                 weight_decay=1e-6, sigma=1.0, iters_per_checkpoint=3000, seed=None,
                 unfreeze_modules="all", binarization_start_iter=0,
                 output_directory="/debug", log_decoder_samples=True,
                 scale_mel=True, vocoder_config_path: Optional[str]=None,
                 vocoder_checkpoint_path: Optional[str]=None, sampling_rate=22050,
                 symbol_set="radtts", cleaner_names=["radtts_cleaners"],
                 heteronyms_path="tts_text_processing/heteronyms",
                 phoneme_dict_path="tts_text_processing/cmudict-0.7b", p_phoneme=1.0,
                 handle_phoneme='word', handle_phoneme_ambiguous='ignore', prepend_space_to_text=True,
                 append_space_to_text=True, add_bos_eos_to_text=False,
                 p_estimate_ambiguous_phonemes=0.0,
                 phoneme_estimation_start_iter=1000,
                 decoder_path: Optional[str]=None, encoders_path: Optional[str]=None,
                 f0_loss_voiced_only=True,
                 n_speakers=1, n_speaker_dim=16,
                 use_accent=False, n_accents=0, n_accent_dim=0,
                 n_text_dim=512,
                 n_text_tokens=185, lstm_norm_fn='spectral',
                 n_mel_channels=80,
                 use_syncbnorm=False,
                 prediction_output_dir: Optional[str]=None,
                 predict_mode="tts",
                 use_accent_emb_for_encoder=False,
                 use_accent_emb_for_decoder=False,
                 use_accent_emb_for_alignment=False,
                 use_speaker_emb_for_alignment=False,
                 n_augmentations: int=0,
                 phonemizer_cfg: Optional[str]=None
                 ):

        super().__init__()

        if phonemizer_cfg is not None and \
            type(phonemizer_cfg) == str:
                phonemizer_cfg = json.loads(phonemizer_cfg)

        self.tp_inference = TextProcessing(
            symbol_set, cleaner_names, heteronyms_path, phoneme_dict_path,
            p_phoneme=p_phoneme, handle_phoneme=handle_phoneme,
            handle_phoneme_ambiguous='random',
            prepend_space_to_text=prepend_space_to_text,
            append_space_to_text=append_space_to_text,
            add_bos_eos_to_text=add_bos_eos_to_text,
            phonemizer_cfg=phonemizer_cfg)

        self.predict_mode = predict_mode
        assert(predict_mode in {'tts', 'reconstruction'})
        self.f0_loss_voiced_only = f0_loss_voiced_only
        self.phoneme_estimation_start_iter = phoneme_estimation_start_iter
        self.scale_mel = scale_mel
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_mel_channels = n_mel_channels
        
        # text encoding modules
        self.text_embeddings = nn.Embedding(n_text_tokens, n_text_dim)
        self.text_encoder = text_encoder
        
        # speaker encoding module
        if n_augmentations > 0:
            print(f'updating the speakers set: {n_speakers}')
            n_speakers =  n_speakers * (1 + n_augmentations)
        
        self.speaker_embeddings = nn.Embedding(n_speakers, n_speaker_dim)
        
        # accent encoding module
        self.use_accent = bool(use_accent)
        if self.use_accent:
            self.accent_embeddings = nn.Embedding(n_accents, n_accent_dim)
        
        self.use_accent_emb_for_encoder = bool(use_accent_emb_for_encoder)
        self.use_accent_emb_for_decoder = bool(use_accent_emb_for_decoder)
        self.use_accent_emb_for_alignment = bool(use_accent_emb_for_alignment)
        self.use_speaker_emb_for_alignment = bool(use_speaker_emb_for_alignment)

        # attention
        attention_key_dim = n_text_dim
        if use_accent_emb_for_alignment:
            attention_key_dim += n_accent_dim
        elif use_speaker_emb_for_alignment:
            attention_key_dim += n_speaker_dim
        
        self.attention = ConvAttention(n_mel_channels, attention_key_dim)

        self.decoder = decoder
        decoder_loss.n_group_size = decoder.n_group_size
        self.decoder_criterion = decoder_loss
        self.optim_algo = optim_algo
        self.output_directory=output_directory
        self.log_decoder_samples = log_decoder_samples
        self.vocoder_config_path = vocoder_config_path
        self.vocoder_checkpoint_path = vocoder_checkpoint_path
        self.sampling_rate = sampling_rate
        self.p_estimate_ambiguous_phonemes = p_estimate_ambiguous_phonemes
        self.prediction_output_dir = prediction_output_dir
        # load vocoder to CPU to avoid taking up valuable GPU vRAM
        ## attribute prediction submodules
        self.f0_predictor = f0_predictor
        if f0_predictor is not None:
            print("Initializing f0 predictor")
            print(f0_predictor)
            self.f0_predictor_loss = f0_predictor_loss
        self.energy_predictor = energy_predictor
        if energy_predictor is not None:
            print("Initializing energy predictor")
            print(energy_predictor)
            self.energy_predictor = energy_predictor
            self.energy_predictor_loss = energy_predictor_loss
        self.voiced_predictor = voiced_predictor
        if voiced_predictor is not None:
            print("Initializing voiced_predictor")
            print(voiced_predictor)
            self.voiced_predictor_loss = voiced_predictor_loss
        self.duration_predictor = duration_predictor
        if duration_predictor is not None:
            print("Initializing duration_predictor")
            print(duration_predictor)
            self.duration_predictor_loss = duration_predictor_loss
        
        if speaker_embed_regularization_loss is not None:
            print("Initializing Speaker Regularization Component")
            self.speaker_embed_regularization_loss = speaker_embed_regularization_loss

        if accent_embed_regularization_loss is not None:
            print("Initializing Accent Regularization Component")
            self.accent_embed_regularization_loss = accent_embed_regularization_loss
        else:
            self.accent_embed_regularization_loss = None

        if speaker_accent_cross_regularization_loss is not None:
            print("Initializing Speaker<>Accent Regularization Component")
            self.speaker_accent_cross_regularization_loss = speaker_accent_cross_regularization_loss


        if self.global_rank == 0: # seems like global rank is still set at rank0 for all processes here
            vocoder_config = {
                'vocoder_type': 'hifigan',
                'vocoder_map': None,
                'vocoder_config_path': self.vocoder_config_path,
                'vocoder_checkpoint_path': self.vocoder_checkpoint_path
            }
            vocoder, denoiser = get_vocoder(
                **vocoder_config,
                to_cuda=False)
            # put it in a tuple to avoid being registered as a pytorch module
            self.synth_vocoder = (vocoder, denoiser)
        print(self.decoder)
        self.binarize = False
        self.binarization_start_iter = binarization_start_iter
        self.decoder_path = decoder_path
        self.encoders_path = encoders_path
        # string name for top level module such as decoder, f0_predictor etc.
        # we'll load these explicitly but won't be saving them nor updating
        self.pretrained_modules = [] # check this when saving to exclude anything in here
        if self.decoder_path is not None:
            loaded_submodules = self.load_pretrained_decoder(decoder_path)
            self.pretrained_modules += loaded_submodules
            # freezing must be applied to all DDP instances
            freeze(self.decoder)
            print("Loaded pretrained decoder")
        if self.encoders_path is not None:
            loaded_submodules = self.load_pretrained_txt_and_spk_encoders(encoders_path)
            self.pretrained_modules += loaded_submodules
            freeze(self.text_embeddings)
            freeze(self.text_encoder)
            freeze(self.speaker_embeddings)
            if self.use_accent:
                freeze(self.accent_embeddings)
            freeze(self.attention)
            print("Loaded pretrained text, speaker, attention modules")
        self.toggle_syncbnorm(use_syncbnorm)


    def toggle_syncbnorm(self, use_syncbnorm=False):
        for md in [md for md in self.modules() if isinstance(md, MaskedBatchNorm1d)]:
            md.distributed_sync = use_syncbnorm

    def encode_speaker(self, spk_ids):
        spk_vecs = self.speaker_embeddings(spk_ids)
        return spk_vecs

    def encode_accent(self, accent_ids):
        accent_vecs = self.accent_embeddings(accent_ids)
        return accent_vecs

    def encode_text(self, text, in_lens, accent_vecs=None):
        """ encode text """
        text_embeddings = self.text_embeddings(text).transpose(1, 2)

        if accent_vecs is not None:
            accent_vecs_expanded = accent_vecs[..., None].expand(-1, -1, text_embeddings.shape[-1])
            text_embeddings_w_context = torch.cat((text_embeddings, accent_vecs_expanded), dim=1)
        else:
            text_embeddings_w_context = text_embeddings

        if in_lens is None:
            text_enc = self.text_encoder.infer(text_embeddings_w_context).transpose(1, 2)
        else:
            text_enc = self.text_encoder(text_embeddings_w_context, in_lens).transpose(1, 2)

        return text_enc, text_embeddings

    def binarize_attention(self, attn, in_lens, out_lens):
        """For training purposes only. Binarizes attention with MAS. These will
        no longer recieve a gradient
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
        b_size = attn.shape[0]
        with torch.no_grad():
            attn_cpu = attn.data.cpu().numpy()
            attn_out = torch.zeros_like(attn)
            for ind in range(b_size):
                hard_attn = mas(attn_cpu[ind, 0, :out_lens[ind], :in_lens[ind]])
                attn_out[ind, 0, :out_lens[ind], :in_lens[ind]] = torch.tensor(
                    hard_attn, device=attn.get_device())
        return attn_out

    def sample_full(self, raw_text, speaker_ids, decoder_speaker_ids=None,
                    f0_speaker_ids=None, energy_speaker_ids=None, duration_speaker_ids=None,
                    accent_ids=None, language=None, f0_mean=None, f0_std=None,
                    shift_stats=True):
        """
        txt: List[str]
        decoder_speaker_ids: spk ids to use for the decoder
        prosody_speaker_ids: spk ids to use for attribute prediction, in case we want to mix and match
        """
        encoded_text = [] # encoded by tp, not output of text encoder, which comes later
        for raw_t, lang in zip(raw_text, language):
            encoded_text.append(self.tp_inference.encode_text(raw_t, language=lang))

        # compute lengths
        txt_lens = torch.tensor([len(t) for t in encoded_text]).to(self.device)
        txt_lens = SequenceLength(txt_lens)
        max_len = txt_lens.lengths.max()
        
        # convert encoded_text to right-zero-padded tensor
        encoded_text_tensor = torch.zeros((len(encoded_text), max_len), device=self.device)
        for row in range(len(encoded_text)):
            row_len = len(encoded_text[row])
            encoded_text_tensor[row, :row_len] = torch.tensor(encoded_text[row])

        if decoder_speaker_ids is None:
            decoder_speaker_ids = speaker_ids
        if f0_speaker_ids is None:
            f0_speaker_ids = speaker_ids
        if duration_speaker_ids is None:
            duration_speaker_ids = speaker_ids
        if energy_speaker_ids is None:
            energy_speaker_ids = speaker_ids

        decoder_spk_vecs = self.encode_speaker(decoder_speaker_ids)
        f0_spk_vecs = self.encode_speaker(f0_speaker_ids)
        energy_spk_vecs = self.encode_speaker(energy_speaker_ids)
        duration_spk_vecs = self.encode_speaker(duration_speaker_ids)
        
        if accent_ids is not None:
            accent_ids = accent_ids.to(self.device)
            accent_vecs = self.encode_accent(accent_ids)
        elif language is not None:
            accent_ids = [self.encode_accent(torch.tensor(lang)) for lang in language]
            accent_ids = torch.tensor(accent_ids)

        if f0_mean is not None:
            f0_mean = f0_mean.to(self.device)

        if f0_std is not None:
            f0_std = f0_std.to(self.device)

        # model-embedded and further encoded
        # learned embedding (lookup) and encoder (convlstm) representations
        # RADMMM utilizes accent conditioned encoded representations to account
        # for differences in pronunciation of same phoneme for different accents.
        if self.use_accent_emb_for_encoder:
            txt_enc, txt_emb = self.encode_text(encoded_text_tensor.long(),
                                                txt_lens.lengths,
                                                accent_vecs=accent_vecs)
        else:
            txt_enc, txt_emb = self.encode_text(encoded_text_tensor.long(),
                                                txt_lens.lengths)

        # get durations
        durations = self.duration_predictor.infer(txt_enc, duration_spk_vecs, txt_lens)
        durations_int = torch.clamp(torch.round(durations), min=1)*txt_lens.mask.unsqueeze(1)
        durations_int = durations_int.long()
        
        # apply durations
        lr = LengthRegulator()
        context = lr(txt_enc.transpose(1, 2), durations_int[:, 0]).transpose(1, 2)

        # get all attributes
        out_lens = SequenceLength(durations_int[:,0].sum(1))
        voiced_pred = torch.sigmoid(self.voiced_predictor.infer(context, f0_spk_vecs, out_lens)) > 0.5
        f0_pred = self.f0_predictor.infer(context, f0_spk_vecs, out_lens, x_mean=f0_mean, x_std=f0_std) * voiced_pred
        
        # import pdb
        # pdb.set_trace()

        if shift_stats and f0_mean is not None:
            print('....f0 tranformation...')
            f0_mu, f0_sigma = f0_pred[voiced_pred].mean(), f0_pred[voiced_pred].std()
            f0_pred[voiced_pred] = (f0_pred[voiced_pred] - f0_mu) / f0_sigma
            
            f0_mean_exp = f0_mean[:, None, None].expand(-1, 1, f0_pred.shape[2])
            f0_std_exp = f0_std[:, None, None].expand(-1, 1, f0_pred.shape[2])

            f0_pred = f0_pred.float()
            f0_pred[voiced_pred] = f0_pred[voiced_pred].float() * f0_std_exp[voiced_pred].float() + f0_mean_exp[voiced_pred].float()
            
        energy_pred = self.energy_predictor.infer(context, energy_spk_vecs, out_lens)

        # run through the decoder sampling
        output = self.sample_decoder(decoder_spk_vecs, txt_enc, sigma=1.0, dur=durations_int.squeeze(1), f0=f0_pred[:,0], energy_avg=energy_pred[:, 0], out_lens=out_lens.lengths, accent_vecs=accent_vecs)
        output_mel = self.mel_descale(output['mel'])
        output_audio = self.vocode_mels(output_mel, out_lens)
        return output_audio

    def sample_decoder(self, spk_vec, txt_enc, sigma, dur=None, f0=None, energy_avg=None, out_lens=None, accent_vecs=None):
        return self.decoder.infer(spk_vec, txt_enc, sigma, dur=dur, f0=f0, energy_avg=energy_avg, out_lens=out_lens, accent_vecs=accent_vecs)

    def reconstruct_from_batch_attributes(self, batch, durations=None, vocode=True):
        """
        To be use for predict when predict_mode is reconstruction
        Gets attributes from standard dataloader and runs inference, attempting to
        reconstruct the original mel from text, speaker id, f0, energy, and attention-extracted-durations
        Voice cloning can be achieved by changing the speaker id in the filelist
        """
        (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior,
         f0, voiced_mask, p_voiced, energy_avg,
         audiopaths, _dc, f0_mean, 
         f0_std, energy_mean, energy_std,
         language) = self.unpack_batch(batch)
        spk_vecs = self.encode_speaker(speaker_ids)
        
        if self.use_accent:
            accent_vecs = self.encode_accent(accent_ids)
        else:
            accent_vecs = None
        
        # RADMMM utilizes accent conditioned encoded representations to account
        # for differences in pronunciation of same phoneme for different accents.
        if self.use_accent_emb_for_encoder:
            txt_enc, txt_emb = self.encode_text(text,
                                                in_lens.lengths,
                                                accent_vecs=accent_vecs)
        else:
            txt_enc, txt_emb = self.encode_text(text,
                                                in_lens.lengths)

        attn_used, attn_soft, _, attn_logprob = self.compute_attention(
            mel, txt_emb, spk_vecs, accent_vecs, out_lens.lengths, in_lens.lengths,
            attn_prior, True)

        durations = attn_used[:, 0].sum(1).long()
        
        out_lens = SequenceLength(durations.sum(1).long())

        model_inference_output = self.sample_decoder(
            spk_vec=spk_vecs, txt_enc=txt_enc, sigma=1.0,
            dur=durations, f0=f0,
            energy_avg=energy_avg, out_lens=out_lens.lengths,
            accent_vecs=accent_vecs)

        output_mel = self.mel_descale(model_inference_output['mel'])
        output_audio = None
        if vocode:
            output_audio = self.vocode_mels(output_mel, out_lens)
        return {"output_audio": output_audio, "output_mel": output_mel, "attn_used": attn_used,
                "attn_soft": attn_soft}


    def compute_attention(self, mel, txt_emb, spk_vecs, accent_vecs,
                          out_lens, in_lens,
                          attn_prior, binarize_attention=False):
        """
        mel: b x n_mel_channels x max(out_lens)  (un-grouped)
        txt_emb: b x n_text_dim x max(in_lens)
        spk_vecs: b x n_spk_dims
        out_lens, in_lens: list[int] (not SequenceLength object)
        binarize_attention: bool
        """
        attn_mask = get_mask_from_lengths(in_lens)[..., None] == 0
        # attn_mask shld be 1 for unsd t-steps in txt_enc_w_spkvec tensor
        text_embeddings_for_attn = txt_emb
        if self.use_accent_emb_for_alignment:
            accent_vecs_expd = accent_vecs[:, :, None].expand(
                    -1, -1, txt_emb.shape[2])
            text_embeddings_for_attn = torch.cat(
                (text_embeddings_for_attn, accent_vecs_expd.detach()), 1)
        elif self.use_speaker_emb_for_alignment:
            speaker_vecs_expd = spk_vecs[:, :, None].expand(
                -1, -1, txt_emb.shape[2])
            text_embeddings_for_attn = torch.cat(
                (text_embeddings_for_attn, speaker_vecs_expd.detach()), 1)

        attn_soft, attn_logprob = self.attention(
            mel, text_embeddings_for_attn, out_lens, attn_mask, key_lens=in_lens,
            attn_prior=attn_prior)

        attn_hard = None
        if binarize_attention:
            attn = self.binarize_attention(attn_soft, in_lens, out_lens)
            attn_hard = attn
            attn_hard = attn_soft + (attn_hard - attn_soft).detach()
        else:
            attn = attn_soft
        return attn, attn_soft, attn_hard, attn_logprob

    def load_pretrained_submodules(self, checkpoint_path, sm_names):
        loaded_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)['state_dict']
        current_dict = self.state_dict()
        kwrds = sm_names
        for kwrd in kwrds:
            kwrd_len = len(kwrd)
            for k, v in current_dict.items():
                if k[:kwrd_len] == kwrd:
                    current_dict[k] = loaded_dict[k]
        self.load_state_dict(current_dict)
        return kwrds

    def load_pretrained_decoder(self, decoder_path):
        return self.load_pretrained_submodules(decoder_path, sm_names=["decoder"])

    def load_pretrained_txt_and_spk_encoders(self, encoders_path):
        submodule_names = ["text_embeddings", "text_encoder", "speaker_embeddings",
                           "attention"]
        if self.use_accent:
            submodule_names += ["accent_embeddings"]
        return self.load_pretrained_submodules(encoders_path, sm_names=submodule_names)

    # as with load_pretrained_decoder, return the top level module name so it
    # doesn't get overriden later, then manually call self.load_state_dict to load
    # just the modules of interest
    def load_pretrained_f0_predictor(self, f0_predictor_path):
        pass

    def load_pretrained_energy_predictor(self, energy_predictor_path):
        pass

    def load_pretrained_voiced_predictor(self, voiced_predictor_path):
        pass

    def load_pretrained_duration_predictor(self, duration_predictor_path):
        pass

    def on_save_checkpoint(self, checkpoint):
        # delete untracked modules
        to_delete = []
        # we loaded these, so we won't be saving them
        for modules in self.pretrained_modules:
            to_delete += [k for k in checkpoint['state_dict'].keys() if k[:len(modules)] == modules]
        for param in to_delete:
            checkpoint['state_dict'].pop(param)

    def on_load_checkpoint(self, checkpoint):
        # fill in the missing modules to checkpoint using the current state dict
        current_state_dict = self.state_dict()
        # start off with everything that's missing
        missing_keys = current_state_dict.keys() - checkpoint['state_dict'].keys()

        # now add anything pretrained so we don't override those weights
        for k in current_state_dict.keys():
            for tlm in self.pretrained_modules:
                if k[:len(tlm)] == tlm:
                    missing_keys.add(k)
                    break
        loaded_modules = set()
        for k in missing_keys:
            loaded_modules.add(k.split('.')[0]) # just the top level pointer
            checkpoint['state_dict'][k] = current_state_dict[k] # override checkpoint values or add if missing
        for tlm in loaded_modules:
            print("Module %s not loaded from checkpoint" %(tlm))


    def mel_scale(self, mel, lens):
        mel = (mel + 5) / 2
        return mel

    def mel_descale(self, mel):
        mel = mel * 2 - 5
        return mel

    def configure_optimizers(self):
        optim_algo = self.optim_algo
        print("Initializing {} optimizer".format(optim_algo))
        if optim_algo == 'Adam':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                          weight_decay=self.weight_decay, amsgrad=True)
        elif optim_algo == 'RAdam': # original implementation, do not use pytorch built-in version
            optimizer = RAdam(self.parameters(), lr=self.learning_rate,
                              weight_decay=self.weight_decay)
        else:
            print("Unrecognized optimizer {}!".format(optim_algo))
            exit(1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        return optimizer

    def vocode_mels(self, mels, out_lens: SequenceLength):
        vocoder, denoiser = self.synth_vocoder
        output_audio = []
        for i in range(mels.shape[0]):
            current_mel = mels[i:i+1, :, :out_lens.lengths[i]]
            audio_denoised = get_audio_for_mels(mels,
                                                'hifigan',
                                                vocoder,
                                                denoiser=denoiser)
            #audio_denoised = audio_denoised.float()
            output_audio.append(audio_denoised)
        return output_audio

    def forward(self, x):
        # for test-time operations
        return 0

    def forward(self, batch):
        if not os.path.isdir(self.prediction_output_dir):
            os.makedirs(self.prediction_output_dir)
            os.chmod(self.prediction_output_dir, 0o775)
            print("predictions saved to %s" %(self.prediction_output_dir))
        if self.predict_mode == "tts":
            print(batch)
            audio_outputs = self.sample_full(batch['script'], batch['spk_id'], batch['decoder_spk_id'],
                                             batch['f0_spk_id'], batch['energy_spk_id'],
                                             batch['duration_spk_id'],
                                             batch['accent_id'],
                                             language=batch['language'],
                                             f0_mean=batch['speaker_f0_mean'],
                                             f0_std=batch['speaker_f0_std']
                                             )
        elif self.predict_mode == "reconstruction":
            output_dict = self.reconstruct_from_batch_attributes(batch)
            audio_outputs = output_dict['output_audio']
        for iter, wav in enumerate(audio_outputs):
            curr_fname = os.path.join(self.prediction_output_dir, "output_sample_%d_%s.wav" %(batch['idx'][iter], self.predict_mode))
            write_wav(curr_fname, self.sampling_rate, wav)
        return curr_fname

    def on_fit_start(self):
        if self.global_rank == 0:
            save_current_code(self.output_directory)

    def unpack_batch(self, batch):
        speaker_ids = batch['speaker_ids']
        accent_ids = batch['accent_ids']
        text = batch['text']
        language = batch['language']
        in_lens = SequenceLength(batch['input_lengths'])
        out_lens = SequenceLength(batch['output_lengths'])
        mel = self.mel_scale(batch['mel'], out_lens)
        attn_prior = batch['attn_prior']
        f0 = batch['f0']
        voiced_mask = batch['voiced_mask']
        p_voiced = batch['p_voiced']
        energy_avg = batch['energy_avg']
        audiopaths = batch['audiopaths']
        text_raw = batch['text_raw']

        f0_mean = batch['speaker_f0_mean']
        f0_std = batch['speaker_f0_std']
        energy_mean = batch['speaker_energy_mean']
        energy_std = batch['speaker_energy_std']
        
        return (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior, f0,
                voiced_mask, p_voiced, energy_avg, audiopaths, text_raw, f0_mean, 
                f0_std, energy_mean, energy_std, language)

    def on_train_batch_start(self, batch, batch_idx):
        # make sure they're all in eval in case of batchnorm/dropouts
        for module_name in self.pretrained_modules:
            getattr(self, module_name).eval()


    def training_step(self, batch, batch_idx):
        if self.global_step >= self.binarization_start_iter:
            self.binarize = True   # binarization training phase
        else:
            self.binarize = False  # no binarization, soft alignments only
        (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior,
         f0, voiced_mask, p_voiced, energy_avg,
         audiopaths, _dc, f0_mean, 
         f0_std, energy_mean, energy_std,
         language) = self.unpack_batch(batch)
        self.log('global_step', int(self.global_step))
        spk_vecs = self.encode_speaker(speaker_ids)

        if self.use_accent:
            accent_vecs = self.encode_accent(accent_ids)
        
        if self.use_accent_emb_for_encoder:
            txt_enc, txt_emb = self.encode_text(text, in_lens.lengths,
                                                accent_vecs=accent_vecs)
        else:
            txt_enc, txt_emb = self.encode_text(text, in_lens.lengths)

        attn, attn_soft, _, attn_logprob = self.compute_attention(
            mel, txt_emb, spk_vecs, accent_vecs,
            out_lens.lengths, in_lens.lengths,
            attn_prior, self.binarize)

        context = torch.bmm(txt_enc, attn.squeeze(1).transpose(1, 2))

        outputs = self.decoder(
            mel, spk_vecs, context, out_lens,
            f0=f0, energy_avg=energy_avg, accent_vecs=accent_vecs)
        outputs['attn'] = attn
        outputs['attn_soft'] = attn_soft
        outputs['attn_logprob'] = attn_logprob
        outputs['context'] = context
        outputs['spk_vecs'] = spk_vecs
        outputs['accent_vecs'] = accent_vecs

        loss_outputs = dict()
        if self.decoder.training:
            decoder_loss = self.decoder_criterion(
                outputs, in_lens, out_lens, self.global_step)
            loss_outputs.update(decoder_loss)

        if self.f0_predictor is not None:
            # add f0 transformation if applicable
            # import pdb
            # pdb.set_trace()
            f0_outputs = self.f0_predictor(f0.unsqueeze(1),
                                           context.detach(),
                                           spk_vecs.detach(), out_lens,
                                           f0_mean,
                                           f0_std)
            f0_mask = None
            if self.f0_loss_voiced_only:
                f0_mask = voiced_mask.unsqueeze(1)
            f0_loss = self.f0_predictor_loss(f0_outputs, in_lens, out_lens, self.global_step, mask=f0_mask)
            loss_outputs.update(f0_loss)
        if self.energy_predictor is not None:
            # add energy transformation if applicable
            energy_outputs = self.energy_predictor(energy_avg.unsqueeze(1),
                                                   context.detach(),
                                                   spk_vecs.detach(), out_lens)
            energy_loss = self.energy_predictor_loss(energy_outputs, in_lens, out_lens, self.global_step)
            loss_outputs.update(energy_loss)
        if self.voiced_predictor is not None:
            voiced_outputs = self.voiced_predictor(voiced_mask.unsqueeze(1),
                                           context.detach(),
                                           spk_vecs.detach(), out_lens)
            voiced_loss = self.voiced_predictor_loss(voiced_outputs, in_lens, out_lens, self.global_step)
            loss_outputs.update(voiced_loss)

        if self.duration_predictor is not None:
            duration_targets = attn.sum(2).detach()
            duration_outputs = self.duration_predictor(duration_targets,
                                           txt_enc.detach(),
                                           spk_vecs.detach(), in_lens)
            duration_loss = self.duration_predictor_loss(duration_outputs, None, None, self.global_step, in_lens.mask.unsqueeze(1))
            loss_outputs.update(duration_loss)

        if self.speaker_embed_regularization_loss is not None:
            speaker_reg_loss = self.speaker_embed_regularization_loss(self.speaker_embeddings)
            loss_outputs.update(speaker_reg_loss)

        if self.accent_embed_regularization_loss is not None:
            accent_reg_loss = self.accent_embed_regularization_loss(self.accent_embeddings)
            loss_outputs.update(accent_reg_loss)

        if self.speaker_accent_cross_regularization_loss is not None:
            speaker_accent_reg_loss = self.speaker_accent_cross_regularization_loss(
                spk_vecs,
                accent_vecs,
                self.speaker_embeddings,
                self.accent_embeddings
            )
            loss_outputs.update(speaker_accent_reg_loss)

        loss = None
        for k, (v, w) in loss_outputs.items():
            self.log("train/"+k, v, sync_dist=True, on_step=True)
            loss = v * w if loss is None else loss + v * w
        return loss

    def validation_step(self, batch, batch_idx):
        (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior,
         f0, voiced_mask, p_voiced, energy_avg,
         audiopaths, text_raw, f0_mean, 
         f0_std, energy_mean, energy_std,
         language) = self.unpack_batch(batch)
        output_dict = {}

        spk_vecs = self.encode_speaker(speaker_ids)
        accent_vecs = self.encode_accent(accent_ids)
        
        if self.use_accent_emb_for_encoder:
            txt_enc, txt_emb = self.encode_text(text, in_lens.lengths,
                                                accent_vecs=accent_vecs)
        else:
            txt_enc, txt_emb = self.encode_text(text, in_lens.lengths)

        attn, attn_soft, _, attn_logprob = self.compute_attention(
            mel, txt_emb, spk_vecs, accent_vecs,
            out_lens.lengths, in_lens.lengths,
            attn_prior, self.binarize)
        context = torch.bmm(txt_enc, attn.squeeze(1).transpose(1, 2))

        outputs = self.decoder(
            mel, spk_vecs, context, out_lens,
            f0=f0, energy_avg=energy_avg, accent_vecs=accent_vecs)
        outputs['attn'] = attn
        outputs['attn_soft'] = attn_soft
        outputs['attn_logprob'] = attn_logprob
        outputs['context'] = context
        outputs['spk_vecs'] = spk_vecs
        outputs['accent_vecs'] = accent_vecs
        outputs['txt_enc'] = txt_enc
        output_dict['decoder_outputs'] = outputs
        loss_outputs = self.decoder_criterion(
            outputs, in_lens, out_lens, 100000)

        # attribute prediction,
        # don't backprop through this
        if self.f0_predictor is not None:
            f0_outputs = self.f0_predictor(f0.unsqueeze(1),
                                           context,
                                           spk_vecs, out_lens,
                                           f0_mean,
                                           f0_std)
            f0_mask = None
            
            if self.f0_loss_voiced_only:
                f0_mask = voiced_mask.unsqueeze(1)
            
            f0_loss = self.f0_predictor_loss(f0_outputs, in_lens, out_lens, self.global_step, mask=f0_mask)
            
            loss_outputs.update(f0_loss)
            
            output_dict['f0_outputs'] = f0_outputs

        if self.energy_predictor is not None:
            energy_outputs = self.energy_predictor(energy_avg.unsqueeze(1),
                                           context,
                                           spk_vecs, out_lens)
            energy_loss = self.energy_predictor_loss(energy_outputs, in_lens, out_lens, self.global_step)
            loss_outputs.update(energy_loss)
            output_dict['energy_outputs'] = energy_outputs

        if self.voiced_predictor is not None:
            voiced_outputs = self.voiced_predictor(voiced_mask.unsqueeze(1),
                                           context,
                                           spk_vecs, out_lens)
            voiced_loss = self.voiced_predictor_loss(voiced_outputs, in_lens, out_lens, self.global_step)
            loss_outputs.update(voiced_loss)
            output_dict['voiced_outputs'] = voiced_outputs

        if self.duration_predictor is not None:
            duration_targets = attn.sum(2).detach()
            duration_outputs = self.duration_predictor(duration_targets,
                                           txt_enc.detach(),
                                           spk_vecs.detach(), in_lens)
            duration_loss = self.duration_predictor_loss(duration_outputs, None, None, self.global_step, in_lens.mask.unsqueeze(1))
            loss_outputs.update(duration_loss)
            output_dict['duration_outputs'] = duration_outputs

        if self.speaker_embed_regularization_loss is not None:
            speaker_reg_loss = self.speaker_embed_regularization_loss(self.speaker_embeddings)
            loss_outputs.update(speaker_reg_loss)

        if self.accent_embed_regularization_loss is not None:
            accent_reg_loss = self.accent_embed_regularization_loss(self.accent_embeddings)
            loss_outputs.update(accent_reg_loss)

        if self.speaker_accent_cross_regularization_loss is not None:
            speaker_accent_reg_loss = self.speaker_accent_cross_regularization_loss(
                spk_vecs,
                accent_vecs,
                self.speaker_embeddings,
                self.accent_embeddings
            )
            loss_outputs.update(speaker_accent_reg_loss)

        output_dict['loss_outputs'] = loss_outputs
        loss = None
        for k, (v, w) in loss_outputs.items():
            self.log("val/"+k, v)
            loss = v * w if loss is None else loss + v * w
        return output_dict