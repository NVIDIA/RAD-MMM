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
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tts_text_processing.text_processing import TextProcessing
from data import AudioDataset, DataCollate, TextOnlyData
import inspect
from typing import Optional
import json


class BaseAudioDataModule(pl.LightningDataModule):
    def __init__(self, dataloader_type, training_files, validation_files, filter_length,
                 hop_length, win_length, sampling_rate, n_mel_channels,
                 mel_fmin, mel_fmax, f0_min, f0_max, max_wav_value, use_f0,
                 use_energy_avg, use_log_f0, use_scaled_energy,
                 symbol_set="radtts", cleaner_names=["radtts_cleaners"],
                 heteronyms_path="tts_text_processing/heteronyms",
                 phoneme_dict_path="tts_text_processing/cmudict-0.7b",
                 p_phoneme=1.0, handle_phoneme='word',
                 handle_phoneme_ambiguous='ignore', speaker_ids=None,
                 include_speakers=None, n_frames=-1,
                 use_attn_prior_masking=True, prepend_space_to_text=True,
                 append_space_to_text=True, add_bos_eos_to_text=True,
                 betabinom_cache_path: Optional[str]=None,
                 betabinom_scaling_factor=0.05, lmdb_cache_path:
                 Optional[str]=None, dur_min=None, dur_max=None,
                 combine_speaker_and_emotion=False, distance_tx_unvoiced=False,
                 mel_noise_scale=0.0, scale_mel=True, speaker_map=None,
                 accent_map=None,
                 use_prior_interpolator=True, batch_size=8, num_workers=8,
                 phonemizer_cfg: Optional[str]=None, return_audio=False,
                 inference_transcript: Optional[str]=None, predict_mode="tts",
                 reconstruction_files: Optional[str]=None,
                 prediction_recon_files: Optional[dict]=None,
                 use_multilingual_model=False,
                 use_wave_augmentations=False,
                 wave_aug_config=None,
                 g2p_type='phonemized',
                 speaker_stats_path=None,
                 include_emotions=None,
                 f0_pred_type="norm_log_f0"
                 ):
        all_args = locals()
        super().__init__()
        # filter for stuff that's not in the init
        self.prepare_data_per_node = True
        dset_signature = inspect.signature(AudioDataset.__init__).parameters.keys() - {'self', 'datasets'}
        dset_args = {}
        for k, v in all_args.items():
            if k in dset_signature:
                dset_args[k] = v
        self.dset_args = dset_args
        self.batch_size = batch_size
        self.training_files = training_files
        self.validation_files = validation_files
        self.prediction_recon_files = prediction_recon_files
        self.num_workers = num_workers
        self.inference_transcript = inference_transcript
        self.predict_mode = predict_mode
        self.collate_fn = DataCollate()
        if phonemizer_cfg is not None and \
            type(phonemizer_cfg) == str:
                phonemizer_cfg = json.loads(phonemizer_cfg)
        
        self.phonemizer_cfg = phonemizer_cfg
        self.tp = TextProcessing(
            symbol_set, cleaner_names, heteronyms_path, phoneme_dict_path,
            p_phoneme=p_phoneme, handle_phoneme=handle_phoneme,
            handle_phoneme_ambiguous=handle_phoneme_ambiguous,
            prepend_space_to_text=prepend_space_to_text,
            append_space_to_text=append_space_to_text,
            add_bos_eos_to_text=add_bos_eos_to_text,
            g2p_type=g2p_type,
            phonemizer_cfg=self.phonemizer_cfg)
        self.combine_speaker_and_emotion = combine_speaker_and_emotion
        self.speaker_stats_path = speaker_stats_path
        assert(predict_mode in {'tts', 'ttsLive', 'reconstruction'})
        self.f0_pred_type = f0_pred_type


    def setup(self, stage: Optional[str]=None, inferData: Optional[str]=None):
        trainset = AudioDataset(datasets=self.training_files,
                                tp=self.tp,
                                **self.dset_args)
        self.speaker_ids = trainset.speaker_ids
        self.accent_ids = trainset.accent_ids
        if stage == "fit" or stage is None:
            print("initializing training dataloader")

            print("initializing validation dataloader")
            self.dset_args['speaker_ids'] = self.speaker_ids
            valset = AudioDataset(datasets=self.validation_files,
                                  tp=self.tp,
                                  **self.dset_args)

            self.trainset, self.valset = trainset, valset

        elif stage == "predict":
            # initialize trainset for the speaker id mapping

            if self.predict_mode == "tts":
                self.predictset = TextOnlyData(self.inference_transcript,
                                                self.tp,
                                                self.speaker_ids,
                                                self.accent_ids,
                                                self.combine_speaker_and_emotion,
                                                self.speaker_stats_path,
                                                self.f0_pred_type)
            else:
                self.dset_args['speaker_ids'] = self.speaker_ids
                self.predictset = AudioDataset(datasets=self.prediction_recon_files,
                                               tp=self.tp,
                                               **self.dset_args)


    def train_dataloader(self):
        train_loader = DataLoader(
            self.trainset, num_workers=self.num_workers, shuffle=False,
            batch_size=self.batch_size, pin_memory=False, drop_last=True,
            collate_fn=self.collate_fn, timeout=20)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.valset, num_workers=self.num_workers, shuffle=False,
            batch_size=self.batch_size, pin_memory=False, drop_last=False,
            collate_fn=self.collate_fn, timeout=20)
        return val_loader

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        predict_loader = DataLoader(self.predictset, batch_size=self.batch_size,
                                    shuffle=False, pin_memory=False, drop_last=False,
                                    collate_fn=self.collate_fn if self.predict_mode=="reconstruction" else None)
        return predict_loader
