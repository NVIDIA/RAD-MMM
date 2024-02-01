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
import os
from plotting_utils import plot_alignment_to_numpy, plot_mel_to_numpy, plot_curves_to_numpy
import numpy as np
import copy
from loss import AttentionCTCLoss
import torch
from scipy.io.wavfile import read as readwav
import timeit
from data import beta_binomial_prior_distribution
import torch.nn.functional as F
import torch
from numba import jit
import traceback

class LogDecoderSamplesCallback(pl.Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        decoder_outputs = outputs['decoder_outputs']
        if pl_module.global_rank == 0 and batch_idx == 0 and len(batch) > 0:
            (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior,
             f0, voiced_mask, p_voiced, energy_avg,
             audiopaths, text_raw, f0_mean, 
            f0_std, energy_mean, energy_std,
            language) = pl_module.unpack_batch(batch)

            vocode = pl_module.binarize and pl_module.log_decoder_samples
            reconstruction_outputs = pl_module.reconstruct_from_batch_attributes(batch, vocode=vocode)
            attn_used = reconstruction_outputs['attn_used'][:, :, :, :in_lens.lengths[0]]
            attn_soft = reconstruction_outputs['attn_soft'][:, :, :, :in_lens.lengths[0]]
            audioname = os.path.basename(audiopaths[0])

            pl_module.logger.experiment.add_image(
                'attention_weights_soft',
                plot_alignment_to_numpy(
                    attn_soft[0, 0].data.cpu().numpy().T, title=audioname),
                pl_module.global_step, dataformats='HWC')
            pl_module.logger.experiment.add_image(
                'attention_weights_used',
                plot_alignment_to_numpy(
                    attn_used[0, 0].data.cpu().numpy().T, title=audioname),
                pl_module.global_step, dataformats='HWC')
            if vocode:
                audio_denoised = reconstruction_outputs['output_audio']
                output_mel = reconstruction_outputs['output_mel']
                sample_tag = "decoder_sample_gt_attributes_vocoded"
                pl_module.logger.experiment.add_audio(sample_tag,
                    audio_denoised[0], pl_module.global_step,
                    pl_module.sampling_rate)

                pl_module.logger.experiment.add_image('mel_hat',
                    plot_mel_to_numpy(output_mel[0].data.cpu().numpy(),
                                      title=audioname),
                    pl_module.global_step, dataformats='HWC')

                # vocode the gt mel for comparison
                output_mel = pl_module.mel_descale(mel[0:1,:,:out_lens.lengths[0]])
                vocoder, denoiser = pl_module.synth_vocoder
                audio = vocoder(output_mel.cpu()).float()[0]
                audio_denoised = denoiser(
                    audio, strength=0.00001)[0].float()
                audio_denoised = audio_denoised[0].detach().cpu().numpy()
                audio_denoised = audio_denoised / np.abs(audio_denoised).max()
                sample_tag = "gt_mel_vocoded"
                pl_module.logger.experiment.add_audio(sample_tag,
                                                      audio_denoised, 0,
                                                      pl_module.sampling_rate)


class LogAttributeSamplesCallback(pl.Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch,
                                batch_idx, dataloader_idx=0):
        if pl_module.global_rank == 0 and batch_idx == 0:
            if pl_module.duration_predictor is not None:
                sample_text = ["She sells sea shells by the sea shore, the shells she sells are surely sea shells, so if she sells shells on the sea shore, I am sure she sells seashore shells.",
                               "If Peter Piper picked a peck of pickled peppers, where's the peck of pickled peppers Peter Piper picked?",
                               "Fuzzy Wuzzy was a bear. Fuzzy Wuzzy had no hair. Fuzzy Wuzzy wasn't fuzzy, was he?",
                               "So, this is the sushi chef.",
                               "I wish to wash my Irish wristwatch.",
                               "Scissors sizzle, thistles sizzle, fizzle sizzle, whiz fizz.",
                               "Poo boo too two tow tipsy tee pee Pete.",
                               "Broadway is dark. Curtains won't rise again before the new year. But we have a revival of our look at the new production of Westside story."
                               ]

                language = ["en_US",
                            "en_US",
                            "en_US",
                            "en_US",
                            "en_US",
                            "en_US",
                            "en_US",
                            "en_US"
                           ]

                # accent_ids = torch.tensor([1]*len(language))
                speaker_ids = batch['speaker_ids'][0:len(language)]
                accent_ids = batch['accent_ids'][0:len(language)]

                f0_mean = batch['speaker_f0_mean'][0:len(language)]
                f0_std = batch['speaker_f0_std'][0:len(language)]

                audio_denoised = pl_module.sample_full(sample_text, speaker_ids,
                                                        language=language, accent_ids=accent_ids,
                                                        f0_mean=f0_mean, f0_std=f0_std,
                                                        shift_stats=True)
                for i in range(len(sample_text)):
                    sample_tag = "full_pipeline_sample_vocoded_sequence_%d" %(i)
                    pl_module.logger.experiment.add_audio(sample_tag,
                                                          audio_denoised[i], pl_module.global_step,
                                                          pl_module.sampling_rate)

            self.plot_attribute_predictions_gt_duration(pl_module, outputs)
            self.sample_f0(pl_module, outputs, batch)


    def plot_attribute_predictions_gt_duration(self, pl_module, outputs):

        if pl_module.f0_predictor is not None:
            # plot f0 curves
            f0_outputs = outputs['f0_outputs']
            f0_plot = plot_curves_to_numpy(f0_outputs['x'][0,0].data.cpu().numpy(),
                                           f0_outputs['x_hat'][0,0].data.cpu().numpy(),
                                           't', 'F0')
            pl_module.logger.experiment.add_image('f0_contours', f0_plot,
                                                  pl_module.global_step, dataformats='HWC')

        if pl_module.energy_predictor is not None:
            # plot energy curves
            energy_outputs = outputs['energy_outputs']
            energy_plot = plot_curves_to_numpy(energy_outputs['x'][0,0].data.cpu().numpy(),
                                               energy_outputs['x_hat'][0,0].data.cpu().numpy(),
                                               't', 'ENERGY')
            pl_module.logger.experiment.add_image('energy_contours', energy_plot,
                                                  pl_module.global_step, dataformats='HWC')

        if pl_module.voiced_predictor is not None:
            # plot voiced decision curves
            voiced_outputs = outputs['voiced_outputs']
            voiced_plot = plot_curves_to_numpy(voiced_outputs['x'][0,0].data.cpu().numpy(),
                                               torch.sigmoid(voiced_outputs['x_hat'][0,0]).data.cpu().numpy(), 't', 'VOICED')
            pl_module.logger.experiment.add_image('voiced_decisions', voiced_plot,
                                                  pl_module.global_step, dataformats='HWC')


    def sample_f0(self, pl_module, outputs, batch):
        (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior,
         f0, voiced_mask, p_voiced, energy_avg,
         audiopaths, text_raw, f0_mean, 
         f0_std, energy_mean, energy_std,
         language) = pl_module.unpack_batch(batch)
        decoder_outputs = outputs['decoder_outputs']
        attn_used = decoder_outputs['attn'][:, :, :, :in_lens.lengths[0]]
        attn_soft = decoder_outputs['attn_soft'][:, :, :, :in_lens.lengths[0]]
        text = text[:, :in_lens.lengths[0]]
        audioname = os.path.basename(audiopaths[0])
        context = decoder_outputs['context']
        spk_vecs = decoder_outputs['spk_vecs']
        accent_vecs = decoder_outputs['accent_vecs']
        # if pl_module.f0_predictor is not None:
        #     f0_pred = pl_module.f0_predictor.infer(context, spk_vecs, out_lens, f0_mean, f0_std)
        #     f0_pred = f0_pred * voiced_mask.unsqueeze(1)
        # else:
        f0_pred = f0.unsqueeze(1)
        if attn_used is not None and \
           pl_module.log_decoder_samples and \
           pl_module.binarize: # wait until binarization enabled
            durations = attn_used[0, 0].sum(0, keepdim=True).int()
            try:
                spk_vecs = decoder_outputs['spk_vecs']
                accent_vecs = decoder_outputs['accent_vecs']
                txt_enc = decoder_outputs['txt_enc']
                model_inference_output = pl_module.sample_decoder(
                    spk_vec=spk_vecs[0:1], txt_enc=txt_enc[0:1], sigma=1.0,
                    dur=durations, f0=f0_pred[0:1, 0, :durations.sum()],
                    energy_avg=energy_avg[0:1, :durations.sum()])
                output_mel = pl_module.mel_descale(model_inference_output['mel'])
                vocoder, denoiser = pl_module.synth_vocoder
                audio = vocoder(output_mel.cpu()).float()[0]
                audio_denoised = denoiser(
                    audio, strength=0.00001)[0].float()
                audio_denoised = audio_denoised[0].detach().cpu().numpy()
                audio_denoised = audio_denoised / np.abs(audio_denoised).max()
                sample_tag = "decoder_sample_pred_f0"
                pl_module.logger.experiment.add_audio(sample_tag,
                    audio_denoised, pl_module.global_step,
                    pl_module.sampling_rate)
            except:
                print("error occured during sample generation")
                traceback.print_exc()

class LogAttributeSamplesCallbackv2(LogAttributeSamplesCallback):
    def sample_f0(self, pl_module, outputs, batch):
        (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior,
         f0, voiced_mask, p_voiced, energy_avg,
         audiopaths, text_raw, f0_mean, 
         f0_std, energy_mean, energy_std,
         language) = pl_module.unpack_batch(batch)
        decoder_outputs = outputs['decoder_outputs']
        attn_used = decoder_outputs['attn'][:, :, :, :in_lens.lengths[0]]
        attn_soft = decoder_outputs['attn_soft'][:, :, :, :in_lens.lengths[0]]
        text = text[:, :in_lens.lengths[0]]
        audioname = os.path.basename(audiopaths[0])
        context = decoder_outputs['context']
        spk_vecs = decoder_outputs['spk_vecs']
        accent_vecs = decoder_outputs['accent_vecs']
        if pl_module.f0_predictor is not None:
            f0_pred = pl_module.f0_predictor.infer(context, spk_vecs, out_lens, f0_mean, f0_std)
            f0_pred = f0_pred * voiced_mask.unsqueeze(1)
        else:
            f0_pred = f0.unsqueeze(1)
        if attn_used is not None and \
           pl_module.log_decoder_samples and \
           pl_module.binarize: # wait until binarization enabled
            durations = attn_used[0, 0].sum(0, keepdim=True).int()
            try:
                spk_vecs = decoder_outputs['spk_vecs']
                txt_emb = decoder_outputs['txt_emb']
                accent_vecs = decoder_outputs['accent_vecs']
                model_inference_output = pl_module.sample_decoder(
                    spk_vec=spk_vecs[0:1], txt_emb=txt_emb[0:1], sigma=1.0,
                    dur=durations, f0=f0_pred[0:1, 0, :durations.sum()],
                    energy_avg=energy_avg[0:1, :durations.sum()])
                output_mel = pl_module.mel_descale(model_inference_output['mel'])
                vocoder, denoiser = pl_module.synth_vocoder
                audio = vocoder(output_mel.cpu()).float()[0]
                audio_denoised = denoiser(
                    audio, strength=0.00001)[0].float()
                audio_denoised = audio_denoised[0].detach().cpu().numpy()
                audio_denoised = audio_denoised / np.abs(audio_denoised).max()
                sample_tag = "decoder_sample_pred_f0"
                pl_module.logger.experiment.add_audio(sample_tag,
                    audio_denoised, pl_module.global_step,
                    pl_module.sampling_rate)
            except:
                print("error occured during sample generation")
                traceback.print_exc()