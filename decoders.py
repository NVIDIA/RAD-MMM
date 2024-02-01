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
from typing import Optional
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import Invertible1x1ConvLUS, Invertible1x1Conv, DataInitializedInvertible1x1Conv
from common import AffineTransformationLayer, SplineTransformationLayer
from common import SequenceLength
from common import ConvLSTMLinear, LinearNorm, WN, WaveNetOriginal
from hifigan_models import Generator
from hifigan_env import AttrDict
from models.radmmm import RADMMM
from utils import freeze


class FlowStep(nn.Module):
    def __init__(self,  n_mel_channels, n_context_dim, n_layers,
                 affine_model='simple_conv', scaling_fn='exp',
                 mode='LUS', affine_activation='softplus',
                 use_partial_padding=False, cache_inverse=False,
                 use_spline=False, use_bn=True):
        super(FlowStep, self).__init__()
        assert mode in {'LUS', 'whiten'}
        if mode == 'LUS':
            self.invtbl_conv = Invertible1x1ConvLUS(
                n_mel_channels, cache_inverse=cache_inverse)
        elif mode == 'whiten':
            self.invtbl_conv = DataInitializedInvertible1x1Conv(
                n_mel_channels, cache_inverse=cache_inverse)

        if use_spline:
            left = -3
            right= 3
            top = 3
            bottom = -3
            n_bins = 32
            self.coupling_tfn = SplineTransformationLayer(
                n_mel_channels, n_context_dim, n_layers, scaling_fn=scaling_fn,
                top=top, bottom=bottom, left = left, right=right,
                n_bins=n_bins, use_quadratic=True, use_bn=use_bn
            )
        else:
            self.coupling_tfn = AffineTransformationLayer(
                n_mel_channels, n_context_dim, n_layers, affine_model=affine_model,
                scaling_fn=scaling_fn, affine_activation=affine_activation,
                use_partial_padding=use_partial_padding)

    def enable_inverse_cache(self):
        """Enable during inference to avoid recomputing invertible 1x1s """
        self.invtbl_conv.cache_inverse = True

    def forward(self, z, context, inverse=False, seq_lens=None):
        if inverse:  # for inference z -> mel
            z = self.coupling_tfn(z, context, inverse, seq_lens=seq_lens)
            z = self.invtbl_conv(z, inverse, lens=seq_lens)
            return z
        else:  # training mel -> z
            z, log_det_W = self.invtbl_conv(z, lens=seq_lens)
            z, log_s = self.coupling_tfn(z, context, seq_lens=seq_lens)
            return z, log_det_W, log_s

class RADMMMFlow(RADMMM):
    def __init__(self, n_speaker_dim=16,
                 use_accent=True,
                 n_accent_dim=1,
                 n_text_dim=512, n_group_size=1, n_mel_channels=80,
                 use_spk_emb_for_alignment=False,
                 n_f0_dims=1, n_energy_avg_dims=1,
                 context_w_f0_and_energy=True,
                 use_context_lstm=True,
                 context_lstm_norm: Optional[str]=None,
                 n_flows=8, n_conv_layers_per_step=4, n_early_size=2,
                 n_early_every=2, affine_model: str='wavenet', scaling_fn: str='tanh',
                 affine_activation: str='softplus', use_partial_padding=True, n_splines=0, use_bn=True,
                 freeze_whitening_layer=False,
                 use_accent_emb_for_decoder=False):
        super(RADMMMFlow, self).__init__(n_speaker_dim,
            use_accent, n_accent_dim,
            n_text_dim, n_group_size, n_mel_channels,
            use_spk_emb_for_alignment,
            n_f0_dims,
            n_energy_avg_dims, context_w_f0_and_energy, use_context_lstm,
            context_lstm_norm,
            use_accent_emb_for_decoder=use_accent_emb_for_decoder)
        assert n_speaker_dim % 2 == 0
        assert n_early_size % 2 == 0
        self.use_accent = bool(use_accent)
        if self.use_accent:
            assert n_accent_dim % 2 == 0
        self.matrix_decomposition = "LUS"
        self.use_partial_padding = use_partial_padding
        self.flows = nn.ModuleList()
        self.affine_activation = affine_activation
        self.freeze_whitening_layer = freeze_whitening_layer
        self.n_flows = n_flows
        self.n_group_size = n_group_size

        if self.n_group_size > 1:
            self.unfold_params = {'kernel_size': (n_group_size, 1),
                                  'stride': n_group_size,
                                  'padding': 0, 'dilation': 1}
            self.unfold = nn.Unfold(**self.unfold_params)

        self.exit_steps = []
        self.n_early_size = n_early_size
        n_mel_channels = n_mel_channels * n_group_size

        for i in range(self.n_flows):
            if i > 0 and i % n_early_every == 0:  # early exitting
                n_mel_channels -= self.n_early_size
                self.exit_steps.append(i)
            is_spline_step = i < n_splines
            invtbl_conv_mode = 'LUS'
            if i == 0:
                invtbl_conv_mode = 'whiten'
            self.flows.append(FlowStep(
                n_mel_channels, self.decoder_cond_dims,
                n_conv_layers_per_step, affine_model, scaling_fn,
                invtbl_conv_mode, affine_activation=affine_activation,
                use_partial_padding=self.use_partial_padding,
                use_spline=is_spline_step, use_bn=use_bn))
        if self.freeze_whitening_layer:
            freeze(self.flows[0].invtbl_conv)

    def is_attribute_unconditional(self):
        """
        returns true if the decoder is conditioned on neither energy nor F0
        """
        return self.n_f0_dims == 0 and self.n_energy_avg_dims == 0

    def fold(self, mel):
        """Inverse of the self.unfold(mel.unsqueeze(-1)) operation used for the
        grouping or "squeeze" operation on input

        Args:
            mel: B x C x T tensor of temporal data
        """
        mel = nn.functional.fold(
            mel, output_size=(mel.shape[2]*self.n_group_size, 1),
            **self.unfold_params).squeeze(-1)
        return mel

    def enable_inverse_cache(self):
        """Enable during inference to avoid recomputing invertible 1x1s """
        for flow_step in self.flows:
            flow_step.enable_inverse_cache()

    def forward(self, mel, spk_vecs, context, out_lens,
                f0=None, energy_avg=None, accent_vecs=None):
        context_w_spkvec = self.preprocess_context(
            context, spk_vecs, out_lens.lengths, f0, energy_avg,
            accent_vecs=accent_vecs)

        if self.n_group_size > 1:
            # might truncate some frames at the end, but that's ok
            # sometimes referred to as the "squeeeze" operation
            # invert this by calling self.fold(mel_or_z)
            mel = self.unfold(mel.unsqueeze(-1))

        z_out = []
        log_s_list, log_det_W_list, z_out = [], [], []
        unfolded_seq_lens = SequenceLength(torch.div(out_lens.lengths, self.n_group_size, rounding_mode="floor"))

        for i, flow_step in enumerate(self.flows):

            if i in self.exit_steps:
                z = mel[:, :self.n_early_size]
                z_out.append(z)
                mel = mel[:, self.n_early_size:]
            mel, log_det_W, log_s = flow_step(
                mel, context_w_spkvec, seq_lens=unfolded_seq_lens)
            log_s_list.append(log_s)
            log_det_W_list.append(log_det_W)


        z_out.append(mel)
        z_mel = torch.cat(z_out, 1)

        outputs = {'z_mel': z_mel,
                   'log_det_W_list': log_det_W_list,
                   'log_s_list': log_s_list,
                   'context_w_spkvec': context_w_spkvec
                   }

        return outputs

    def infer(self, spk_vec, txt_enc, sigma, dur=None, f0=None,
              energy_avg=None, out_lens=None, accent_vecs=None):
        if out_lens is None:
            out_lens = torch.LongTensor([dur.sum(1)]).to(txt_enc.device)
        max_n_frames = out_lens.max()

        # get attributes f0, energy, vpred, etc)
        txt_enc_time_expanded = self.length_regulator(
            txt_enc.transpose(1, 2), dur).transpose(1, 2)

        context_w_spkvec = self.preprocess_context(
            txt_enc_time_expanded, spk_vec, out_lens, f0,
            energy_avg, accent_vecs=accent_vecs)

        residual = torch.cuda.FloatTensor(
            txt_enc.shape[0], self.n_mel_channels * self.n_group_size,
            max_n_frames // self.n_group_size)

        residual = residual.normal_() * sigma
        # map from z sample to data
        exit_steps_stack = self.exit_steps.copy()
        mel = residual[:, len(exit_steps_stack) * self.n_early_size:]
        remaining_residual = residual[
            :, :len(exit_steps_stack)*self.n_early_size]
        unfolded_seq_lens = SequenceLength(torch.div(out_lens, self.n_group_size, rounding_mode="floor"))
        for i, flow_step in enumerate(reversed(self.flows)):
            curr_step = len(self.flows) - i - 1
            mel = flow_step(
                mel, context_w_spkvec, inverse=True, seq_lens=unfolded_seq_lens)
            if len(exit_steps_stack) > 0 and curr_step == exit_steps_stack[-1]:
                # concatenate the next chunk of z
                exit_steps_stack.pop()
                residual_to_add = remaining_residual[
                    :, len(exit_steps_stack)*self.n_early_size:]
                remaining_residual = remaining_residual[
                    :, :len(exit_steps_stack)*self.n_early_size]
                mel = torch.cat((residual_to_add, mel), 1)

        if self.n_group_size > 1:
            mel = self.fold(mel)

        return {'mel': mel}
