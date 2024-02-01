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
import torch
from torch import nn
from typing import Optional
from alignment import mas_width1 as mas
from common import Encoder, ConvAttention, LengthRegulator
from common import get_mask_from_lengths, expand_over_time


class RADMMM(torch.nn.Module):
    def __init__(self, n_speaker_dim=16, use_accent=True,
                 n_accent_dim=8, n_text_dim=512, 
                 n_group_size=1, n_mel_channels=80,
                 use_spk_emb_for_alignment=False,
                 n_f0_dims=0,
                 n_energy_avg_dims=0, context_w_f0_and_energy=True,
                 use_context_lstm=True,
                 context_lstm_norm: Optional[str]=None,
                 use_accent_emb_for_decoder=False):
        super(RADMMM, self).__init__()
        self.n_speaker_dim = n_speaker_dim
        self.n_accent_dim = n_accent_dim
        self.n_mel_channels = n_mel_channels
        self.n_f0_dims = n_f0_dims  # >= 1 to trains with f0
        self.n_energy_avg_dims = n_energy_avg_dims  # >= 1 trains with energ
        self.n_speaker_dim = n_speaker_dim
        self.length_regulator = LengthRegulator()

        self.context_w_f0_and_energy = context_w_f0_and_energy
        self.n_group_size = n_group_size

        self.use_accent_emb_for_decoder = bool(use_accent_emb_for_decoder)
        if self.use_accent_emb_for_decoder:
            decoder_cond_dims = (
                self.n_speaker_dim + self.n_accent_dim + 
                (n_text_dim + n_f0_dims + n_energy_avg_dims) * n_group_size)

        self.use_context_lstm = use_context_lstm
        if self.use_context_lstm:
            if self.use_accent_emb_for_decoder:
                n_in_context_lstm = (
                        self.n_speaker_dim + self.n_accent_dim + \
                                n_text_dim * n_group_size)

                n_context_lstm_hidden = int(
                    (self.n_speaker_dim + self.n_accent_dim + \
                            n_text_dim * n_group_size) / 2)
            else:
                n_in_context_lstm = (
                    self.n_speaker_dim + n_text_dim * n_group_size)
                
                n_context_lstm_hidden = int(
                    (self.n_speaker_dim + n_text_dim * n_group_size) / 2)

            n_in_context_lstm = (
                n_f0_dims + n_energy_avg_dims + n_text_dim)
            n_in_context_lstm *= n_group_size
            n_in_context_lstm += self.n_speaker_dim
            if self.use_accent_emb_for_decoder:
                n_in_context_lstm += self.n_accent_dim

            decoder_cond_dims = n_context_lstm_hidden * 2

            self.context_lstm = nn.LSTM(
                input_size=n_in_context_lstm,
                hidden_size=n_context_lstm_hidden, num_layers=1,
                batch_first=True, bidirectional=True)

            if context_lstm_norm is not None:
                if 'spectral' in context_lstm_norm:
                    print("Applying spectral norm to context encoder LSTM")
                    lstm_norm_fn_pntr = nn.utils.spectral_norm
                elif 'weight' in context_lstm_norm:
                    print("Applying weight norm to context encoder LSTM")
                    lstm_norm_fn_pntr = nn.utils.weight_norm

                self.context_lstm = lstm_norm_fn_pntr(
                    self.context_lstm, 'weight_hh_l0')
                self.context_lstm = lstm_norm_fn_pntr(
                    self.context_lstm, 'weight_hh_l0_reverse')
        self.decoder_cond_dims = decoder_cond_dims
        self.decoder_out_dims = n_mel_channels

    def preprocess_context(self, context, spk_vecs, out_lens=None, f0=None,
                           energy_avg=None, accent_vecs=None):
        """ Unfolds speaker, f0 and energy, concatenate them to text and send
        them through a bi-lstm
        f0, energy: b x length
        """
        if f0 is not None:
            f0 = f0[:, None]
        if energy_avg is not None:
            energy_avg = energy_avg[:, None]

        if self.n_group_size > 1:
            # unfolding zero-padded values
            context = self.unfold(context.unsqueeze(-1))
            if f0 is not None:
                f0 = self.unfold(f0[..., None])
            if energy_avg is not None:
                energy_avg = self.unfold(energy_avg[..., None])

        spk_vecs = expand_over_time(spk_vecs, context.shape[2])
        context_w_spkvec = torch.cat((context, spk_vecs), 1)
        if self.use_accent_emb_for_decoder:
            assert accent_vecs != None
            accent_vecs = expand_over_time(accent_vecs, context.shape[2])
            context_w_spkvec = torch.cat((context, spk_vecs, accent_vecs), 1)

        if self.context_w_f0_and_energy:
            if f0 is not None:
                context_w_spkvec = torch.cat((context_w_spkvec, f0), 1)
            if energy_avg is not None:
                context_w_spkvec = torch.cat(
                    (context_w_spkvec, energy_avg), 1)
                    
        if self.use_context_lstm:
            unfolded_out_lens = (torch.div(out_lens, self.n_group_size, rounding_mode='floor')).long().cpu()
            unfolded_out_lens_packed = nn.utils.rnn.pack_padded_sequence(
                context_w_spkvec.transpose(1, 2), unfolded_out_lens,
                batch_first=True, enforce_sorted=False)
            self.context_lstm.flatten_parameters()
            context_lstm_packed_output, _ = self.context_lstm(
                unfolded_out_lens_packed)
            context_lstm_padded_output, _ = nn.utils.rnn.pad_packed_sequence(
                context_lstm_packed_output, batch_first=True)
            context_w_spkvec = context_lstm_padded_output.transpose(1, 2)

        return context_w_spkvec

    def remove_norms(self):
        """Removes spectral and weightnorms from model. Call before inference
        """
        for name, module in self.named_modules():
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0')
                print(f"Removed spectral norm from {name}")
            except:
                pass
            try:
                nn.utils.remove_spectral_norm(module, name='weight_hh_l0_reverse')
                print(f"Removed spectral norm from {name}")
            except:
                pass
            try:
                nn.utils.remove_weight_norm(module)
                print(f"Removed wnorm from {name}")
            except:
                pass