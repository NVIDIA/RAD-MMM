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
from common import ConvNorm, Invertible1x1Conv
from common import ConvLSTMLinear, SequenceLength, LSTMConv
from typing import Optional

class BottleneckLayer(nn.Module):
    def __init__(self, in_dim, reduction_factor=16, norm='weightnorm',
                 non_linearity='leakyrelu', kernel_size=3, use_partial_padding=True):
        super(BottleneckLayer, self).__init__()

        self.reduction_factor = reduction_factor
        reduced_dim = int(in_dim / reduction_factor)
        self.out_dim = reduced_dim
        if self.reduction_factor > 1:
            fn = ConvNorm(in_dim, reduced_dim, kernel_size=kernel_size,
                          use_weight_norm=(norm == 'weightnorm'))
            if norm == 'instancenorm':
                fn = nn.Sequential(
                    fn, nn.InstanceNorm1d(reduced_dim, affine=True))

            self.projection_fn = fn
            self.non_linearity = nn.ReLU()
            if non_linearity == 'leakyrelu':
                self.non_linearity= nn.LeakyReLU()

    def forward(self, x, mask):
        if self.reduction_factor > 1:
            x = self.projection_fn(x, mask.unsqueeze(1))
            x = self.non_linearity(x)
        return x


class AttributePredictor(nn.Module):
    def __init__(self, target_scale=1, target_offset=0, log_target=False,
                normalize_target=False, normalization_type=None):
        super(AttributePredictor, self).__init__()
        self.target_scale = target_scale
        self.target_offset = target_offset
        self.log_target = log_target
        self.normalize_target = normalize_target
        self.normalization_type = normalization_type

    def tx_data(self, x, x_mean=None, x_std=None):
        if self.normalize_target:
            print(f'x_mean={x_mean}|x_std={x_std}')
            assert self.normalization_type is not None
            if self.normalization_type == 'norm_lin_space':
                assert torch.all(x_mean > 0.0).item()
                assert torch.all(x_std > 0.0).item()
                x_recon = x.clone()
                x_mean_expanded = x_mean[:, None].expand(-1, x_recon.shape[1])
                x_std_expanded = x_std[:, None].expand(-1, x_recon.shape[1])
                x_recon = x_recon - x_mean_expanded / x_std_expanded
                
                x_recon = torch.log(x_recon + 10)
                
                x = x_recon / 3 # scale to ~ [0, 1] in log space
            elif self.normalization_type == 'norm_log_space':
                assert torch.all(x_mean > 0.0).item()
                assert torch.all(x_std > 0.0).item()
                x_recon = x.clone()
                print(f'begin f0 stats: {x_recon.mean()} +/- {x_recon.std()}')
                print(f'speaker f0 stats: {x_mean.mean()} +/- {x_std.mean()}')
                
                # f0 already in log space
                # x_recon = torch.log(
                #     x_recon + 10)
                
                x_mean_exp = x_mean[:, None, None].expand(-1, 1, x_recon.shape[2])
                x_std_exp = x_std[:, None, None].expand(-1, 1, x_recon.shape[2])

                # normalize in the log space.
                x_recon = (x_recon - x_mean_exp) / x_std_exp

                x_target = (x_recon + 5) / 10 # scale to ~ [0, 1] in log space
                print(f'transformed f0 stats: {x_target.mean()} +/- {x_target.std()}')

                x = x_target
        else:
            x = x*self.target_scale + self.target_offset
            if self.log_target:
                x = torch.log(x+1)

        return x

    def inv_tx_data(self, x, x_mean=None, x_std=None):
        if self.normalize_target:
            assert self.normalization_type is not None
            if self.normalization_type == 'norm_lin_space' and \
                x_mean is not None and \
                    x_std is not None:
                x = torch.exp(x * 3) - 10
                x = x * x_std + x_mean
            elif self.normalization_type == 'norm_log_space' and \
                x_mean is not None and \
                    x_std is not None:
                assert self.normalization_type is not None
                
                x = x * 10 - 5

                x_mean_exp = x_mean[:, None, None].expand(-1, 1, x.shape[2])
                x_std_exp = x_std[:, None, None].expand(-1, 1, x.shape[2])

                x = x * x_std_exp + x_mean_exp
                # already in log space
                # x = torch.exp(x) - 10 
        else:
            if self.log_target:
                x = torch.exp(x) - 1
            x = (x - self.target_offset)/self.target_scale
        return x

    def forward(self, x_target, text_enc, spk_emb, lens: SequenceLength):
        pass

    def infer(self, text_enc, spk_emb, lens: SequenceLength):
        pass


class ConvLSTMLinearDAP(AttributePredictor):
    def __init__(self, n_speaker_dim=16, n_accent_dim=0, in_dim=512, out_dim=1, reduction_factor=16,
                 n_backbone_layers=2, n_hidden=256, kernel_size=3,
                 p_dropout=0.25, target_scale=1, target_offset=0, log_target=False, lstm_type: Optional[str]='bilstm',
                 use_speaker_embedding=True,
                 normalize_target=False,
                 normalization_type=None):
        super(ConvLSTMLinearDAP, self).__init__(target_scale, target_offset, log_target,
                                                normalize_target, normalization_type)
        self.use_speaker_embedding = bool(use_speaker_embedding)
        self.bottleneck_layer = BottleneckLayer(in_dim=in_dim,
                                                reduction_factor=reduction_factor)
        backbone_in_dim = self.bottleneck_layer.out_dim
        if use_speaker_embedding:
            backbone_in_dim += n_speaker_dim
        
        self.feat_pred_fn = ConvLSTMLinear(in_dim=backbone_in_dim,
                                           out_dim=out_dim, n_layers=n_backbone_layers,
                                           n_channels=n_hidden,
                                           kernel_size=kernel_size,
                                           p_dropout=p_dropout,
                                           lstm_type=lstm_type)

        self.normalize_target = normalize_target
        self.normalization_type = normalization_type

    def forward(self, x_target, text_enc, spk_emb, lens: SequenceLength,
                x_mean=None, x_std=None):
        if x_target is not None:
            x_target = self.tx_data(x_target, x_mean, x_std)

        
        # print(f'x_target={x_target}, x_mean={x_mean}, x_std={x_std}')
        
        txt_enc = self.bottleneck_layer(text_enc, lens.mask)
        spk_emb_expanded = spk_emb[..., None].expand(-1, -1, text_enc.shape[2])
        context = torch.cat((txt_enc, spk_emb_expanded), 1)
        x_hat = self.feat_pred_fn(context, lens)
        outputs = {'x_hat': x_hat, 'x': x_target}
        return outputs

    def infer(self, text_enc, spk_emb, lens: SequenceLength,
             x_mean=None, x_std=None):
        res = self.forward(None, text_enc, spk_emb, lens)
        return self.inv_tx_data(res['x_hat'], x_mean, x_std)


class LSTMConvDAP(AttributePredictor):
    def __init__(self, n_speaker_dim=16, in_dim=512, out_dim=1, reduction_factor=16,
                 n_backbone_layers=2, n_hidden=256, kernel_size=3,
                 p_dropout=0.25, target_scale=1, target_offset=0, log_target=False, lstm_norm_fn='spectral'):
        super(LSTMConvDAP, self).__init__(target_scale, target_offset, log_target)

        self.bottleneck_layer = BottleneckLayer(in_dim=in_dim,
                                                reduction_factor=reduction_factor)
        backbone_in_dim = self.bottleneck_layer.out_dim + n_speaker_dim
        self.feat_pred_fn = LSTMConv(in_dim=backbone_in_dim,
                                     out_dim=out_dim, n_layers=n_backbone_layers,
                                     n_channels=n_hidden,
                                     kernel_size=kernel_size,
                                     p_dropout=p_dropout, lstm_norm_fn=lstm_norm_fn)

    def forward(self, x_target, text_enc, spk_emb, lens: SequenceLength):
        if x_target is not None:
            x_target = self.tx_data(x_target)
        txt_enc = self.bottleneck_layer(text_enc, lens.mask)
        spk_emb_expanded = spk_emb[..., None].expand(-1, -1, text_enc.shape[2])
        context = torch.cat((txt_enc, spk_emb_expanded), 1)
        x_hat = self.feat_pred_fn(context, lens)
        outputs = {'x_hat': x_hat, 'x': x_target}
        return outputs

    def infer(self, text_enc, spk_emb, lens: SequenceLength):
        res = self.forward(None, text_enc, spk_emb, lens)
        return self.inv_tx_data(res['x_hat'])