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
import torch.nn as nn
from torch.nn import functional as F
from common import get_mask_from_lengths, SequenceLength
from stft_loss import MultiResolutionSTFTLoss
from typing import Optional


def gan_feature_loss(fmap_r, fmap_g, len_ratios=None):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            if len_ratios is None:
                loss += torch.mean(torch.abs(rl - gl))
            else:
                lens = (len_ratios * rl.shape[2]).ceil().long()
                mask = get_mask_from_lengths(lens).float()
                if len(rl.shape) == 4:
                    mask = mask[:, None, :, None]
                    b, d, t, k = rl.shape
                else:
                    mask = mask[:, None]
                    b, d, t = rl.shape
                    k = 1
                loss += (torch.abs(rl - gl) * mask).sum() / (mask.sum() * d * k)

    return loss

def gan_discriminator_loss(disc_real_outputs, disc_generated_outputs,
                           len_ratios=None):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # least squares gan loss
        if len_ratios is not None:
            lens = (len_ratios * dr.shape[1]).ceil().long()
            mask = get_mask_from_lengths(lens).float()
            r_loss = (((1-dr)**2) * mask).sum() / mask.sum()
            g_loss = (dg**2 * mask).sum() / mask.sum()
        else:
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def gan_generator_loss(disc_outputs, len_ratios=None):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        if len_ratios is not None:
            lens = (len_ratios * dg.shape[1]).ceil().long()
            mask = get_mask_from_lengths(lens).float()
            loss_d = (((1-dg)**2) * mask).sum() / mask.sum()
        else:
            loss_d = torch.mean((1-dg)**2)
        gen_losses.append(loss_d)
        loss += loss_d

    return loss, gen_losses

def compute_flow_loss(z, log_det_W_list, log_s_list, n_elements, n_dims, mask,
                     sigma=1.0):

    log_det_W_total = 0.0
    for i, log_s in enumerate(log_s_list):
        if i == 0:
            log_s_total = torch.sum(log_s * mask)
            if len(log_det_W_list):
                log_det_W_total = log_det_W_list[i]
        else:
            log_s_total = log_s_total + torch.sum(log_s * mask)
            if len(log_det_W_list):
                log_det_W_total += log_det_W_list[i]

    if len(log_det_W_list):
        log_det_W_total *= n_elements

    z = z * mask
    prior_NLL = torch.sum(z*z)/(2*sigma*sigma)

    loss = prior_NLL - log_s_total - log_det_W_total

    denom = n_elements * n_dims
    loss = loss / denom
    loss_prior = prior_NLL / denom
    return loss, loss_prior

class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens, return_all=False):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0),
            value=self.blank_logprob)
        cost_total = 0.0
        all_ctc = []
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid]+1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                :query_lens[bid], :, :key_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(curr_logprob, target_seq,
                                    input_lengths=query_lens[bid:bid+1],
                                    target_lengths=key_lens[bid:bid+1])
            cost_total += ctc_cost
            all_ctc.append(ctc_cost)
        cost = cost_total/attn_logprob.shape[0]
        if return_all:
            return cost, all_ctc

        return cost

class AttentionBinarizationLoss(torch.nn.Module):
    def __init__(self):
        super(AttentionBinarizationLoss, self).__init__()

    def forward(self, hard_attention, soft_attention):
        return F.binary_cross_entropy(
            soft_attention[hard_attention == 1],
            torch.ones_like(soft_attention[hard_attention == 1]),
            reduction='mean')

class AttentionLoss(torch.nn.Module):
    def __init__(self, CTC_blank_logprob=-1, kl_loss_start_iter=5000,
                 binarization_loss_weight=1.0, ctc_loss_weight=0.1):
        super(AttentionLoss, self).__init__()
        self.attn_ctc_loss = AttentionCTCLoss(blank_logprob=CTC_blank_logprob)
        self.attn_bin_loss = AttentionBinarizationLoss()
        self.kl_loss_start_iter = kl_loss_start_iter
        self.binarization_loss_weight = binarization_loss_weight
        self.ctc_loss_weight = ctc_loss_weight

    def forward(self, attn, attn_soft, attn_logprob, global_step, in_lens,
                out_lens):
        loss_dict = {}
        ctc_cost = self.attn_ctc_loss(attn_logprob, in_lens, out_lens)

        loss_dict['loss_ctc'] = (ctc_cost, self.ctc_loss_weight)
        binarization_loss = 0.0
        if global_step > self.kl_loss_start_iter:

            binarization_loss = self.attn_bin_loss(attn, attn_soft)
            loss_dict['binarization_loss'] = (
                binarization_loss, self.binarization_loss_weight)
        else:
            loss_dict['binarization_loss'] = (
                0.0, self.binarization_loss_weight)

        return loss_dict


class RADTTSLoss(torch.nn.Module):
    def __init__(self, sigma=1.0, n_group_size=1, CTC_blank_logprob=-1,
                 kl_loss_start_iter=5000, binarization_loss_weight=1.0,
                 ctc_loss_weight=0.1):
        super(RADTTSLoss, self).__init__()
        self.sigma = sigma
        self.n_group_size = n_group_size
        self.attn_loss = AttentionLoss(CTC_blank_logprob, kl_loss_start_iter,
            binarization_loss_weight, ctc_loss_weight)

    def forward(self, model_output, in_lens: SequenceLength, out_lens: SequenceLength, global_step):
        loss_dict = {}
        if len(model_output['z_mel']):
            n_elements = torch.div(out_lens.lengths.sum(),  self.n_group_size, rounding_mode="floor")
            mask = get_mask_from_lengths(torch.div(out_lens.lengths, self.n_group_size, rounding_mode="floor"))
            mask = mask[:, None].float()
            n_dims = model_output['z_mel'].size(1)
            loss_mel, loss_prior_mel = compute_flow_loss(
                model_output['z_mel'], model_output['log_det_W_list'],
                model_output['log_s_list'], n_elements, n_dims, mask,
                self.sigma)
            loss_dict['loss_mel'] = (loss_mel, 1.0)  # loss, weight
            loss_dict['loss_prior_mel'] = (loss_prior_mel, 0.0)

        attn_loss_dict = self.attn_loss(
            model_output['attn'], model_output['attn_soft'],
            model_output['attn_logprob'], global_step, in_lens.lengths, out_lens.lengths)

        loss_dict.update(attn_loss_dict)
        return loss_dict

class AttributeBCELoss(torch.nn.Module):
    def __init__(self, prefix: Optional[str]=None, weight=1.0):
        super(AttributeBCELoss, self).__init__()
        self.prefix = prefix
        self.weight = weight

    def forward(self, model_output, in_lens, out_lens, global_step, mask=None):
        target = model_output['x']
        prediction = model_output['x_hat']
        if mask is None:
            mask = out_lens.mask.unsqueeze(1)
        assert(len(mask.shape) == len(target.shape))
        mask = mask.bool()
        loss = F.binary_cross_entropy_with_logits(prediction[mask],
                                                  target[mask],
                                                  reduction='sum')/mask.sum()
        loss_dict = {self.prefix + 'loss': (loss, self.weight)}
        return loss_dict


class AttributeRegressionLoss(torch.nn.Module):
    def __init__(self, prefix: Optional[str]=None, weight=1.0):
        super(AttributeRegressionLoss, self).__init__()
        self.prefix = prefix
        self.weight = weight

    def forward(self, model_output, in_lens, out_lens, global_step, mask=None):
        target = model_output['x']
        prediction = model_output['x_hat']

        if mask is None:
            mask = out_lens.mask.unsqueeze(1)
        assert(len(mask.shape) == len(target.shape))
        mask = mask.bool()
        loss = F.mse_loss(prediction[mask], target[mask], reduction='sum')/mask.sum()
        loss_dict = {self.prefix + 'loss': (loss, self.weight)}
        return loss_dict


class AttributeMinCrossCovarianceRegLoss(torch.nn.Module):
    def __init__(self, attr_name1, attr_name2, 
                loss_cross_covariance_weight,
                gamma=1):
        super(AttributeMinCrossCovarianceRegLoss, self).__init__()
        self.attr_name1 = attr_name1
        self.attr_name2 = attr_name2
        self.loss_cross_covariance_weight = float(loss_cross_covariance_weight)

    def forward(self, batch_attr1, batch_attr2, 
                attr1_embeddings, attr2_embeddings):
        
        if attr1_embeddings is not None:
            attr1_embeddings = attr1_embeddings.weight
        else:
            attr1_embeddings = batch_attr1

        if attr2_embeddings is not None:
            attr2_embeddings = attr2_embeddings.weight
        else:
            attr2_embeddings = batch_attr2

        n_dims_attr1 = attr1_embeddings.shape[1]
        n_dims_attr2 = attr2_embeddings.shape[1]

        n_minibatch = batch_attr1.shape[0]

        # mean normalize the batch wrt the base embeddings for each attr
        attr1_minibatch_norm_embs = batch_attr1 - attr1_embeddings.mean(dim=0, keepdim=True)
        attr2_minibatch_norm_embs = batch_attr2 - attr2_embeddings.mean(dim=0, keepdim=True)

        # cross covariance matrix for the minibatch
        cross_cov_embs = (attr1_minibatch_norm_embs.T @ attr2_minibatch_norm_embs) / (n_minibatch - 1)
        
        # cross covariance loss; loss = 1/(d1*d2)*(sum_i,j)[cov(x_i, x_j)^2]
        # the 1/(d1*d2) factor is to normalize wrt number of dimensions used. 
        # the cross covariance matrix is of shape [d1, d2] and we normalize to
        # reduce the effects of those specific dimension numbers used.
        cross_cov_loss = (cross_cov_embs).pow_(2).sum().div(n_dims_attr1*n_dims_attr2)

        loss_dict = {
        "loss_{}-{}_cross_covariance".format(self.attr_name1, self.attr_name2): (cross_cov_loss, self.loss_cross_covariance_weight)
        }

        return loss_dict


class AttributeInvariance(torch.nn.Module):
    def __init__(self, name, loss_invariance_weight):
        super(AttributeInvariance, self).__init__()
        self.loss_invariance_weight = float(loss_invariance_weight)
        self.name = name

    def forward(self, embeddings, transformed_embeddings):
        # invariance is just mse, no normalization required
        inv_loss = torch.nn.MSELoss(reduction='mean')(embeddings, transformed_embeddings)
        loss_dict = {
            "loss_{}_invariance".format(self.name): (inv_loss, self.loss_invariance_weight),
        }
        return loss_dict
        

class VarianceCovarianceEmbeddingRegLoss(torch.nn.Module):
    def __init__(self, name, loss_variance_weight, 
                loss_covariance_weight,
                gamma=1):
        super(VarianceCovarianceEmbeddingRegLoss, self).__init__()
        self.name = name
        self.loss_variance_weight = float(loss_variance_weight)
        self.loss_covariance_weight = float(loss_covariance_weight)
        self.gamma = gamma

    def forward(self, embeddings, lens=None):
        if 'weight' in embeddings.__dict__['_parameters']:
            embs = embeddings.weight
        else:
            embs = embeddings

        n_attributes = embs.shape[0]
        n_dims = embs.shape[1]

        # variance loss
        std_embs = torch.sqrt(embs.var(dim=0) + 1e-04)
        std_loss = torch.mean(torch.relu(self.gamma - std_embs))
        
        # covariance loss
        embs = embs - embs.mean(dim=0, keepdim=True)
        cov_embs = (embs.T @ embs) / (n_attributes - 1)
        mask = ~torch.eye(cov_embs.shape[0]).bool()
        cov_loss = (cov_embs[mask]).pow_(2).sum().div(n_dims)

        loss_dict = {
            "loss_{}_variance".format(self.name): (std_loss, self.loss_variance_weight),
            "loss_{}_covariance".format(self.name): (cov_loss, self.loss_covariance_weight)
        }
        return loss_dict



class RADTTSDeterministicLoss(torch.nn.Module):
    def __init__(self, CTC_blank_logprob=-1, kl_loss_start_iter=5000,
                 binarization_loss_weight=1.0, ctc_loss_weight=0.1):
        super(RADTTSDeterministicLoss, self).__init__()
        self.attn_loss = AttentionLoss(CTC_blank_logprob, kl_loss_start_iter,
            binarization_loss_weight, ctc_loss_weight)

    def forward(self, model_output, in_lens, out_lens, global_step):
        loss_dict = {}
        if len(model_output['mel']):
            mask = get_mask_from_lengths(out_lens)
            mask = mask[:, None].float()
            mel, mel_hat = model_output['mel'], model_output['mel_hat']
            n_dims = mel.shape[1]
            loss_mel = (((mel - mel_hat).abs() * mask).sum() /
                         (n_dims * mask.sum()))
            loss_dict['mel_mae_loss'] = (loss_mel, 1.0)

        attn_loss_dict = self.attn_loss(
            model_output['attn'], model_output['attn_soft'],
            model_output['attn_logprob'], global_step, in_lens, out_lens)

        for k, v in attn_loss_dict.items():
            loss_dict[k] = v

        return loss_dict

class RADTTSDiffusionLoss(torch.nn.Module):
    def __init__(self, CTC_blank_logprob=-1, kl_loss_start_iter=5000,
                 binarization_loss_weight=1.0, ctc_loss_weight=0.1):

        super(RADTTSDiffusionLoss, self).__init__()
        self.attn_loss = AttentionLoss(CTC_blank_logprob, kl_loss_start_iter,
            binarization_loss_weight, ctc_loss_weight)

    def forward(self, model_output, in_lens, out_lens, global_step):
        """ loss function for diffusion model. note that mel and mel_hat refer
        to noise and noise prediction respectively """
        loss_dict = {}
        if len(model_output['noise_hat']):
            mask = out_lens.mask
            mask = mask[:, None].float()
            noise, noise_hat = model_output['noise'], model_output['noise_hat']
            n_dims = noise.shape[1]
            loss_noise = (((noise - noise_hat)**2 * mask).sum() /
                         (n_dims * mask.sum()))
            loss_dict['noise_mse_loss'] = (loss_noise, 1.0)

        attn_loss_dict = self.attn_loss(
            model_output['attn'], model_output['attn_soft'],
            model_output['attn_logprob'], global_step, in_lens.lengths, out_lens.lengths)

        for k, v in attn_loss_dict.items():
            loss_dict[k] = v

        return loss_dict

class RADTTSE2EGANLoss(torch.nn.Module):
    def __init__(self, CTC_blank_logprob=-1, kl_loss_start_iter=5000,
                 binarization_loss_weight=1.0, ctc_loss_weight=0.1,
                 stft_loss_sc_weight=1.0, stft_loss_mag_weight=1.0,
                 fft_lengths=[1024, 2048, 512, 64, 8192],
                 hop_lengths=[120, 240, 50, 10, 2000],
                 win_lengths=[600, 1200, 240, 50, 8000],
                 window="hann_window", sampling_rate=22050, a_weighting=True):
        super(RADTTSE2EGANLoss, self).__init__()

        self.stft_loss_sc_weight = stft_loss_sc_weight
        self.stft_loss_mag_weight = stft_loss_mag_weight
        self.attn_loss = AttentionLoss(CTC_blank_logprob, kl_loss_start_iter,
            binarization_loss_weight, ctc_loss_weight)
        self.multires_stft_loss_fn = MultiResolutionSTFTLoss(
            fft_lengths, hop_lengths, win_lengths, window, sampling_rate,
            a_weighting)

    def forward(self, model_output, audio, audio_lens, in_lens, out_lens,
                global_step, msd_g, msd_fmap_r, msd_fmap_g, mpd_g, mpd_fmap_r,
                mpd_fmap_g):
        loss_dict = {}

        audio_hat = model_output['audio_hat']
        min_audio_len = min(audio.shape[2], audio_hat.shape[2])
        audio = audio[..., :min_audio_len]
        audio_hat = audio_hat[..., :min_audio_len]

        max_len = audio_lens.max()
        len_ratios = audio_lens / max_len

        # spectrogram reconstruction loss
        stft_loss_sc, stft_loss_mag = self.multires_stft_loss_fn(
            audio, audio_hat, len_ratios)

        loss_dict['stft_loss_sc'] = (stft_loss_sc, self.stft_loss_sc_weight)
        loss_dict['stft_loss_mag'] = (stft_loss_mag, self.stft_loss_mag_weight)

        """
        # adversarial and feature embedding loss
        max_len = float(audio.shape[2])
        len_ratios = audio_lens / max_len

        loss_dict_gan = self.compute_generator_loss(msd_g, msd_fmap_r,
            msd_fmap_g, mpd_g, mpd_fmap_r, mpd_fmap_g, len_ratios)
        for k, v in loss_dict_gan .items():
            loss_dict[k] = v
        """

        loss_dict_attn  = self.attn_loss(
            model_output['attn'], model_output['attn_soft'],
            model_output['attn_logprob'], global_step, in_lens, out_lens)

        for k, v in loss_dict_attn.items():
            loss_dict[k] = v

        return loss_dict

    def compute_generator_loss(self, msd_g, msd_fmap_r, msd_fmap_g, mpd_g,
                               mpd_fmap_r, mpd_fmap_g, len_ratios=None):
        loss_fm_msd = gan_feature_loss(msd_fmap_r, msd_fmap_g, len_ratios)
        loss_gen_msd = gan_generator_loss(msd_g, len_ratios)[0]
        loss_fm_mpd, loss_gen_msd = 0.0, 0.0
        if mpd_g is not None:
            loss_fm_mpd = self.feature_loss(mpd_fmap_r, mpd_fmap_g, len_ratios)
            loss_gen_mpd = self.generator_loss(mpd_g, len_ratios)

        loss_dict = {
            'gen_adv_loss': (loss_gen_msd + loss_gen_mpd, 1.0),
            'gen_fm_loss': (loss_fm_msd + loss_fm_mpd, 1.0)
        }

        return loss_dict

    def compute_discriminator_loss(self, msd_real, msd_gen, mpd_real=None,
                                   mpd_gen=None, audio_lens=None):
        # (refactor) get len ratios for masking loss
        max_len = audio_lens.max()
        len_ratios = audio_lens / max_len
        loss_msd = gan_discriminator_loss(msd_real, msd_gen, len_ratios)[0]

        # mpd loss
        loss_mpd = 0.0
        if mpd_real is not None:
            loss_mpd = gan_discriminator_loss(mpd_real, mpd_gen, len_ratios)[0]

        loss_dict = {
            'disc_msd_loss': (loss_msd, 1.0),
            'disc_mpd_loss': (loss_mpd, 1.0)
        }


class RADMMMLoss(torch.nn.Module):
    def __init__(self, sigma=1.0, n_group_size=1, CTC_blank_logprob=-1,
                 kl_loss_start_iter=5000, binarization_loss_weight=1.0,
                 ctc_loss_weight=0.1,
                 use_spk_embed_reg=False, use_accent_embed_reg=False,
                 reg_loss_config=None, use_spk_accent_cross_covariance=False,
                 cross_reg_loss_config=None):
        super(RADMMMLoss, self).__init__()
        self.sigma = sigma
        self.n_group_size = n_group_size
        self.use_spk_embed_reg = bool(use_spk_embed_reg)
        self.use_accent_embed_reg = bool(use_accent_embed_reg)
        self.use_spk_accent_cross_covariance = bool(use_spk_accent_cross_covariance)
        self.reg_loss_config = reg_loss_config
        self.cross_reg_loss_config = cross_reg_loss_config
        self.attn_loss = AttentionLoss(CTC_blank_logprob, kl_loss_start_iter,
            binarization_loss_weight, ctc_loss_weight)
        
    def forward(self, model_output, in_lens: SequenceLength, out_lens: SequenceLength, global_step):
        loss_dict = {}
        if len(model_output['z_mel']):
            n_elements = torch.div(out_lens.lengths.sum(),  self.n_group_size, rounding_mode="floor")
            mask = get_mask_from_lengths(torch.div(out_lens.lengths, self.n_group_size, rounding_mode="floor"))
            mask = mask[:, None].float()
            n_dims = model_output['z_mel'].size(1)
            loss_mel, loss_prior_mel = compute_flow_loss(
                model_output['z_mel'], model_output['log_det_W_list'],
                model_output['log_s_list'], n_elements, n_dims, mask,
                self.sigma)
            loss_dict['loss_mel'] = (loss_mel, 1.0)  # loss, weight
            loss_dict['loss_prior_mel'] = (loss_prior_mel, 0.0)

        attn_loss_dict = self.attn_loss(
            model_output['attn'], model_output['attn_soft'],
            model_output['attn_logprob'], global_step, in_lens.lengths, out_lens.lengths)

        loss_dict.update(attn_loss_dict)
        
        return loss_dict