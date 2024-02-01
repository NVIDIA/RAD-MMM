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

# Original Copyright 2019 Tomoki Hayashi
# Original License: (https://opensource.org/licenses/MIT)
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
"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F

from distutils.version import LooseVersion
from common import get_mask_from_lengths

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False)
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class ComplexSTFTLoss(torch.nn.Module):
    """Complex STFT loss module. Loss is magnitude of error vector between
    target and predicted STFTs."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window",
        sampling_rate=22050, a_weighting=False
        ):
        """Initialize STFT loss module."""
        super(ComplexSTFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, y_hat, y):
        """Calculate forward propagation.

        Args:
            y_hat (Tensor): Predicted signal in time domain.
            y (Tensor): Ground truth signal in time domain.

        Returns:
            Tensor: Complex STFT loss value.
        """
        Y = torch.stft(y, self.fft_size, self.shift_size, self.win_length, self.window,
                return_complex=True)
        Y_hat = torch.stft(y_hat, self.fft_size, self.shift_size, self.win_length, self.window,
                return_complex=True)
        loss = torch.sum(torch.log(torch.sqrt(torch.clamp((Y - Y_hat)**2, min=1e-7))))
        return loss


class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag, len_ratios=None):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.

        """
        if len_ratios is None:
            return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        else:
            loss = 0.0
            lens = (len_ratios * y_mag.shape[1]).ceil().long()
            for i in range(len(y_mag)):
                y_i, x_i = y_mag[i, :lens[i]], x_mag[i, :lens[i]]
                loss_i = (
                    torch.norm(y_i - x_i, dim=1, p="fro") /
                    torch.norm(y_i, dim=1, p="fro"))
                loss += loss_i.sum()
            loss = loss / lens.sum()
            return loss


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initialize STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag, len_ratios):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        if len_ratios is None:
            return F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        else:
            lens = (len_ratios * y_mag.shape[1]).ceil().long()
            b, t, d = y_mag.shape
            lens = (len_ratios * y_mag.shape[1]).ceil().long()
            mask = get_mask_from_lengths(lens)[..., None]
            loss = (torch.log(y_mag) - torch.log(x_mag)).abs()
            loss = (loss * mask).sum() / (mask.sum() * d)
            return loss


class AWeightedLogSTFTMagnitudeLoss(torch.nn.Module):
    """A-weighted STFT magnitude loss module."""
    def __init__(self, sampling_rate=22050, fft_size=1024):
        """Initialize STFT magnitude loss module."""
        super(AWeightedLogSTFTMagnitudeLoss, self).__init__()
        self.weights = AWeightedLogSTFTMagnitudeLoss.calc_a_weights(
            sampling_rate, fft_size)
        self.weights += 1e-6 # avoid issues with log

        # FIXME
        self.weights = 1.0

    def forward(self, x_mag, y_mag, len_ratios=None):
        """Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.

        """
        error = self.weights * torch.abs(
            torch.log(y_mag + 1) - torch.log(x_mag + 1))

        if len_ratios is None:
            error = torch.mean(error)
        else:
            b, t, d = y_mag.shape
            lens = (len_ratios * y_mag.shape[1]).ceil().long()
            mask = get_mask_from_lengths(lens)[..., None]
            error = (error * mask).sum() / (mask.sum() * d)
        return error

    @staticmethod
    def calc_a_weights(sampling_rate, fft_size):
        f = torch.linspace(0, sampling_rate/2.0, fft_size // 2 + 1)
        return (12194**2 * f**4) / ( (f**2 + 20.6**2) * torch.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2)) * (f**2 + 12194**2) )


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600,
        window="hann_window", sampling_rate=22050, a_weighting=False,
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        if a_weighting:
            self.log_stft_magnitude_loss = AWeightedLogSTFTMagnitudeLoss(
                sampling_rate, self.fft_size)
        else:
            self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x, y, lens):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.

        """
        x_mag = stft(
            x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(
            y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag, lens)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag, lens)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        sampling_rate=22050,
        a_weighting=False,
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(
                fs, ss, wl, window, sampling_rate=sampling_rate,
                a_weighting=a_weighting)]

    def forward(self, x, y, lens=None):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y, lens)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss

class MultiResolutionComplexSTFTLoss(torch.nn.Module):
    """Multi resolution complex STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        sampling_rate=22050,
        a_weighting=False
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [ComplexSTFTLoss(fs, ss, wl, window,
                sampling_rate=sampling_rate)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T) or (B, #subband, T).
            y (Tensor): Groundtruth signal (B, T) or (B, #subband, T).

        Returns:
            Tensor: Multi resolution log STFT magnitude loss value.

        """
        if len(x.shape) == 3:
            x = x.view(-1, x.size(2))  # (B, C, T) -> (B x C, T)
            y = y.view(-1, y.size(2))  # (B, C, T) -> (B x C, T)
        complex_loss = 0.0
        for f in self.stft_losses:
            complex_loss += f(x, y)
        complex_loss /= len(self.stft_losses)

        return complex_loss
