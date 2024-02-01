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

# Original Source: https://gist.github.com/amiasato/902fc14afa37a7537386f7b0c5537741

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn as distnn

class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.

    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py

    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.

    Check pytorch's BatchNorm1d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

        self.distributed_sync = False

    def forward(self, inp, mask=None):
        """
        inp: B x C x D
        mask: B x 1 x D (binary)
        """
        self._check_input_dim(inp)

        exponential_average_factor = 0.0

        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        # if mask == None:
        #     assert(lengths is not None)
        #     mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = mask.sum()

        sync_results = self.training and dist.is_initialized() and self.distributed_sync
        mask = mask.expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            sum_x = (mask * inp).sum([0, 2])
            sum_xsq = (mask * inp ** 2).sum([0, 2])
            if sync_results:
                # concat N at the end so we can all_reduce everything at once
                params_to_sync = torch.stack([sum_x, sum_xsq, n+torch.zeros_like(sum_x)])
                distnn.all_reduce(params_to_sync, op=dist.ReduceOp.SUM)

                sum_x = params_to_sync[0]
                sum_xsq = params_to_sync[1]
                n = params_to_sync[2,0]
            mean = sum_x / n
            exp_xsq = sum_xsq / n
            # Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = exp_xsq - mean ** 2

            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var

        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        mtsr = inp[0,:,:(mask>0)[0].sum()].mean()
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp
