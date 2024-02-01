
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
# Original Authors: Rohan Badlani (rbadlani@nvidia.com) and Rafael Valle 
# Uses praat (https://www.fon.hum.uva.nl/praat/) for wave transforms

import numpy as np
import ast
import torch
from torch.nn import functional as F

import parselmouth
from parselmouth.praat import call as praat_call


def scale_formant(audio, sampling_rate, scale, f0_min=80, f0_max=600):
    # https://www.fon.hum.uva.nl/praat/manual/Sound__Change_speaker___.html
    sound = parselmouth.Sound(audio, sampling_frequency=sampling_rate)
    audio_alt = praat_call(sound, "Change speaker", f0_min, f0_max,
                           scale, 1.0, 1.0, 1.0)
    audio_alt = torch.from_numpy(np.array(audio_alt))[0].float()

    # avoid change in max amplitude
    max_amp = torch.abs(audio).max()
    max_amp_alt = torch.abs(audio_alt).max()
    audio_alt = audio_alt / max_amp_alt
    audio_alt = audio_alt * max_amp

    return audio_alt


def scale_pitch(audio, sampling_rate, scale, f0_min=80, f0_max=600):
    # https://www.fon.hum.uva.nl/praat/manual/Sound__Change_speaker___.html
    sound = parselmouth.Sound(audio, sampling_frequency=sampling_rate)
    audio_alt = praat_call(sound, "Change speaker", f0_min, f0_max,
                           1.0, scale, 1.0, 1.0)
    audio_alt = torch.from_numpy(np.array(audio_alt))[0].float()

    # avoid change in max amplitude
    max_amp = torch.abs(audio).max()
    max_amp_alt = torch.abs(audio_alt).max()
    audio_alt = audio_alt / max_amp_alt
    audio_alt = audio_alt * max_amp

    return audio_alt


def scale_duration(audio, sampling_rate, scale, f0_min=80, f0_max=600):
    # https://www.fon.hum.uva.nl/praat/manual/Sound__Change_speaker___.html
    sound = parselmouth.Sound(audio, sampling_frequency=sampling_rate)
    audio_alt = praat_call(sound, "Change speaker", f0_min, f0_max,
                           1.0, 1.0, 1.0, scale)
    audio_alt = torch.from_numpy(np.array(audio_alt))[0].float()

    # avoid change in max amplitude
    max_amp = torch.abs(audio).max()
    max_amp_alt = torch.abs(audio_alt).max()
    audio_alt = audio_alt / max_amp_alt
    audio_alt = audio_alt * max_amp

    return audio_alt


class WaveAugmentations(object):
    """
    This class performs augmentations over a given input audio and returns all the
    augmented audio and original audio as a list of elements. Updated speaker id is also
    returned.
    """
    def __init__(self, aug_types, aug_scales, 
                        aug_languages_applicable,
                        aug_probabilities, 
                        num_aug_in_batch=0,
                        randomize_transform=False,
                        n_augmentations=0):
        # self.aug_types = aug_types if aug_types is list else ast.literal_eval(aug_types)
        # self.aug_scales = aug_scales if aug_scales is list else ast.literal_eval(aug_scales)
        self.aug_types = aug_types
        self.aug_scales = aug_scales
        print(aug_probabilities)
        print(round(sum(aug_probabilities)), 2)
        assert(round(sum(aug_probabilities), 2) == 1.0)
        self.aug_probabilities = aug_probabilities
        
        self.aug_languages_applicable = aug_languages_applicable
        self.num_aug_in_batch = num_aug_in_batch
        self.randomize_transform = bool(randomize_transform)
        assert len(self.aug_types) == len(self.aug_scales)
        assert n_augmentations == len(self.aug_types) - 1

        self.aug_modules = {
            'none': None,
            'scale_formant': scale_formant,
            'scale_pitch': scale_pitch,
            'scale_duration': scale_duration
        }

        for aug_type in self.aug_types:
            assert(aug_type in self.aug_modules.keys())

    def __call__(self, audio, sampling_rate, speaker_id, language, speaker_id_dict):
        aug_idx = self.get_aug_idx()
        aug_type = self.aug_types[aug_idx]
        aug_scale = self.aug_scales[aug_idx]

        if aug_type == None or aug_type == "none":
            return audio, speaker_id, False

        if self.randomize_transform:
            if aug_scale > 1.0:
                low = 1.0
                high = aug_scale
            else:
                low = aug_scale
                high = 1.0

            aug_scale = np.random.uniform(low=low, high=high, size=(1,))
            aug_scale = aug_scale[0]
        # else:
        #     print('NON RANDOMIZED TRANSFORM')
            

        wave_aug_fn_ptr = self.aug_modules[aug_type]

        audio_aug = wave_aug_fn_ptr(audio, sampling_rate, aug_scale)

        # append updated spk id
        assert aug_idx >= 1
        speaker_aug_id = ((len(speaker_id_dict.keys()) * aug_idx) + speaker_id)
        aug_applied = True

        return audio_aug, speaker_aug_id, aug_applied

    def get_aug_idx(self):
        return np.random.choice(len(self.aug_probabilities), 1, p=self.aug_probabilities)[0]

    def print_settings(self):
        print("Wave Augmentation settings:")
        print("applied over ALL language(s): ")
        for i in range(len(self.aug_types)):
            print("Applying \t" + self.aug_types[i].__str__() + \
                " with scaling : %f" % (self.aug_scales[i]))