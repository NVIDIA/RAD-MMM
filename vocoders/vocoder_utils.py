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
import sys
import os
sys.path.append(os.path.join(sys.path[0],'vocoders/waveglow_for_LIMMITS23/tacotron2'))
sys.path.append(os.path.join(sys.path[0],'vocoders/'))
import json
import torch
import numpy as np

from hifigan_models import Generator
from hifigan_env import AttrDict
from hifigan_denoiser import Denoiser as HG_Denoiser

MAX_WAV_VALUE = 32768.0

def get_audio_for_mels(mels, vocoder_type, vocoder, 
                        denoiser=None,
                        sigma=0.667,
                        denoising_strength=0.001):

    # print(f'ds={denoising_strength}_voc_s={sigma}')
    if vocoder_type == 'hifigan':
        with torch.no_grad():
            audio = vocoder(mels[0].cpu()).float()
            audio_denoised = denoiser(
                            audio, strength=denoising_strength)[0].float()
        audio_denoised = audio_denoised[0].detach().cpu().numpy()
        audio_denoised = audio_denoised / np.abs(audio_denoised).max()
        return audio_denoised
    elif vocoder_type == 'waveglow':
        with torch.no_grad():
            audio = vocoder.infer(mels, sigma=sigma).float()
            audio = denoiser(audio, denoising_strength)
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        # audio = audio.astype('int16')
        
        audio = audio / np.abs(audio).max()
        return audio
    else:
        print('vocoder not supported')
        sys.exit(1)

def get_vocoder_map(vocoder_checkpoints):
    vocoder_map = json.load(vocoder_checkpoints)
    return vocoder_map

def get_vocoder(vocoder_type, 
                vocoder_map,
                speaker_name=None,
                vocoder_config_path=None, 
                vocoder_checkpoint_path=None, 
                to_cuda=False):
    if vocoder_map is not None:
        # load speaker specific vocoders
        assert speaker_name is not None
        assert speaker_name.lower() in vocoder_map.keys()
        vocoder_config_path = vocoder_map[speaker_name.lower()]['vocoder_config_path']
        vocoder_checkpoint_path = vocoder_map[speaker_name.lower()]['vocoder_checkpoint_path']

    if vocoder_type == 'hifigan':
        print('HIFIGAN loading...')
        return load_hifigan_vocoder(vocoder_checkpoint_path,
                                    vocoder_config_path,
                                    to_cuda=to_cuda,
                                    vocoder_name=vocoder_type)
    elif vocoder_type == 'waveglow':
        return load_waveglow_vocoder(vocoder_checkpoint_path,
                                    vocoder_config_path,
                                    to_cuda=True)
    else:
        print('vocoder not supported yet')
        return None

def get_speaker_specific_vocoder(speaker_name, vocoder_type,
                                vocoder_config, vocoders_checkpoints):
    vocoder_map = get_vocoder_map(vocoders_checkpoints)
    if speaker_name in vocoder_map:
        vocoder_config, vocoder_path = vocoder_map[speaker_name]
    else:
        vocoder_config, vocoder_path = vocoder_map['universal']
    return get_vocoder(vocoder_type, vocoder_config, vocoder_path)


def load_hifigan_vocoder(vocoder_path, config_path, to_cuda=True, vocoder_name='hifigan'):
    with open(config_path) as f:
        data_vocoder = f.read()
    config_vocoder = json.loads(data_vocoder)
    h = AttrDict(config_vocoder)
    if 'blur' in vocoder_path:
        config_vocoder['gaussian_blur']['p_blurring'] = 0.5
    else:
        if 'gaussian_blur' in config_vocoder:
            config_vocoder['gaussian_blur']['p_blurring'] = 0.0
        else:
            config_vocoder['gaussian_blur'] = {'p_blurring': 0.0}
            h['gaussian_blur'] = {'p_blurring': 0.0}

    state_dict_g = torch.load(vocoder_path, map_location='cpu')['generator']

    # load hifigan
    vocoder = Generator(h)
    vocoder.load_state_dict(state_dict_g)
    denoiser = HG_Denoiser(vocoder)
    
    if to_cuda:
        vocoder.cuda()
        denoiser.cuda()
    
    vocoder.eval()
    denoiser.eval()

    return vocoder, denoiser

def load_waveglow_vocoder(vocoder_path, config_path, to_cuda=True):
    waveglow = torch.load(vocoder_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    denoiser = Denoiser(waveglow)
    if to_cuda:
        waveglow.cuda()
        denoiser.cuda()
    waveglow.eval()
    
    return waveglow, denoiser