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

# Based on https://github.com/NVIDIA/flowtron/blob/master/data.py
# Original license text:
###############################################################################
#
#  Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import os
import argparse
import json
import numpy as np
import lmdb
import pickle as pkl
# from scripts.phonemize_text import phonemize_text
import torch
import torch.utils.data
import torch.nn.functional as F
from scipy.io.wavfile import read
from audio_processing import TacotronSTFT
from scipy.stats import betabinom
from librosa import pyin
from common import update_params
from scipy.ndimage import distance_transform_edt as distance_transform
from scipy.ndimage import zoom
from typing import Optional
from functools import lru_cache
from wave_transforms import WaveAugmentations


class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.
    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    Source: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/FastPitch/fastpitch/data_function.py
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = lru_cache(beta_binomial_prior_distribution)

    @staticmethod
    def round(val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, p_count, m_count):
        bh = self.round(m_count, to=self.round_mel_len_to)
        bw = self.round(p_count, to=self.round_text_len_to)
        anchor = self.bank(bw, bh)
        ret = zoom(self.bank(bw, bh), zoom=(m_count / bh, p_count / bw), order=1, mode='nearest')
        #print(numpyt)
        #ret2 = F.interpolate(anchor_t.cuda(), scale_factor = (m_count / bh, p_count / bw), mode='bilinear', align_corners=True)
        # renormalize
        ret = ret/ret.sum(1, keepdims=True)
        assert ret.shape[0] == m_count, ret.shape
        assert ret.shape[1] == p_count, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count,
                                     scaling_factor=0.05):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling_factor*i, scaling_factor*(M+1-i)
        rv = betabinom(P-1, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return np.array(mel_text_probs)


def load_wav_to_torch(full_path):
    """ Loads wavdata into torch array """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(np.array(data)).float(), sampling_rate


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, tp, dataloader_type, filter_length, hop_length, win_length,
                 sampling_rate, n_mel_channels, mel_fmin, mel_fmax, f0_min,
                 f0_max, max_wav_value, use_f0, use_energy_avg, use_log_f0,
                 use_scaled_energy, symbol_set, cleaner_names, heteronyms_path,
                 phoneme_dict_path, p_phoneme, handle_phoneme='word',
                 handle_phoneme_ambiguous='ignore', speaker_ids=None,
                 accent_ids=None,
                 include_speakers=None, n_frames=-1,
                 use_attn_prior_masking=True, prepend_space_to_text=True,
                 append_space_to_text=True, add_bos_eos_to_text=False,
                 betabinom_cache_path="", betabinom_scaling_factor=0.05,
                 lmdb_cache_path: Optional[str]=None, dur_min=None, dur_max=None,
                 combine_speaker_and_emotion=False, distance_tx_unvoiced=False,
                 mel_noise_scale=0.0, speaker_map=None, accent_map=None,
                 use_prior_interpolator:bool=True,
                 phonemizer_cfg: Optional[str]=None, return_audio=False,
                 use_multilingual_model=False,
                 g2p_type='phonemizer',
                 use_wave_augmentations=False,
                 wave_aug_config=None,
                 speaker_stats_path=None,
                 f0_pred_type="norm_log_f0",
                 include_emotions=None,
                 phonemizer_dicts=None):
        super().__init__()
        self.f0_pred_type = f0_pred_type
        self.dataloader_type = dataloader_type
        self.combine_speaker_and_emotion = combine_speaker_and_emotion
        self.return_audio = return_audio
        self.max_wav_value = max_wav_value
        self.audio_lmdb_dict = {}  # dictionary of lmdbs for audio data
        self.multilingual_model = bool(use_multilingual_model)
        self.data = self.load_data(datasets)
        self.distance_tx_unvoiced = distance_tx_unvoiced
        self.use_prior_interpolator = use_prior_interpolator
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 n_mel_channels=n_mel_channels,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)
        self.mel_noise_scale = mel_noise_scale
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.use_f0 = use_f0
        self.use_log_f0 = use_log_f0
        self.use_energy_avg = use_energy_avg
        self.use_scaled_energy = use_scaled_energy
        self.sampling_rate = sampling_rate
        self.tp = tp
        self.dur_min = dur_min
        self.dur_max = dur_max
        if self.use_prior_interpolator:
            self.prior_interpolator = BetaBinomialInterpolator()
        if speaker_ids is None or speaker_ids == '':
            self.speaker_ids = self.create_attribute_lookup_table(self.data)
        else:
            self.speaker_ids = speaker_ids
            print("Using provided Speaker IDS map", self.speaker_ids)
    
        if accent_ids is None or accent_ids == '':
            self.accent_ids = self.create_attribute_lookup_table(self.data, 'language')
        else:
            self.accent_ids = accent_ids
            print("Using provided Accent IDS map", self.accent_ids)

        print("Number of files", len(self.data))
        if include_speakers is not None:
            for (speaker_set, include) in include_speakers:
                self.filter_by_speakers_(speaker_set, include)
            print("Number of files after speaker filtering", len(self.data))

        if include_emotions is not None:
            for (emotion_set, include) in include_emotions:
                self.filter_by_emotions_(emotion_set, include)
            print("Number of files after emotion filtering", len(self.data))

        if dur_min is not None and dur_max is not None:
            self.filter_by_duration_(dur_min, dur_max)
            print("Number of files after duration filtering", len(self.data))

        if use_wave_augmentations:
            # wave augmentations are enabled. update speaker set
            assert wave_aug_config is not None
            self.wave_augmentations = WaveAugmentations(**wave_aug_config)
            self.wave_augmentations.print_settings()
        else:
            self.wave_augmentations = None
            print("Dataloader initialized with no augmentations")

        if phonemizer_dicts is not None:
            self.phonemizer_dicts = phonemizer_dicts
            print(self.phonemizer_dicts)
            import pdb
            pdb.set_trace()

        self.use_attn_prior_masking = bool(use_attn_prior_masking)
        self.prepend_space_to_text = bool(prepend_space_to_text)
        self.append_space_to_text = bool(append_space_to_text)
        self.betabinom_cache_path = betabinom_cache_path
        self.betabinom_scaling_factor = betabinom_scaling_factor
        self.lmdb_cache_path = lmdb_cache_path
        if self.lmdb_cache_path is not None and self.lmdb_cache_path != "":
            self.cache_data_lmdb = lmdb.open(
                self.lmdb_cache_path, readonly=True, max_readers=1024,
                lock=False).begin()

        # make sure caching path exists
        if not os.path.exists(self.betabinom_cache_path):
            os.makedirs(self.betabinom_cache_path)
        self.speaker_map = speaker_map
        self.accent_map = accent_map

        self.speaker_stats_path = speaker_stats_path
        if self.speaker_stats_path is not None and \
            self.speaker_stats_path != '':
            
            with open(speaker_stats_path) as f:
                data = f.read()
            loaded_speaker_stats = json.loads(data)

            # transform keys to lowercase for easy match
            self.speaker_stats = {}
            for key, value in loaded_speaker_stats.items():
                self.speaker_stats[key.lower()] = value

            print(self.speaker_stats)
        else:
            self.speaker_stats = None

    def load_data(self, datasets, split='|'):
        dataset = []
        for dset_name, dset_dict in datasets.items():
            folder_path = dset_dict['basedir']
            sampling_rate = dset_dict['sampling_rate']
            filelist_basedir = dset_dict['filelist_basedir']
            filename = dset_dict['filelist']
            filelist_path = os.path.join(filelist_basedir, filename)
            if self.multilingual_model:
                language = dset_dict['language']
            else:
                language = 'en_US' # default to english

            phonemized = False
            print(dset_dict)
            if 'phonemized' in dset_dict:
                phonemized = bool(dset_dict['phonemized'])

            audio_lmdb_key = None
            if 'lmdbpath' in dset_dict.keys() and len(dset_dict['lmdbpath']) > 0:
                self.audio_lmdb_dict[dset_name] = lmdb.open(
                    dset_dict['lmdbpath'], readonly=True, max_readers=256,
                    lock=False).begin()
                audio_lmdb_key = dset_name

            wav_folder_prefix = os.path.join(folder_path, sampling_rate)
            with open(filelist_path, encoding='utf-8') as f:
                data = [line.strip().split(split) for line in f]
            print(f'processing file: {filelist_path}')
            for d in data:
                dataset.append(
                    {'audiopath': os.path.join(wav_folder_prefix, d[0]),
                     'text': d[1],
                     'speaker': d[2] + '-' + d[3] if self.combine_speaker_and_emotion else d[2],
                     'emotion': d[3],
                     'duration': float(d[4]),
                     'lmdb_key': audio_lmdb_key,
                     'language': language,
                     'phonemized': phonemized
                    })
        return dataset

    def filter_by_speakers_(self, speakers, include=True):
        print("Include spaker {}: {}".format(speakers, include))
        if include:
            self.data = [x for x in self.data if x['speaker'] in speakers]
        else:
            self.data = [x for x in self.data if x['speaker'] not in speakers]

    def filter_by_emotions_(self, emotions, include=True):
        emotions = [e.lower() for e in emotions]
        print("Include emotion {}: {}".format(emotions, include))
        if include:
            self.data = [x for x in self.data if x['emotion'].lower() in emotions]
        else:
            self.data = [x for x in self.data if x['emotion'].lower() not in emotions]

    def filter_by_duration_(self, dur_min, dur_max):
        self.data = [x for x in self.data
                     if x['duration'] >= dur_min and x['duration'] <= dur_max]

    def create_attribute_lookup_table(self, data, attribute='speaker'):
        attribute_ids = np.sort(np.unique([x[attribute] for x in data]))
        d = {attribute_ids[i]: i for i in range(len(attribute_ids))}
        print(f'Number of {attribute}s : {len(d)}')
        print(f'{attribute} ids: {d}')
        return d

    def load_speaker_stats(self, speaker_name):
        # print(speaker_name)
        if self.speaker_stats is not None and speaker_name.lower() in self.speaker_stats.keys():
            return self.speaker_stats[speaker_name.lower()]
        else:
            return None

    def f0_normalize(self, x):
        if self.use_log_f0:
            mask = x >= self.f0_min
            x[mask] = torch.log(x[mask])
            x[~mask] = 0.0

        return x

    def f0_denormalize(self, x):
        if self.use_log_f0:
            log_f0_min = np.log(self.f0_min)
            mask = x >= log_f0_min
            x[mask] = torch.exp(x[mask])
            x[~mask] = 0.0
        x[x <= 0.0] = 0.0

        return x

    def energy_avg_normalize(self, x):
        if self.use_scaled_energy:
            x = (x + 20.0) / 20.0
        return x

    def energy_avg_denormalize(self, x):
        if self.use_scaled_energy:
            x = x * 20.0 - 20.0
        return x

    def get_f0_pvoiced(self, audio, sampling_rate=22050, frame_length=1024,
                       hop_length=256, f0_min=100, f0_max=300):

        audio_norm = audio / self.max_wav_value
        f0, voiced_mask, p_voiced = pyin(
            audio_norm, f0_min, f0_max, sampling_rate,
            frame_length=frame_length, win_length=frame_length // 2,
            hop_length=hop_length)
        f0[~voiced_mask] = 0.0
        f0 = torch.FloatTensor(f0)
        p_voiced = torch.FloatTensor(p_voiced)
        voiced_mask = torch.FloatTensor(voiced_mask)
        return f0, voiced_mask, p_voiced

    def get_energy_average(self, mel):
        energy_avg = mel.mean(0)
        energy_avg = self.energy_avg_normalize(energy_avg)
        return energy_avg

    def get_mel(self, audio):
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)
        if self.mel_noise_scale > 0:
            melspec += torch.randn_like(melspec) * self.mel_noise_scale
        return melspec

    def get_speaker_id(self, speaker):
        if self.speaker_map is not None and speaker in self.speaker_map:
            speaker = self.speaker_map[speaker]

        return torch.LongTensor([self.speaker_ids[speaker]])

    def get_accent_id(self, accent):
        if self.accent_map is not None and accent in self.accent_map:
            accent = self.accent_map[accent]

        return torch.LongTensor([self.accent_ids[accent]])

    def get_text(self, text, language=None, is_phonemized=False):
        text = self.tp.encode_text(text, language=language,
                                    is_phonemized=is_phonemized)
        text = torch.LongTensor(text)
        return text

    def get_attention_prior(self, n_tokens, n_frames):
        # cache the entire attn_prior by filename
        if self.use_attn_prior_masking:
            if self.use_prior_interpolator:
                 return torch.tensor(self.prior_interpolator(n_tokens, n_frames))
            filename = "{}_{}".format(n_tokens, n_frames)
            prior_path = os.path.join(self.betabinom_cache_path, filename)
            prior_path += "_prior.pth"
            if self.lmdb_cache_path is not None:
                attn_prior = pkl.loads(
                    self.cache_data_lmdb.get(prior_path.encode('ascii')))
            elif os.path.exists(prior_path):
                attn_prior = torch.load(prior_path)
            else:
                attn_prior = beta_binomial_prior_distribution(
                    n_tokens, n_frames, self.betabinom_scaling_factor)
                attn_prior = torch.tensor(attn_prior)
                torch.save(attn_prior, prior_path)
        else:
            attn_prior = torch.ones(n_frames, n_tokens)  # all ones baseline

        return attn_prior

    def __getitem__(self, index):
        data = self.data[index]
        audiopath, text = data['audiopath'], data['text']
        print(audiopath)
        speaker_id = data['speaker']
        
        # load speaker stats
        speaker_name = str(speaker_id)
        speaker_stats = self.load_speaker_stats(speaker_name)

        if self.speaker_stats is not None and self.f0_pred_type == 'norm_log_f0':
            assert speaker_stats is not None
            f0_mean_key = 'log_f0_mean'
            f0_mean_speaker = speaker_stats[f0_mean_key]
        elif self.speaker_stats is not None:
            f0_mean_key = 'f0_mean'
            assert speaker_stats is not None
            f0_mean_speaker = speaker_stats[f0_mean_key]
        else:
            f0_mean_speaker = 0.0

        if self.speaker_stats is not None and self.f0_pred_type == 'norm_log_f0':
            assert speaker_stats is not None
            f0_std_key = 'log_f0_std'
            f0_std_speaker = speaker_stats[f0_std_key]
        elif self.speaker_stats is not None:
            f0_std_key = 'f0_std'
            assert speaker_stats is not None
            f0_std_speaker = speaker_stats[f0_std_key]
        else:
            f0_std_speaker = 0.0

        if self.speaker_stats is not None:
            energy_mean_key = 'energy_mean'
            assert speaker_stats is not None
            energy_mean_speaker = speaker_stats[energy_mean_key]

            energy_std_key = 'energy_std'
            assert speaker_stats is not None
            energy_std_speaker = speaker_stats[energy_std_key]

        if energy_mean_speaker is None or \
            energy_std_speaker is None or \
                f0_mean_speaker is None or \
                    f0_std_speaker is None:
            print(f'\n\n\n{speaker_name} stats are none...\n\n\n')

        language = data['language']
        phonemized = data['phonemized']
        
        if data['lmdb_key'] is not None:
            data_dict = pkl.loads(
                self.audio_lmdb_dict[data['lmdb_key']].get(
                    audiopath.encode('ascii')))
            audio = data_dict['audio']
            sampling_rate = data_dict['sampling_rate']
        else:
            audio, sampling_rate = load_wav_to_torch(audiopath)

        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        try:
            mel = self.get_mel(audio)
        except Exception as ex:
            print(f'mel loading failed with {ex} for {audiopath}')
            return None

        f0 = None
        p_voiced = None
        voiced_mask = None
        if self.use_f0:
            filename = '_'.join(audiopath.split('/')[-3:])
            f0_path = os.path.join(self.betabinom_cache_path, filename)
            f0_path += "_f0_sr{}_fl{}_hl{}_f0min{}_f0max{}_log{}.pt".format(
                self.sampling_rate, self.filter_length, self.hop_length,
                self.f0_min, self.f0_max, self.use_log_f0)

            dikt = None
            if self.lmdb_cache_path is not None and len(self.lmdb_cache_path) > 0:
                dikt = pkl.loads(
                    self.cache_data_lmdb.get(f0_path.encode('ascii')))
                f0 = dikt['f0']
                p_voiced = dikt['p_voiced']
                voiced_mask = dikt['voiced_mask']
            elif os.path.exists(f0_path):
                try:
                    dikt = torch.load(f0_path)
                except:
                    print(f"f0 loading from {f0_path} is broken, recomputing.")

            if dikt is not None:
                f0 = dikt['f0']
                p_voiced = dikt['p_voiced']
                voiced_mask = dikt['voiced_mask']
            else:
                f0, voiced_mask, p_voiced = self.get_f0_pvoiced(
                    audio.cpu().numpy(), self.sampling_rate,
                    self.filter_length, self.hop_length, self.f0_min,
                    self.f0_max)
                print("saving f0 to {}".format(f0_path))
                torch.save({'f0': f0,
                            'voiced_mask': voiced_mask,
                            'p_voiced': p_voiced}, f0_path)
            if f0 is None:
                raise Exception("STOP, BROKEN F0 {}".format(audiopath))

            f0 = self.f0_normalize(f0)
            if self.distance_tx_unvoiced:
                mask = f0 <= 0.0
                distance_map = np.log(distance_transform(mask))
                distance_map[distance_map <= 0] = 0.0
                f0 = f0 - distance_map

        energy_avg = None
        if self.use_energy_avg:
            energy_avg = self.get_energy_average(mel)
            if self.use_scaled_energy and energy_avg.min() < 0.0:
                print(audiopath, "has scaled energy avg smaller than 0")

        speaker_id = self.get_speaker_id(speaker_id)
        accent_id = self.get_accent_id(language)
        text_encoded = self.get_text(text, language=language, is_phonemized=phonemized)

        attn_prior = self.get_attention_prior(
                text_encoded.shape[0], mel.shape[1])

        if not self.use_attn_prior_masking:
            attn_prior = None

        if self.wave_augmentations is not None:
            audio_aug, aug_speaker_id, aug_applied = self.wave_augmentations(audio,
                                                                            sampling_rate,
                                                                            speaker_id,
                                                                            language,
                                                                            self.speaker_ids)

            
            if aug_applied:
                # features need to be recomputed on augmented audio
                audio = audio_aug
                try:
                    mel = self.get_mel(audio)
                    speaker_id = aug_speaker_id

                    f0, voiced_mask, p_voiced = self.get_f0_pvoiced(
                        audio.cpu().numpy(), self.sampling_rate,
                        self.filter_length, self.hop_length, self.f0_min,
                        self.f0_max)

                    f0 = self.f0_normalize(f0)
                    if self.distance_tx_unvoiced:
                        mask = f0 <= 0.0
                        distance_map = np.log(distance_transform(mask))
                        distance_map[distance_map <= 0] = 0.0
                        f0 = f0 - distance_map

                    energy_avg = self.get_energy_average(mel)
                    if self.use_scaled_energy and energy_avg.min() < 0.0:
                        print(audiopath, "has scaled energy avg smaller than 0")

                    attn_prior = self.get_attention_prior(
                        text_encoded.shape[0], mel.shape[1])

                except Exception as ex:
                    print(f'aug audio for {audiopath} results in error. default to audio')

                
        data_dict = {'mel': mel,
                'speaker_id': speaker_id,
                'accent_id': accent_id,
                'text_raw': text,
                'language': language,
                'text_encoded': text_encoded,
                'audiopath': audiopath,
                'attn_prior': attn_prior,
                'f0': f0,
                'p_voiced': p_voiced,
                'voiced_mask': voiced_mask,
                'energy_avg': energy_avg,
                'idx': index,
                'speaker_f0_mean': f0_mean_speaker,
                'speaker_f0_std': f0_std_speaker,
                'speaker_energy_mean': energy_mean_speaker,
                'speaker_energy_std': energy_std_speaker,
                }

        if self.return_audio:
            data_dict['audio'] = audio[None] / self.max_wav_value

        return data_dict

    def __len__(self):
        return len(self.data)


class DataCollate():
    """ Zero-pads model inputs and targets given number of steps """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate from normalized data """
        # Right zero-pad all one-hot text sequences to max input length
        batch = [item for item in batch if item is not None]
        
        print(len(batch))
        if len(batch) == 0:
            return None

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['text_encoded']) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]]['text_encoded']
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mel_channels = batch[0]['mel'].size(0)
        max_target_len = max([x['mel'].size(1) for x in batch])


        # include mel padded, gate padded and speaker ids
        mel_padded = torch.FloatTensor(len(batch), num_mel_channels, max_target_len)
        mel_padded.zero_()
        audio_padded = None
        f0_padded = None
        p_voiced_padded = None
        voiced_mask_padded = None
        energy_avg_padded = None

        if 'audio' in batch[0]:
            max_audio_len = max([x['audio'].size(1) for x in batch])
            audio_padded = torch.FloatTensor(len(batch), 1, max_audio_len)
            audio_padded.zero_()
            audio_lengths = torch.LongTensor(len(batch))

        if batch[0]['f0'] is not None:
            f0_padded = torch.FloatTensor(len(batch), max_target_len)
            f0_padded.zero_()

        if batch[0]['p_voiced'] is not None:
            p_voiced_padded = torch.FloatTensor(len(batch), max_target_len)
            p_voiced_padded.zero_()

        if batch[0]['voiced_mask'] is not None:
            voiced_mask_padded = torch.FloatTensor(len(batch), max_target_len)
            voiced_mask_padded.zero_()

        if batch[0]['energy_avg'] is not None:
            energy_avg_padded = torch.FloatTensor(len(batch), max_target_len)
            energy_avg_padded.zero_()

        attn_prior_padded = torch.FloatTensor(len(batch), max_target_len, max_input_len)
        attn_prior_padded.zero_()

        if batch[0]['speaker_f0_mean'] is not None:
            f0_mean = torch.FloatTensor(len(batch))

        if batch[0]['speaker_f0_std'] is not None:
            f0_std = torch.FloatTensor(len(batch))
        
        if batch[0]['speaker_energy_mean'] is not None:
            energy_mean = torch.FloatTensor(len(batch))
        
        if batch[0]['speaker_energy_std'] is not None:
            energy_std = torch.FloatTensor(len(batch))

        output_lengths = torch.LongTensor(len(batch))
        idx = torch.LongTensor(len(batch))
        speaker_ids = torch.LongTensor(len(batch))
        accent_ids = torch.LongTensor(len(batch))
        audiopaths = []
        text_raw = []
        language = []
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]]['mel']
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)
            idx[i] = batch[ids_sorted_decreasing[i]]['idx']
            if audio_padded is not None:
                audio = batch[ids_sorted_decreasing[i]]['audio']
                audio_padded[i, :, :audio.size(1)] = audio
                audio_lengths[i] = audio.size(1)

            if energy_avg_padded is not None:
                energy_avg = batch[ids_sorted_decreasing[i]]['energy_avg']
                energy_avg_padded[i, :len(energy_avg)] = energy_avg

            if f0_padded is not None:
                f0 = batch[ids_sorted_decreasing[i]]['f0']
                f0_padded[i, :len(f0)] = f0

            if voiced_mask_padded is not None:
                voiced_mask = batch[ids_sorted_decreasing[i]]['voiced_mask']
                voiced_mask_padded[i, :len(voiced_mask)] = voiced_mask

            if p_voiced_padded is not None:
                p_voiced = batch[ids_sorted_decreasing[i]]['p_voiced']
                p_voiced_padded[i, :len(p_voiced)] = p_voiced

            speaker_ids[i] = batch[ids_sorted_decreasing[i]]['speaker_id']
            accent_ids[i] = batch[ids_sorted_decreasing[i]]['accent_id']
            audiopath = batch[ids_sorted_decreasing[i]]['audiopath']
            audiopaths.append(audiopath)
            
            curr_text_raw = batch[ids_sorted_decreasing[i]]['text_raw']
            text_raw.append(curr_text_raw)
            
            curr_language = batch[ids_sorted_decreasing[i]]['language']
            language.append(curr_language)

            cur_attn_prior = batch[ids_sorted_decreasing[i]]['attn_prior']
            if cur_attn_prior is None:
                attn_prior_padded = None
            else:
                attn_prior_padded[i, :cur_attn_prior.size(0), :cur_attn_prior.size(1)] = cur_attn_prior

            if batch[ids_sorted_decreasing[i]]['speaker_f0_mean'] is not None:
                f0_mean[i] = batch[ids_sorted_decreasing[i]]['speaker_f0_mean']

            if batch[ids_sorted_decreasing[i]]['speaker_f0_std'] is not None:
                f0_std[i] = batch[ids_sorted_decreasing[i]]['speaker_f0_std']

            if batch[ids_sorted_decreasing[i]]['speaker_energy_mean'] is not None:
                energy_mean[i] = batch[ids_sorted_decreasing[i]]['speaker_energy_mean']

            if batch[ids_sorted_decreasing[i]]['speaker_energy_std'] is not None:
                energy_std[i] = batch[ids_sorted_decreasing[i]]['speaker_energy_std']

        # essential variables
        data_dict =  {'mel': mel_padded,
                      'speaker_ids': speaker_ids,
                      'accent_ids': accent_ids,
                      'text_raw': text_raw,
                      'language': language,
                      'text': text_padded,
                      'input_lengths': input_lengths,
                      'output_lengths': output_lengths,
                      'audiopaths': audiopaths,
                      'attn_prior': attn_prior_padded,
                      'idx': idx,
                      'speaker_f0_mean': f0_mean,
                      'speaker_f0_std': f0_std,
                      'speaker_energy_mean': energy_mean,
                      'speaker_energy_std': energy_std
                      }

        # additional variables
        if audio_padded is not None:
            data_dict['audio'] = audio_padded
            data_dict['audio_lengths'] = audio_lengths

        if energy_avg_padded is not None:
            data_dict['energy_avg'] = energy_avg_padded

        if f0_padded is not None:
            data_dict['f0'] = f0_padded

        if voiced_mask_padded is not None:
            data_dict['voiced_mask'] = voiced_mask_padded

        if p_voiced_padded is not None:
            data_dict['p_voiced'] = p_voiced_padded

        return data_dict


class TextOnlyData(torch.utils.data.Dataset):
    """
    Dataset for inference from text directly.
    Parses a transcript file and breaks it into lines
    """
    def __init__(self, transcript_path, tp, speaker_id_map, accent_id_map,
                combine_speaker_and_emotion=False,
                speaker_stats_path=None,
                f0_pred_type=None,
                separator='|'):
        self.load_dataset(transcript_path) # fills in self.data
        self.speaker_id_map = speaker_id_map
        self.accent_id_map = accent_id_map
        self.combine_speaker_and_emotion = combine_speaker_and_emotion
        self.f0_pred_type = f0_pred_type
        self.tp = tp

        self.speaker_stats_path = speaker_stats_path
        if self.speaker_stats_path is not None and \
            self.speaker_stats_path != '':
            
            with open(speaker_stats_path) as f:
                data = f.read()
            loaded_speaker_stats = json.loads(data)

            # transform keys to lowercase for easy match
            self.speaker_stats = {}
            for key, value in loaded_speaker_stats.items():
                self.speaker_stats[key.lower()] = value

            print(self.speaker_stats)
        else:
            self.speaker_stats = None

    def load_dataset(self, transcript_path):
        with open(transcript_path, encoding='utf-8') as f:
            self.data = json.load(f)

    def get_text(self, text, language=None, is_phonemized=False):
        text = self.tp.encode_text(text, language=language,
                                    is_phonemized=is_phonemized)
        text = torch.LongTensor(text)
        return text

    def load_speaker_stats(self, speaker_name):
        print(speaker_name)
        if self.speaker_stats is not None and speaker_name.lower() in self.speaker_stats.keys():
            return self.speaker_stats[speaker_name.lower()]
        else:
            return None

    def __getitem__(self, index):
        elts = self.data[index]
        script = elts['script']

        language = elts['language']
        speaker_name = elts['spk_id'] + '-' + elts['emotion'] if self.combine_speaker_and_emotion else elts['spk_id']
        spk_id = self.speaker_id_map[speaker_name] # default spk id
        language = elts['language'] if 'language' in elts else None
        accent_id = self.accent_id_map[language] # default lang id
        text_encoded = self.get_text(script, language=language, is_phonemized=False)
        
        # load speaker stats
        print(speaker_name)
        speaker_stats = self.load_speaker_stats(speaker_name)
        print(speaker_stats)

        if self.speaker_stats is not None and self.f0_pred_type == 'norm_log_f0':
            assert speaker_stats is not None
            f0_mean_key = 'log_f0_mean'
            f0_mean_speaker = speaker_stats[f0_mean_key]

            f0_std_key = 'log_f0_std'
            f0_std_speaker = speaker_stats[f0_std_key]
        elif self.speaker_stats is not None:
            f0_mean_key = 'f0_mean'
            assert speaker_stats is not None
            f0_mean_speaker = speaker_stats[f0_mean_key]

            f0_std_key = 'f0_std'
            f0_std_speaker = speaker_stats[f0_std_key]
        else:
            f0_mean_speaker = 0.0
            f0_std_speaker = 0.0

        # set defaults
        decoder_spk_id = spk_id
        duration_spk_id = spk_id
        f0_spk_id = spk_id # always use the f0 id for voiced prediction
        energy_spk_id = spk_id

        output_dict = {"script": script,
                       "spk_id": spk_id,
                       "decoder_spk_id" : spk_id,
                       "duration_spk_id": spk_id,
                       "f0_spk_id": spk_id,
                       "energy_spk_id": spk_id,
                       "accent_id": accent_id,
                       "text_encoded": text_encoded,
                       "idx": index,
                       "speaker_f0_mean": f0_mean_speaker,
                       "speaker_f0_std": f0_std_speaker,
                       "language": language}
        
        # apply overrides for attribute speaker ids
        attribute_keys = {"decoder_spk_id" ,
                          "duration_spk_id",
                          "f0_spk_id",
                          "energy_spk_id"}
        
        for key in attribute_keys:
            if key in elts.keys():
                assert(elts[key] is not None)
                name = elts[key] + '-' + elts['emotion'] if self.combine_speaker_and_emotion else elts[key]
                output_dict[key] = self.speaker_id_map[name]

        return output_dict

    def __len__(self):
        return len(self.data)




# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms
# Useful for making test sets
# ===================================================================
if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()
    args.rank = 0

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    update_params(config, args.params)
    print(config)

    data_config = config["data_config"]

    ignore_keys = ['training_files', 'validation_files']
    trainset = Data(data_config['training_files'],
                    **dict((k, v) for k, v in data_config.items()
                    if k not in ignore_keys))

    valset = Data(data_config['validation_files'],
                  **dict((k, v) for k, v in data_config.items()
                  if k not in ignore_keys), speaker_ids=trainset.speaker_ids)

    collate_fn = DataCollate()

    for dataset in (trainset, valset):
        for i, batch in enumerate(dataset):
            out = batch
            print("{}/{}".format(i, len(dataset)))
