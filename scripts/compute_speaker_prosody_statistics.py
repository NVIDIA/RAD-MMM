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
import os
import sys
import argparse
import yaml
sys.path.append(os.path.join(sys.path[0],'../'))
import argparse
import json
import torch
from torch.utils.data import DataLoader
from train import parse_data_from_batch, prepare_dataloaders
from common import get_mask_from_lengths, update_params
from data import Data, DataCollate
from scripts.scripting_utils import load_yaml


def compute_speaker_stats(output_path, batch_size=100, 
                        f0_min=80, f0_max=660,
                        overwrite=False):
    os.makedirs(output_path, exist_ok=True)
    n_gpus = 0
    data_config['trim_db'] = 60
    base_loader, valset, _ = prepare_dataloaders(
        data_config, n_gpus, batch_size)
    speakers = valset.speaker_ids
    ignore_keys = ['training_files', 'validation_files', 'dataset_metadata']
    
    collated_save_path = os.path.join(output_path, f"collated_stats.json")
    collated_speaker_stats_dict = {}
    
    for speaker in speakers:
        save_path = os.path.join(output_path, f"{speaker}.json")
        if os.path.exists(save_path) and not overwrite:
            print(f"Skipping current speaker stats: exists at {save_path}")
            continue

        speaker_attr_dict = {}
        trainset = Data(data_config['dataset_metadata'],
                    "train",
                    **dict((k, v) for k, v in data_config.items()
                    if k not in ignore_keys))

        trainset.filter_by_duration_(trainset.dur_min, trainset.dur_max)
        trainset.filter_by_speakers_((speaker, ), True)
        collate_fn = DataCollate()
        data_loader = DataLoader(trainset, num_workers=8, shuffle=False,
                                 sampler=None, batch_size=batch_size,
                                 pin_memory=False, drop_last=False,
                                 collate_fn=collate_fn)

        for batch in data_loader:
            
            (mel, speaker_ids, accent_ids, text, in_lens, out_lens, attn_prior,
            f0, voiced_mask, p_voiced, energy_avg,
            audiopaths, _dc) = self.unpack_batch(batch)
            
            (mel, mel_transformed, speaker_ids, text, in_lens, out_lens, attn_prior,
             f0, voiced_mask, p_voiced, energy_avg,
             audiopaths, accent_ids, emotion_ids, texts, speaker_names,
             accent_names, 
             f0_mean, f0_std, energy_mean, energy_std) = parse_data_from_batch(batch, device='cpu')

            f0 = f0.flatten()[voiced_mask.flatten().bool()]
            f0 = f0[f0 > f0_min]
            f0 = f0[f0 < f0_max]
            f0_median = torch.median(f0)
            f0_mean = f0.mean()
            f0_std = f0.std()
            log_f0 = torch.log(f0)
            log_f0_mean = log_f0.mean()
            log_f0_std = log_f0.std()
            log_f0_median = torch.median(log_f0)
            mask = get_mask_from_lengths(out_lens, device='cpu')
            energy_mean = energy_avg[mask].mean()
            energy_std = energy_avg[mask].std()
            speaker_attr_dict['f0_median'] = f0_median.item()
            speaker_attr_dict['f0_mean'] = f0_mean.item()
            speaker_attr_dict['f0_std'] = f0_std.item()
            speaker_attr_dict['log_f0_median'] = log_f0_median.item()
            speaker_attr_dict['log_f0_mean'] = log_f0_mean.item()
            speaker_attr_dict['log_f0_std'] = log_f0_std.item()
            speaker_attr_dict['energy_mean'] = energy_mean.item()
            speaker_attr_dict['energy_std'] = energy_std.item()
            speaker_attr_dict['n_files'] = mel.shape[0]
            print(speaker, speaker_attr_dict)
            collated_speaker_stats_dict[speaker] = speaker_attr_dict
            break

        print(speaker_attr_dict)
        with open(save_path, "w") as f:
            json.dump(speaker_attr_dict, f)

    with open(collated_save_path, "w") as f:
        json.dump(collated_speaker_stats_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('-n', '--n_samples', type=int, default=100)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()

    global config
    config = json.loads(data)
    update_params(config, args.params)
    print(config)

    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global model_config
    model_config = config["model_config"]
    print('args', args)

    compute_speaker_stats(args.output_path, args.n_samples)
