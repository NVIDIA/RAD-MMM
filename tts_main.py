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
from pytorch_lightning.cli import LightningCLI, ArgsType
from tts_lightning_modules import TTSModel
from data_modules import BaseAudioDataModule
from jsonargparse import lazy_instance
from decoders import RADMMMFlow
from loss import RADTTSLoss
import inspect
from pytorch_lightning.callbacks import ModelCheckpoint
from training_callbacks import LogDecoderSamplesCallback, \
    LogAttributeSamplesCallback
from utils import get_class_args
from tts_text_processing.text_processing import TextProcessing
from common import Encoder


class RADTTSLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_callback")
        parser.add_lightning_class_args(LogDecoderSamplesCallback, "decoder_samples_callback")
        parser.add_lightning_class_args(LogAttributeSamplesCallback, "attribute_samples_callback")
        parser.set_defaults({"checkpoint_callback.filename": "latest-epoch_{epoch}-iter_{global_step:.0f}",
                             "checkpoint_callback.monitor": "global_step",
                             "checkpoint_callback.mode": "max",
                             "checkpoint_callback.every_n_train_steps": 3000,
                             "checkpoint_callback.dirpath": "/debug",
                             "checkpoint_callback.save_top_k": -1,
                             "checkpoint_callback.auto_insert_metric_name": False})
        parser.link_arguments("model.output_directory", "checkpoint_callback.dirpath")
        parser.link_arguments("model.output_directory", "trainer.default_root_dir")
        parser.link_arguments("data.sampling_rate", "model.sampling_rate")
        parser.link_arguments("data.symbol_set", "model.symbol_set")
        parser.link_arguments("data.cleaner_names", "model.cleaner_names")
        parser.link_arguments("data.heteronyms_path", "model.heteronyms_path")
        parser.link_arguments("data.phoneme_dict_path", "model.phoneme_dict_path")
        parser.link_arguments("data.p_phoneme", "model.p_phoneme")
        parser.link_arguments("data.handle_phoneme", "model.handle_phoneme")
        parser.link_arguments("data.handle_phoneme_ambiguous", "model.handle_phoneme_ambiguous")
        parser.link_arguments("data.prepend_space_to_text", "model.prepend_space_to_text")
        parser.link_arguments("data.append_space_to_text", "model.append_space_to_text")
        parser.link_arguments("data.add_bos_eos_to_text", "model.add_bos_eos_to_text")
        parser.link_arguments("data.phonemizer_cfg", "model.phonemizer_cfg")


def lcli(args: ArgsType = None):
    cli = RADTTSLightningCLI(TTSModel, BaseAudioDataModule, save_config_kwargs={"overwrite": True},args=args)

if __name__=="__main__":
    lcli()
