# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN apt-get update --fix-missing && apt-get install -y \
    curl wget\
    ca-certificates \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg libsndfile1 sox locales vim espeak
RUN pip3 install yt-dlp setuptools
RUN pip3 install Cython phonemizer einops tensorboardX numpy matplotlib pillow scipy tqdm scikit-learn h5py pyyaml tensorboard lmdb opencv-python diffusers transformers pytorch-lightning
RUN pip3 install soundfile audioread sox inflect unidecode natsort pandas jupyter tgt srt peakutils
ENV PRAAT_VER 6309
RUN wget http://www.fon.hum.uva.nl/praat/praat${PRAAT_VER}_linux64barren.tar.gz \
        && tar xzf praat${PRAAT_VER}_linux64barren.tar.gz -C /usr/bin \
        && rm praat${PRAAT_VER}_linux64barren.tar.gz
RUN pip3 install praat-parselmouth
RUN pip3 install albumentations opencv-python omegaconf kornia test-tube torch-fidelity pillow argh dacite accelerate pydub plotly webdataset datasets ftfy Jinja2 torchvision torchaudio ndjson ema_pytorch librosa==0.8.0 'jsonargparse[signatures]>=4.17.0'
RUN apt-get -y install espeak-ng