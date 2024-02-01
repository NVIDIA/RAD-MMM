# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
from pathlib import Path
import numpy as np
MAX_WAV_VALUE = 32768.0

def main(mel_files, 
        waveglow_path, 
        output_dir,
        sigma=1.0, 
        sampling_rate=22050):
    print(waveglow_path)
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    print(waveglow)
  
    for i, file_path in enumerate(mel_files):
        with open(file_path, 'rb') as f:
            mel = np.load(f)
        mel = torch.from_numpy(mel)
        mel = torch.autograd.Variable(mel.cuda())
        if mel.dim() < 3:
            mel = torch.unsqueeze(mel, 0)
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma)
            audio = audio * MAX_WAV_VALUE
        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            output_dir, "{}_synthesis.wav".format(Path(file_path).stem))
        write(audio_path, sampling_rate, audio)
        print(audio_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path", required=True)
    parser.add_argument('-w', '--waveglow_path',
                        default='../vocoder_weights/')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    files = os.listdir(args.filelist_path)
    spk2files = {}
    for f in files:
        spk = "ljs"
        if spk not in spk2files: spk2files[spk] = []
        spk2files[spk].append(os.path.join(args.filelist_path, f))

    for spk in spk2files:
        vocoder_name = "waveglow_256channels_universal_v5.pt"
        main(mel_files=spk2files[spk],
            waveglow_path=os.path.join(args.waveglow_path, vocoder_name),
            output_dir=args.output_dir)
