# RADMMM
Pytorch Lightning(PTL) repository for [RADMMM](https://arxiv.org/pdf/2301.10335.pdf) and [VANI](https://arxiv.org/abs/2303.07578). Intended to be released publically.

## Installation
Please use the Dockerfile listed inside `docker/` to build a Docker image or use the following:

```
pip install requirements.txt
```

## Getting started
In order to get started, please follow the steps below:

1. Download the data. Use the links below to download the opensource dataset or use this [link](https://drive.google.com/drive/folders/1tALLXAR-quig3yAvKcCW12oAKWgV9w_2?usp=sharing) to download our version of the following dataset. Please place it under a directory with the following name:

```bash
multilingual-dataset/
```

| Language        | Train Prompts | Val Prompts | Dataset Link                                                                                        | Speaker Name      |
|-----------------|---------------|-------------|-----------------------------------------------------------------------------------------------------|-------------------|
| English (US)    | 10000         | 10          | https://keithito.com/LJ-Speech-Dataset                                                              | LJ Speech         |
| German (DE)     | 10000         | 10          | https://opendata.iisys.de/opendata/Datasets/HUI-Audio-Corpus-German/dataset_full/Bernd_Ungerer.zip  | Bernd Ungerer     |
| French (FR)     | 10000         | 10          | https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset                                         | Nadine Eckert     |
| Spanish (ES)    | 10000         | 10          | https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset                                         | Tux               |
| Hindi (HI)      | 8237          | 10          | https://aclanthology.org/2020.lrec-1.789.pdf                                                        | Indic TTS         |
| Portuguese (BR) | 3085          | 10          | https://github.com/Edresson/TTS-Portuguese-Corpus                                                   | Edresson Casanova |
| Spanish (LATN)  | 7376          | 10          | https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset                                         | Karen Savage      |
| Total           | 58698         | 70          |                                                                                                     |                   |

2. Filelists are already present in `datasets/`


3. Place the vocoders in the `vocoders/hifigan_vocoder` directory.

4. Preprocess the dataset to phonemize the data:
```bash
python3 scripts/phonemize_text.py -c configs/RADMMM_opensource_data_config_phonemizerless.yaml
```

5. Train the model by following the steps in Training or download the pretrained checkpoints for RADMMM(mel-spectogram generator as well as HiFi-GAN(vocoder) as explained in Pretrained Checkpoints.

6. Run inference by following the steps in Inference. Inference requires the text to be phonemized and we suggest using [Phonemizer (GPL License)](https://github.com/bootphon/phonemizer) for best results. We provide an alternative to Phonemizer based on language dictionaries that can be downloaded from this [link](https://drive.google.com/drive/folders/1woNCODwXh9aHu7Fd6b4Jo42aL7f5RFZg?usp=sharing). Please place these under the directory `assets/` and refer to usage in Inference.

## Training 
Train the decoder and attribute predictor using the commands given below:

1. Train the decoder (for single GPU use trainer.devices=1, for multi-gpu use trainer.devices=nGPUs)
```bash
python3 tts_main.py fit -c configs/RADMMM_train_config.yaml -c configs/RADMMM_opensource_data_config_phonemizerless.yaml -c configs/RADMMM_model_config.yaml --trainer.num_nodes=1 --trainer.devices=1
```

2. Train the attribute prediction modules (for single GPU use trainer.devices=1, for multi-gpu use trainer.devices=nGPUs)
```bash
python3 tts_main.py fit -c configs/RADMMM_f0model_config.yaml -c configs/RADMMM_energymodel_config.yaml -c configs/RADMMM_durationmodel_config.yaml -c configs/RADMMM_vpredmodel_config.yaml -c configs/RADMMM_train_config.yaml -c configs/RADMMM_opensource_data_config_phonemizerless.yaml -c configs/RADMMM_model_config.yaml --trainer.num_nodes=1 --trainer.devices=1  --model.encoders_path=<decoder_path> --model.decoder_path=<decoder_path>
```

## Inference
Inference can be performed in the following way. There's a separate INPUT\_FILEPATH required for inference, some samples of which are in the `model_inputs/` directory. 


The provided [example transcript](model_inputs/sample_transcript.json) demonstrates how to specify `tts`-mode transcripts. Note that it is possible to mix and match speaker identities for individual attribute predictors using the keys `decoder_spk_id`, `duration_spk_id`, `f0_spk_id`, and `energy_spk_id`. Any unspecified speaker ids will default to whatever is specified for `spk_id`. For implementation details, please see the dataset class `TextOnlyData` in `data.py`.

**NOTE**: speaker id mapping is determined by the dictionary constructed in the training dataset. If the training dataset is modified or unavailable during inference, be sure to manually specify the original dictionary used during training as `self.speaker_ids` in the setup method of `datamodules.py`. Similar for speaker statistics, please use the method `load_speaker_stats(speaker)` to get stats for the speaker.


```bash
# setup the model paths and configs (see Pretrained Checkpoints for details)
MODEL_PATH=<model_path>
CONFIG_PATH=<config_path>

# setup the vocoder paths and configs (see Pretrained Checkpoints for details)
VOCODER_PATH=<vocoder_path>
VOCODER_CONFIG_PATH=<vocoder_config_path>

INPUT_FILEPATH=model_inputs/resynthesis_prompts.json
#INPUT_FILEPATH=model_inputs/language_transfer_prompts.json

python tts_main.py predict -c $CONFIG_PATH --ckpt_path=$MODEL_PATH --model.predict_mode="tts" --data.inference_transcript=$INPUT_FILEPATH --model.prediction_output_dir=outdir --trainer.devices=1 --data.batch_size=1 --model.vocoder_checkpoint_path=$VOCODER_PATH --model.vocoder_config_path=$VOCODER_CONFIG_PATH --data.phonemizer_cfg='{"en_US": "assets/en_US_word_ipa_map.txt","de_DE": "assets/de_DE_word_ipa_map.txt","en_UK": "assets/en_UK_word_ipa_map.txt","es_CO": "assets/es_CO_word_ipa_map.txt","es_ES": "assets/es_ES_word_ipa_map.txt","fr_FR": "assets/fr_FR_word_ipa_map.txt","hi_HI": "assets/hi_HI_word_ipa_map.txt","pt_BR": "assets/pt_BR_word_ipa_map.txt","te_TE": "assets/te_TE_word_ipa_map.txt", "es_MX": "assets/es_ES_word_ipa_map.txt"}' 
```

## Pretrained checkpoint(s)
### RADMMM (mel-spectogram synthesizer)

[RADMMM checkpoint](https://drive.google.com/file/d/1m-pAIeCBuT6yD77kIETqkAYYDtA_cbzs/view?usp=sharing)  
[RADMMM config](https://drive.google.com/file/d/1sPFFy6aYufbseox5Rxwt-EjDbMogkUwP/view?usp=sharing)  

### Vocoder (HiFi-GAN - waveform synthesizer)

[HiFi-GAN checkpoint](https://drive.google.com/file/d/1VaH5_MhAjAjHlihi2k-lcOOoy4NqtRV4/view?usp=sharing). 
[HiFi-GAN config](https://drive.google.com/file/d/1-eBTNfIh-LSstNirQawHW4jsI-t01jTU/view?usp=sharing). 

## Reference samples using pretrained model
[Link](https://drive.google.com/drive/folders/1nORslyv1rB_wKPAqonTBwYuJwWMGxT2r?usp=sharing)

## Reference Samples and more information
Please visit this [page](https://research.nvidia.com/labs/adlr/projects/radmmm) for samples related to models trained on the opensource dataset.

## Support
rbadlani@nvidia.com

## Contributing
Please create a new branch for your feature or bug fix.
Create a PR to the main branch describing your contributions.

## Authors and acknowledgment
Authors: Rohan Badlani, Rafael Valle and Kevin J. Shih.

The authors would like to thank Akshit Arora, Subhankar Ghosh, Siddharth Gururani, João Felipe Santos, Boris Ginsburg and Bryan Catanzaro for their support and guidance.

[Phonemizer(GPL License)](https://github.com/bootphon/phonemizer), without modifications, is separately used to convert from graphemes to IPA phonemes.  
[Pratt](https://www.fon.hum.uva.nl/praat/) is used for speech manipulation.

The symbol set used in RADMMM heavily draws inspiration from [WikiPedia's IPA CHART](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet#/media/File:IPA_chart_2020.svg)

The code in this repository is heavily inspired by or makes use of source code from the following works:

1. [RADTTS](https://github.com/nvidia/radtts)
2. [Tacotron's implementation by Keith Ito](https://github.com/keithito/tacotron)
3. [STFT code from Prem Seetharaman](https://github.com/pseeth/torch-stft)
4. Masked Autoregressive Flows
5. [Flowtron](https://github.com/nvidia/flowtron)
6. neural spline functions used in this work: https://github.com/ndeutschmann/zunis
7. Original Source for neural spline functions: https://github.com/bayesiains/nsf
8. Bipartite Architecture based on code from WaveGlow
9. [HiFi-GAN](https://github.com/jik876/hifi-gan)
10. [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
11. [WaveGlow](https://github.com/NVIDIA/waveglow/)

# Relevant Papers
Rohan Badlani, Rafael Valle, Kevin J. Shih, João Felipe Santos, Siddharth Gururani, Bryan Catanzaro. [RAD-MMM: Multi-lingual Multi-accented Multi-speaker Text to Speech](https://arxiv.org/abs/2301.10335). Interspeech 2023.

Rohan Badlani, Akshit Arora, Subhankar Ghosh, Rafael Valle, Kevin J. Shih, João Felipe Santos, Boris Ginsburg, Bryan Catanzaro. [VANI: Very-lightweight Accent-controllable TTS for Native and Non-native speakers with Identity Preservation](https://ieeexplore.ieee.org/abstract/document/10096613). ICASSP 2023

Rafael Valle, João Felipe Santos, Kevin J. Shih, Rohan Badlani, Bryan Catanzaro. [High-Acoustic Fidelity Text To Speech Synthesis With Fine-Grained Control Of Speech Attributes](https://ieeexplore.ieee.org/document/10096279). ICASSP 2023.

Rohan Badlani, Adrian Łańcucki, Kevin J. Shih, Rafael Valle, Wei Ping, Bryan Catanzaro. [One TTS Alignment to Rule Them All](https://ieeexplore.ieee.org/document/9747707). ICASSP 2022

Kevin J Shih, Rafael Valle, Rohan Badlani, Adrian Lancucki, Wei Ping, Bryan Catanzaro. [RAD-TTS: Parallel flow-based TTS with robust alignment learning and diverse synthesis](https://openreview.net/pdf?id=0NQwnnwAORi). ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models 2021

Kevin J Shih, Rafael Valle, Rohan Badlani, Jõao Felipe Santos, Bryan Catanzaro. [Generative Modeling for Low Dimensional Speech Attributes with Neural Spline Flows](https://arxiv.org/abs/2203.01786)
## License
MIT
