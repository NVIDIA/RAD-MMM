#!/bin/sh

# setup the model paths and configs (see Pretrained Checkpoints for details)
MODEL_PATH=${1:-"/akshit/scratch/generator_ckpt/radmmm_public/attribute_model.ckpt"}
CONFIG_PATH=${2:-"/akshit/scratch/generator_ckpt/radmmm_public/config.yaml"}

# setup the vocoder paths and configs (see Pretrained Checkpoints for details)
VOCODER_PATH=${3:-"/akshit/scratch/generator_ckpt/hfg_public/g_00072000"}
VOCODER_CONFIG_PATH=${4:-"/akshit/scratch/generator_ckpt/hfg_public/config_16khz.json"}

INPUT_FILEPATH=${5:-"model_inputs/resynthesis_prompts.json"}
# INPUT_FILEPATH=${5:-"model_inputs/language_transfer_prompts.json"}

cd /akshit/scratch/RAD-MMM && python tts_main.py predict -c $CONFIG_PATH --ckpt_path=$MODEL_PATH --model.predict_mode="tts" --data.inference_transcript=$INPUT_FILEPATH --model.prediction_output_dir="/akshit/scratch/RAD-MMM/tutorials/out1" --trainer.devices=1 --data.batch_size=1 --model.vocoder_checkpoint_path=$VOCODER_PATH --model.vocoder_config_path=$VOCODER_CONFIG_PATH --data.phonemizer_cfg='{"en_US": "assets/en_US_word_ipa_map.txt","es_MX": "assets/es_MX_word_ipa_map.txt","de_DE": "assets/de_DE_word_ipa_map.txt","en_UK": "assets/en_UK_word_ipa_map.txt","es_CO": "assets/es_CO_word_ipa_map.txt","es_ES": "assets/es_ES_word_ipa_map.txt","fr_FR": "assets/fr_FR_word_ipa_map.txt","hi_HI": "assets/hi_HI_word_ipa_map.txt","pt_BR": "assets/pt_BR_word_ipa_map.txt","te_TE": "assets/te_TE_word_ipa_map.txt"}' --model.encoders_path="/akshit/scratch/generator_ckpt/radmmm_public/decoder.ckpt" --model.decoder_path="/akshit/scratch/generator_ckpt/radmmm_public/decoder.ckpt" --model.output_directory="/akshit/scratch/RAD-MMM/tutorials/run1"