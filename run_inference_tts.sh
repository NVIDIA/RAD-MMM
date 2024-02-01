MODEL_PATH=/home/dcg-adlr-rbadlani-output/RADMMM_PTL2/epic_dpm_apmprorietary-tj-radmmm-ptl/latest-epoch_20-iter_29999.ckpt
CONFIG_PATH=/home/dcg-adlr-rbadlani-output/RADMMM_PTL2/epic_dpm_apmprorietary-tj-radmmm-ptl/lightning_logs/version_1209154/config.yaml

VOCODER_PATH=/home/dcg-adlr-rafaelvalle-output.cosmos356/hifigan/universal_44khz/g_01224000
VOCODER_CONFIG_PATH=/home/dcg-adlr-rafaelvalle-source.cosmos597/repos/hifigan-internal/config_44khz_new.json

#INPUT_FILEPATH=model_inputs/sample_transcript.json
INPUT_FILEPATH=model_inputs/tj_language_transfer_prompts.json

python tts_main.py predict -c $CONFIG_PATH --ckpt_path=$MODEL_PATH --model.predict_mode="tts" \
--data.inference_transcript=$INPUT_FILEPATH --model.prediction_output_dir=outdir --trainer.devices=1 \
--data.batch_size=1 --model.vocoder_checkpoint_path=$VOCODER_PATH --model.vocoder_config_path=$VOCODER_CONFIG_PATH
