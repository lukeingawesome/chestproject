export CUDA_VISIBLE_DEVICES=1
CSV_PATH=/data3/private/knee/supplementary/fm/pilot_knee.csv
OUT_CSV=medgemma_val_predictions2.csv
ADAPTER_DIR=medgemma-4b-it-sft-lora-knee-ddp

python3 inference_medgemma.py \
  --csv_path $CSV_PATH \
  --adapter_dir $ADAPTER_DIR \
  --out_csv   $OUT_CSV