export DATA_DIR=/home/ubuntu/efs/teach-dataset
export IMAGE_DIR=/home/ubuntu/efs/teach-dataset/images

teach_eval \
    --data_dir $DATA_DIR \
    --inference_output_dir $INFERENCE_OUTPUT_PATH/inference__teach_hlsm_trial  \
    --split valid_seen \
    --metrics_file ./run_metrics_valid_seen.json
