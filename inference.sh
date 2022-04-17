source init.sh
teach_inference \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --image_dir $IMG_DIR
    --split valid_seen \
    --metrics_file $METRICS_FILE \
    --model_module teach.inference.sample_model \
    --model_class SampleModel

teach_eval \
    --data_dir $DATA_DIR \
    --inference_output_dir $OUTPUT_DIR \
    --split valid_seen \
    --metrics_file $METRICS_FILE

