#python main/rollout_and_evaluate.py teach/eval/hlsm_full/eval_hlsm_valid_unseen


cd $TEACH_ROOT_DIR
python src/teach/cli/hlsm_inference.py \
	    --model_module teach.inference.hlsm_model \
	    --model_class HLSM_MODEL \
	    --data_dir $ET_DATA \
	    --output_dir $INFERENCE_OUTPUT_PATH/inference__teach_hlsm_trial \
	    --split valid_seen \
	    --metrics_file $INFERENCE_OUTPUT_PATH/metrics__teach_hlsm_trial.json \
	    --seed 4 \
	    --model_dir /home/ubuntu/efs/teach/src/teach/modeling/hlsm/teach_hlsm_trial \
	    --device cpu \
	    --images_dir /home/ubuntu/efs/teach-dataset/images \
	    --experirment_def_name "teach/eval/hlsm_full/eval_hlsm_test" \
	    --num_processes 1
			# --object_predictor $ET_LOGS/pretrained/maskrcnn_model.pth \
