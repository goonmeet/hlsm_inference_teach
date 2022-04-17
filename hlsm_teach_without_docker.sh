export DATA_DIR=/home/ubuntu/efs/teach-dataset
export IMAGE_DIR=/home/ubuntu/efs/teach-dataset/images

# teach_api \
# 	    --data_dir $DATA_DIR \
# 	    --images_dir $IMAGE_DIR \
# 	    --split valid_seen \
# 	    --model_module teach.inference.hlsm_model \
# 	    --model_class HLSM_MODEL \
# 	    --model_dir ./models/baseline_models/hlsm \
# 	    --visual_checkpoint ./models/et_pretrained_models/fasterrcnn_model.pth \
# 	    --object_predictor ./models/et_pretrained_models/maskrcnn_model.pth \
# 	    --seed 4

export DATA_DIR=/home/ubuntu/efs/hlsm_formatted_edh_format
export IMAGE_DIR=/home/ubuntu/efs/teach-dataset/images

# teach_api \
# 	    --data_dir $DATA_DIR \
# 	    --images_dir $IMAGE_DIR \
# 	    --split valid_seen \
# 	    --model_module teach.inference.hlsm_model \
# 	    --model_class HLSM_MODEL \
# 	    --model_dir ./models/hlsm_edh_ckpt/hlsm \
# 			--subgoal_model_file  ./models/hlsm_edh_ckpt/alfred_hlsm_subgoal_model.pytorch \
# 			--depth_model_file  ./models/hlsm_edh_ckpt/hlsm_depth_model.pytorch \
# 			--navigation_model_file  ./models/hlsm_edh_ckpt/hlsm_gofor_navigation_model.pytorch \
# 			--seg_model_file  ./models/hlsm_edh_ckpt/hlsm_segmentation_model.pytorch \
# 	    --seed 4 \
# 			--experirment_def_name teach/eval/hlsm_full/eval_hlsm_valid_unseen


# cd $TEACH_ROOT_DIR
# CUDA_LAUNCH_BLOCKING=1
python src/teach/cli/hlsm_inference.py \
    --model_module teach.inference.hlsm_model \
    --model_class HLSM_MODEL \
		--images_dir $IMAGE_DIR \
    --data_dir $DATA_DIR \
    --output_dir $INFERENCE_OUTPUT_PATH/inference__teach_hlsm_trial \
    --split valid_seen \
    --metrics_file $INFERENCE_OUTPUT_PATH/metrics__teach_hlms_trial.json \
    --seed 4 \
    --model_dir teach_hlsm_trial \
    --model_dir ./models/hlsm_edh_ckpt/hlsm \
    --subgoal_model_file  ./models/hlsm_edh_ckpt/alfred_hlsm_subgoal_model.pytorch \
    --depth_model_file  ./models/hlsm_edh_ckpt/hlsm_depth_model.pytorch \
    --navigation_model_file  ./models/hlsm_edh_ckpt/hlsm_gofor_navigation_model.pytorch \
    --seg_model_file  ./models/hlsm_edh_ckpt/hlsm_segmentation_model.pytorch \
    --device cpu \
		--experirment_def_name teach/eval/hlsm_full/eval_hlsm_valid_unseen #\
    #--num_processes 1

## To speed up remove force cpu:
## home/ubuntu/efs/teach/src/teach/modeling/hlsm/lgp/models/teach/hlsm/transformer_modules/subgoal_history_encoder.py
