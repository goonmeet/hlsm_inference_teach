export DATA_DIR=/home/ubuntu/efs/teach-dataset
export IMAGE_DIR=/home/ubuntu/efs/teach-dataset/images

teach_api \
	    --data_dir $DATA_DIR \
	    --images_dir $IMAGE_DIR \
	    --split valid_seen \
	    --model_module teach.inference.et_model \
	    --model_class ETModel \
	    --model_dir ./models/baseline_models/et \
	    --visual_checkpoint ./models/et_pretrained_models/fasterrcnn_model.pth \
	    --object_predictor ./models/et_pretrained_models/maskrcnn_model.pth \
	    --seed 4 
