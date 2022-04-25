import numpy as np
import os
import glob
inference_output_files = glob.glob(os.path.join("/home/ubuntu/efs/teach/src/inference_execution_files/inference__teach_hlsm_trial/", "inference__*.json"))
finished_edh_instance_files = [os.path.join(fn.split("__")[1]) for fn in inference_output_files]
edh_instance_files = [
    os.path.join("/home/ubuntu/efs/hlsm_formatted_edh_format/", "edh_instances", 'valid_seen', f)
    for f in os.listdir('/home/ubuntu/efs/hlsm_formatted_edh_format/edh_instances/valid_seen')
    if f not in finished_edh_instance_files
]
num_jobs = 2
ranges_list = np.array_split(range(len(edh_instance_files)), num_jobs)

for x in ranges_list:
    print(x[0], len(x))
    print("python src/teach/cli/hlsm_inference.py \
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
    	--experirment_def_name teach/eval/hlsm_full/eval_hlsm_valid_unseen \
    	--num_processes 2 \
        --start_file_index {} \
        --num_files {} \
        ".format(x[0], len(x))
    )
