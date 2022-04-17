export HOST_DATA_DIR=/home/ubuntu/efs/teach-dataset
export HOST_IMAGES_DIR=/home/ubuntu/efs/teach-dataset/images_and_states
export HOST_OUTPUT_DIR=/home/ubuntu/efs/output
export API_PORT=5000
export SUBMISSION_PK=168888
export INFERENCE_GPUS='"device=0"'
export API_GPUS='"device=1"'
export SPLIT=valid_seen
export DOCKER_NETWORK=no-internet

mkdir -p $HOST_IMAGES_DIR $HOST_OUTPUT_DIR
docker network create --driver=bridge --internal $DOCKER_NETWORK
