#!/bin/bash
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change this to your desired directory:
export WS_DIR=${SCRIPT_DIR}/"workspace"


# Stuff that HLSM needs
export LGP_WS_DIR="${WS_DIR}"
export LGP_MODEL_DIR="${WS_DIR}/models"
export LGP_DATA_DIR="${WS_DIR}/data"

export ET_DATA=~/efs/teach-dataset
export TEACH_ROOT_DIR=~/efs/teach
export ET_LOGS=~/efs/teach/checkpoints/hlsm/pretrained
export VENV_DIR=~/efs/teach/venv
export TEACH_SRC_DIR=$TEACH_ROOT_DIR/src
export ET_ROOT=$TEACH_SRC_DIR/teach/modeling/HLSM
export INFERENCE_OUTPUT_PATH=$TEACH_SRC_DIR/inference_execution_files
export PYTHONPATH=$TEACH_SRC_DIR:$ET_ROOT:$PYTHONPATH

export WS_DIR=${SCRIPT_DIR}/"workspace"

# Stuff that ALFRED needs
export ALFRED_PARENT_DIR=${WS_DIR}/alfred_src
export ALFRED_ROOT=${ALFRED_PARENT_DIR}/teach_src
export DISPLAY=:0
