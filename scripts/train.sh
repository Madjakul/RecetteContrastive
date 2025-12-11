#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

CONFIG_PATH=$PROJECT_ROOT/configs/train.yml
LOGS_DIR=$PROJECT_ROOT/logs
PROCESSED_DS_DIR=$PROJECT_ROOT/data/

# --------------------------------------------------------------------------------------

CHECKPOINT_DIR=$PROJECT_ROOT/tmp/
NUM_PROC=48

# **************************************************************************************

mkdir -p "$LOGS_DIR" || true
mkdir -p "$PROJECT_ROOT/tmp/" || true

cmd=()
cmd=(python3 "$PROJECT_ROOT/train.py"
    --config_path "$CONFIG_PATH"
    --logs_dir "$LOGS_DIR"
    --processed_ds_path "$PROCESSED_DS_DIR")

if [[ -v NUM_PROC ]]; then
    cmd+=(--num_proc "$NUM_PROC")
fi

"${cmd[@]}"
