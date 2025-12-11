#!/bin/bash

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/.. # Do not modify
DATA_ROOT=$PROJECT_ROOT/data                     # Do not modify

# ************************** Customizable Arguments ************************************

LOGS_DIR=$PROJECT_ROOT/logs
PROCESSED_DS_PATH=$PROJECT_ROOT/data/

# **************************************************************************************

mkdir -p "$LOGS_DIR" || true

cmd=()
cmd=(python3 "$PROJECT_ROOT/preprocess.py"
    --logs_dir "$LOGS_DIR"
    --processed_ds_path "$PROCESSED_DS_PATH")

"${cmd[@]}"
