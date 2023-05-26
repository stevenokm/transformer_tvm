#!/bin/bash
# unset CUDA_VISIBLE_DEVICES
# CUDA_VISIBLE_DEVICES="" # CPU only
CUDA_VISIBLE_DEVICES="1,2,3"
# fix for OOM
PYTORCH_CUDA_ALLOC_CONF=""

epoch_time=$(date +%s)
time_cmd="/usr/bin/time"

exec ${time_cmd} -v python3 dolly-v2-12b.py ${epoch_time} 2>&1 | tee dolly-v2-12b_${epoch_time}.log
