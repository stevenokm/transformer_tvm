#!/bin/bash
# unset CUDA_VISIBLE_DEVICES
# CUDA_VISIBLE_DEVICES="" # CPU only
CUDA_VISIBLE_DEVICES="3,2"
# fix for OOM
# PYTORCH_CUDA_ALLOC_CONF="backend:native,max_split_size_mb:10000"

epoch_time=$(date +%s)
time_cmd="/usr/bin/time"

# exec ${time_cmd} -v python3 dolly-v2-12b.py ${epoch_time} 2>&1 | tee dolly-v2-12b_${epoch_time}.log

exec ${time_cmd} -v python3 dolly-v2-7b.py ${epoch_time} 2>&1 | tee dolly-v2-7b_${epoch_time}.log

# exec ${time_cmd} -v python3 dolly-v2-3b.py ${epoch_time} 2>&1 | tee dolly-v2-3b_${epoch_time}.log

