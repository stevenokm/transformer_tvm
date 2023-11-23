#!/bin/bash -e

# import .env
set -a
source .env
set +a

CUDA_VISIBLE_DEVICES="0"

epoch_time=$(date +%s)
time_cmd="/usr/bin/time"

# exec ${time_cmd} -v python3 dolly-v2-12b.py ${epoch_time} 2>&1 | tee dolly-v2-12b_${epoch_time}.log

# exec ${time_cmd} -v python3 dolly-v2-7b.py ${epoch_time} 2>&1 | tee dolly-v2-7b_${epoch_time}.log

# exec ${time_cmd} -v \
#     python3 dolly-v2-3b.py \
#     --input_model ~/dbfs/dolly_training/dolly__2023-06-05T00\:49\:05 \
#     --timestamp ${epoch_time}
# python3 analyze_act.py \
#     --input_model dolly-v2-3b \
#     --timestamp ${epoch_time}

# python3 dolly-v2-3b.py \
#     --input_model dolly-v2-3b \
#     --timestamp ${epoch_time}

python3 gpt2.py \
    --input_model gpt2 \
    --timestamp ${epoch_time}
