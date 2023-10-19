# Transformer on Auto-scheduler Tutorial

A tutorial on how to compile the Transformer model on auto-scheduler.

## Description

This tutorial is based on dolly-v2, the auto-scheduler of TVM.

## Installation & Run

```bash
git clone --recursive https://github.com/stevenokm/transformer_tvm.git
cd transformer_tvm
python3.8 -m venv .venv
source .venv/bin/activate
bash pytorch_huggingface_tvm.sh
bash run.sh
```

for release build, change `TVM_LIBRARY_PATH=./.venv/lib/tvm/build-debug` to `TVM_LIBRARY_PATH=./.venv/lib/tvm/build-release` in `.env` file.