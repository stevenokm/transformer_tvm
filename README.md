# Transformer on Auto-scheduler Tutorial

A tutorial on how to compile the Transformer model on auto-scheduler.

## Description

This tutorial is based on dolly-v2 & gpt-2, the auto-scheduler of TVM.

## Installation 

```bash
git clone --recursive https://github.com/stevenokm/transformer_tvm.git
cd transformer_tvm
python3.8 -m venv .venv
source .venv/bin/activate
# modify L75 & L86 in pytorch_huggingface_tvm.sh to your GPU architecture
bash pytorch_huggingface_tvm.sh
```

## Run

### dolly-v2-3b

```bash
# remember to source .venv/bin/activate
# modify L133 in dolly-v2-3b.py to your GPU architecture
python3 dolly-v2-3b.py
```

### gpt-2

```bash
# remember to source .venv/bin/activate
# modify L133 in gpt-2.py to your GPU architecture
python3 gpt-2.py
```

for release build, change `TVM_LIBRARY_PATH=./.venv/lib/tvm/build-debug` to `TVM_LIBRARY_PATH=./.venv/lib/tvm/build-release` in `.env` file.