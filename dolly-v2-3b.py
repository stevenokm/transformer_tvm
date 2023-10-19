import pickle
import time
import sys
import os
import gc

import random
import copy
import logging

import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.activations import GELUActivation
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXRotaryEmbedding,
    GPTNeoXModel,
    GPTNeoXMLP,
)
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
import ecco

parser = argparse.ArgumentParser()
parser.add_argument("--input_model", default="dolly-v2-3b")
parser.add_argument(
    "--gpu_family", type=str, default="a2000", choices=["v100", "a10", "a100", "a2000"]
)
parser.add_argument(
    "--repeat_count",
    type=int,
    default=2,
)
parser.add_argument(
    "--timestamp",
    type=str,
    default=time.time(),
)
args = parser.parse_args()

logFormatter = logging.Formatter(
    "%(asctime)s " + "[%(threadName)-12.12s] " + "[%(levelname)-5.5s]  " + "%(message)s"
)

fileHandler = logging.FileHandler("dolly-v2-3b_{}.log".format(int(args.timestamp)))
fileHandler.setFormatter(logFormatter)
fileHandler.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(logFormatter)

logging.basicConfig(handlers=[fileHandler, consoleHandler], level=logging.DEBUG)

logging.debug(str(sys.path))
logging.debug(str(os.environ["PYTHONPATH"]))

start_time = int(args.timestamp)

# setup random seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

logging.info("setup model")

if (
    args.input_model == "dolly-v2-3b"
    or args.input_model == "dolly-v2-7b"
    or args.input_model == "dolly-v2-13b"
):
    model_path_or_name = "databricks/{}".format(args.input_model)
else:
    model_path_or_name = os.path.expanduser(os.path.expandvars(args.input_model))
# config = AutoConfig.from_pretrained(model_path_or_name)

# configure the batch_size
batch_size = 4
if args.gpu_family == "a10":
    batch_size = 6
elif args.gpu_family == "a100":
    batch_size = 8
elif args.gpu_family == "a2000":
    batch_size = 4

tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name,
    padding_side="left",
)
device_map = None

model = AutoModelForCausalLM.from_pretrained(
    model_path_or_name,
    torch_dtype=torch.float32,
    device_map=device_map,
    torchscript=True,
)
model.eval()

if hasattr(model, "hf_device_map"):
    logging.info("device_map: {}".format(model.hf_device_map))

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

# add autoTVM on GPU
# ref: https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_cuda.html

import numpy as np

import tvm
from tvm import relay, auto_scheduler, runtime

# Define the neural network and compilation target
network = "dolly-v2-3b"
batch_size = 2
sequence_size = 128
layout = "NL"
dtype = "float32"
target = tvm.target.Target(
    "cuda -arch=sm_86"
)  # NOTE: for RTX 3090 or later, reffer to https://developer.nvidia.com/cuda-gpus
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)
mod_file = "%s-%s-B%d-%s-mod.json" % (network, layout, batch_size, dtype)
params_file = "%s-%s-B%d-%s-params.bin" % (network, layout, batch_size, dtype)
vmc_lib_file = "%s-%s-B%d-%s-vmc-lib.so" % (network, layout, batch_size, dtype)
vmc_code_file = "%s-%s-B%d-%s-vmc-code.ro" % (network, layout, batch_size, dtype)
main_name = "main"


# prepare network
query = "Explain to me the difference between nuclear fission and fusion."

# generate sequence ids for model input
model_inputs = generate_text.preprocess(query)
input_names = ["input_ids", "attention_mask"]
if batch_size > 1:
    for input_name in input_names:
        model_inputs[input_name] = model_inputs[input_name].repeat(batch_size, 1)
dummy_inputs = [model_inputs[input_name] for input_name in input_names]
batch_size = model_inputs["input_ids"].shape[0]
sequence_size = model_inputs["input_ids"].shape[1]

# trace model
for para in model.parameters():
    para.requires_grad = False

traced_file = "{}-traced.pt".format(network)
if not os.path.exists(traced_file):
    logging.info("Trace model...")
    scripted_model = torch.jit.trace(
        model,
        dummy_inputs,
    )
    torch.jit.save(scripted_model, traced_file)
# TODO: bug? use reloaded traced model can solve int64 assert problem
logging.info("Load traced model...")
scripted_model = torch.jit.load(traced_file)
scripted_model.eval()
for para in scripted_model.parameters():
    para.requires_grad = False

del model
gc.collect()

# Extract tasks from the network
logging.info("Extract tasks...")

if os.path.exists(mod_file) and os.path.exists(params_file):
    logging.info("Load mod and params...")
    mod = tvm.ir.load_json(open(mod_file, "r").read())
    params = relay.load_param_dict(open(params_file, "rb").read())
else:
    logging.info("Import from pytorch...")
    shape_list = [
        (input_name, [batch_size, sequence_size]) for input_name in input_names
    ]
    mod, params = relay.frontend.from_pytorch(
        scripted_model, shape_list, default_dtype=dtype
    )
    with open(mod_file, "w") as f:
        f.write(tvm.ir.save_json(mod))
    with open(params_file, "wb") as f:
        f.write(relay.save_param_dict(params))

del scripted_model
gc.collect()

# perform auto-scheduler
if os.path.exists(log_file):
    logging.info("Load log file...")
else:
    tasks, task_weights = auto_scheduler.extract_tasks(mod[main_name], params, target)

    for idx, task in enumerate(tasks):
        logging.info(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        logging.debug(str(task.compute_dag))

    def run_tuning():
        logging.info("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=300, timeout=10
        )

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=200,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)

    run_tuning()

    del tasks
    del task_weights
    gc.collect()

# Compile with the history best
# NOTE: use vm because graph executor does not support dynamic shape
if os.path.exists(vmc_lib_file) and os.path.exists(vmc_code_file):
    logging.info("Load vmc...")
    lib = tvm.runtime.load_module(vmc_lib_file)
    code = bytearray(open(vmc_code_file, "rb").read())
    vmc = runtime.vm.Executable.load_exec(code, lib)
else:
    logging.info("Compile...")
    # NOTE: NEED CPU MEMORY + SWAP >= 150GB, or segmentation fault
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            vmc = relay.vm.compile(mod, target=target, params=params)
    # Save the vmc file for later use
    code, lib = vmc.save()
    lib.export_library(vmc_lib_file)
    with open(vmc_code_file, "wb") as fo:
        fo.write(code)

del mod
del params
gc.collect()

# Create VM runtime
logging.info("Create runtime...")
dev = tvm.device(str(target), 0)
vm = runtime.vm.VirtualMachine(vmc, dev)
input_data = {}
for input_name in input_names:
    input_data[input_name] = tvm.nd.array(
        model_inputs[input_name].numpy().astype(dtype)
    )
# NOTE: NEED GPU MEMORY >= 22GB, or CUDA OOM
vm.set_input(
    main_name,
    **input_data,
)

# Evaluate
logging.info("Evaluate inference time cost...")
logging.info(str(vm.benchmark(dev, repeat=3, min_repeat_ms=500)))
# NOTE: if the program has error, then remove dolly-v2-3b-traced*, dolly-v2-3b-NL-B2-* files
# and re-run codes start from auto-scheduler
