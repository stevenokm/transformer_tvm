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

# args = {}
# args.input_model = "dolly-v2-3b"
# args.gpu_family = "a2000"
# args.timestamp = int(time.time())

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
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config)
# model.tie_weights()

# analyze device_map under memory constraints
# device_map = infer_auto_device_map(model, verbose=True)
# device_map = "auto"
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

# model_config = {
#     "embedding": "gpt_neox.embed_in",
#     "type": "mlm",
#     "activations": ["mlp\.dense_h_to_4h"],  # This is a regex
#     "token_prefix": "",
#     "partial_token_prefix": "",
# }

# ecco_model = ecco.from_pretrained(
#     model_path_or_name, activations=True, model_config=model_config
# )

# text = """ Now I ask you: \n what can be expected of man since he is a being endowed with strange qualities? Shower upon him every earthly blessing, drown him in a sea of happiness, so that nothing but bubbles of bliss can be seen on the surface; give him economic prosperity, such that he should have nothing else to do but sleep, eat cakes and busy himself with the continuation of his species, and even then out of sheer ingratitude, sheer spite, man would play you some nasty trick. He would even risk his cakes and would deliberately desire the most fatal rubbish, the most uneconomical absurdity, simply to introduce into all this positive good sense his fatal fantastic element. It is just his fantastic dreams, his vulgar folly that he will desire to retain, simply in order to prove to himself--as though that were so necessary-- that men still are men and not the keys of a piano, which the laws of nature threaten to control so completely that soon one will be able to desire nothing but by the calendar. And that is not all: even if man really were nothing but a piano-key, even if this were proved to him by natural science and mathematics, even then he would not become reasonable, but would purposely do something perverse out of simple ingratitude, simply to gain his point. And if he does not find means he will contrive destruction and chaos, will contrive sufferings of all sorts, only to gain his point! He will launch a curse upon the world, and as only man can curse (it is his privilege, the primary distinction between him and other animals), may be by his curse alone he will attain his object--that is, convince himself that he is a man and not a piano-key!
# """

# inputs = ecco_model.tokenizer([text], return_tensors="pt")
# output = ecco_model(inputs)

# nmf_1 = output.run_nmf(n_components=8)

# ref: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/

visualisation_act = {}
# visualisation_grad = {}
# module_id_to_name = {}


# def hook_fn(m, i, o):
#     module_full_name = module_id_to_name[m]
#     current_tensor_list = visualisation_act.setdefault(module_full_name, [])
#     if len(current_tensor_list) == 0 or current_tensor_list[-1].shape >= o.shape:
#         current_tensor_list.append(o.clone().detach().cpu())
#     else:
#         current_tensor_list[-1] = o.clone().detach().cpu()


# def get_all_layers(net, parent_name=""):
#     if hasattr(net, "_modules"):
#         for name, layer in net._modules.items():
#             if (
#                 isinstance(layer, nn.Linear)
#                 or isinstance(layer, nn.LayerNorm)
#                 or isinstance(layer, nn.Embedding)
#                 or isinstance(layer, GELUActivation)
#             ):
#                 module_id_to_name[layer] = (parent_name + ".")[1:] + name
#                 layer.register_forward_hook(hook_fn)
#             elif isinstance(layer, GPTNeoXRotaryEmbedding):
#                 # TODO: support rotary embedding
#                 pass
#             else:
#                 get_all_layers(layer, parent_name + "." + name)


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

traced_file = "{}_traced.pt".format(network)
if os.path.exists(traced_file):
    logging.info("Load traced model...")
    scripted_model = torch.jit.load(traced_file)
else:
    logging.info("Trace model...")
    scripted_model = torch.jit.trace(
        model,
        dummy_inputs,
    )
    torch.jit.save(scripted_model, traced_file)
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
# NOTE: if the program has error, then remove log_file, vmc files and re-run codes start from auto-scheduler

exit()

get_all_layers(model)
model.train()

# end ref


logging.info("setup model")

query_orig = [
    "Explain to me the difference between nuclear fission and fusion.",
    "核融合反応と核分裂反応の違いを教えてください。",
    "請解釋核融合與核分裂之間的差異",
    "Please give an example of a 5-paragraph essay.",
    "請給我一段有起承轉合的文章",
]

# generate sequence ids for model input
model_inputs_list = []
for query in query_orig:
    model_inputs_list.extend([generate_text.preprocess(query)])

for model_inputs in model_inputs_list:
    logging.info(model_inputs["input_ids"].shape, model_inputs["attention_mask"].shape)
exit()

repeat_count = args.repeat_count

query = []
for item in query_orig:
    query.extend([item] * repeat_count)


assert batch_size <= len(query)

logging.info("generate text")

# res = generate_text(query[text_id])
# res = generate_text(query)
res = []
for text_id in tqdm(range(0, len(query), batch_size)):
    end = min(text_id + batch_size, len(query))
    res.extend(generate_text(query[text_id:end]))
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if param.grad is None:
    #             continue
    #         logging.info("name: {}, shape: {}".format(name, param.grad.shape))
    #         current_tensor_list = visualisation_grad.setdefault(name, [])
    #         current_tensor_list.append(param.grad.clone().detach().cpu().numpy())

logging.info("generate text")

# # find start pos of sequence of each query
# start_pos = []
# ref_tensor = next(iter(visualisation_act.items()))[1]
# for pos in range(len(ref_tensor)):
#     if ref_tensor[pos].shape[1] != 1:
#         start_pos.append(pos)

# assert len(start_pos) == len(query)
# start_pos.append(len(ref_tensor))
# logging.info("seq_length: {}".format(len(ref_tensor)))

for text_id in range(len(query)):
    logging.info("query text id: {}".format(text_id))
    logging.info(str(query[text_id]))

    logging.info("answer text id: {}".format(text_id))
    logging.info(str(res[text_id][0]["generated_text"]))

    logging.info("gradients text id: {}".format(text_id))

    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info("name: {}, shape: {}".format(name, param.grad.shape))

    logging.info("gradients text id: {}".format(text_id))

    logging.info("activations text id: {}".format(text_id))

    # TODO: support batch_size > 1

    visualisation_act_dump = {}

    # logging.info("seq start: {}, end: {}".format(start_pos[text_id], start_pos[text_id + 1] - 1))

    for key in visualisation_act.keys():
        current_act = visualisation_act[key]
        # TODO: support lotary embedding
        # output_tensors = torch.hstack(
        #     current_act[start_pos[text_id] : start_pos[text_id + 1]]
        # )
        output_tensors = current_act[text_id]
        visualisation_act_dump[key] = output_tensors.numpy()
        output_tensor_mean = torch.mean(output_tensors, dim=1).squeeze()
        output_tensor_vars = torch.var(output_tensors, dim=1).squeeze()
        logging.info("key: {}, shape: {}".format(key, output_tensors.shape))
        # print status along axis 1 in var order
        output_tensor_sort_index = list(
            torch.argsort(output_tensor_vars, dim=-1, descending=True).squeeze()
        )
        # print top 10 status
        for idx in output_tensor_sort_index[:10]:
            logging.info(
                "idx: {}, mean: {:.4f}, var: {:.8f}".format(
                    idx,
                    output_tensor_mean[idx],
                    output_tensor_vars[idx],
                )
            )

    logging.info("activations text id: {}".format(text_id))

    logging.info("save gradients text id: {}".format(text_id))

    with open("dolly-v2-3b" + "_" + str(int(start_time)) + "_grad.npy.plk", "wb") as f:
        pickle.dump(visualisation_grad, f)

    logging.info("save gradients text id: {}".format(text_id))

    logging.info("save activations text id: {}".format(text_id))

    with open(
        "dolly-v2-3b"
        + "_"
        + str(int(start_time))
        + "_"
        + str(text_id)
        + "_act.npy.plk",
        "wb",
    ) as f:
        pickle.dump(visualisation_act_dump, f)

    logging.info("save activations text id: {}".format(text_id))
