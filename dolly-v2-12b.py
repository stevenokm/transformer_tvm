import pickle
import time
import sys
import torch
import torch.nn as nn
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import GELUActivation
from transformers.models.gpt_neox.modeling_gpt_neox import (
    RotaryEmbedding,
    GPTNeoXModel,
    GPTNeoXMLP,
)

start_time = time.time()
if len(sys.argv) >= 2:
    start_time = float(sys.argv[1])
prev_time = start_time


def log(text):
    print(text)
    sys.stdout.flush()


log(str(start_time))

log("setup model")

tokenizer = AutoTokenizer.from_pretrained(
    "databricks/dolly-v2-12b", padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    "databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16
)

# ref: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/

visualisation_act = {}
visualisation_grad = {}
module_id_to_name = {}


def hook_fn(m, i, o):
    module_full_name = module_id_to_name[m]
    visualisation_act.setdefault(module_full_name, []).append(o)


def get_all_layers(net, parent_name=""):
    if hasattr(net, "_modules"):
        for name, layer in net._modules.items():
            if (
                isinstance(layer, nn.Linear)
                or isinstance(layer, nn.LayerNorm)
                or isinstance(layer, nn.Embedding)
                or isinstance(layer, GELUActivation)
            ):
                module_id_to_name[layer] = (parent_name + ".")[1:] + name
                layer.register_forward_hook(hook_fn)
            elif isinstance(layer, RotaryEmbedding):
                # TODO: support rotary embedding
                pass
            else:
                get_all_layers(layer, parent_name + "." + name)


get_all_layers(model)

# end ref

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)


def log_time(title_name):
    global prev_time
    global start_time

    current_time = time.time()
    step_time = current_time - prev_time
    wall_time = current_time - start_time
    log("{} time: step: {:.4f} wall: {:.4f}".format(title_name, step_time, wall_time))
    prev_time = current_time


log_time("setup model")

query = [
    "Explain to me the difference between nuclear fission and fusion.",
    "Explain to me the difference between nuclear fission and fusion.",
    "核融合反応と核分裂反応の違いを教えてください。",
    "核融合反応と核分裂反応の違いを教えてください。",
    "請解釋核融合與核分裂之間的差異",
    "請解釋核融合與核分裂之間的差異",
    "Please give an example of a 5-paragraph essay.",
    "Please give an example of a 5-paragraph essay.",
    "請給我一段有起承轉合的文章",
    "請給我一段有起承轉合的文章",
]

log("generate text")

# res = generate_text(query[text_id])
res = generate_text(query)

log_time("generate text")
log("")

# find start pos of sequence of each query
start_pos = []
ref_tensor = next(iter(visualisation_act.items()))[1]
for pos in range(len(ref_tensor)):
    if ref_tensor[pos].shape[1] != 1:
        start_pos.append(pos)

assert len(start_pos) == len(query)
start_pos.append(len(ref_tensor))
log("seq_length: {}".format(len(ref_tensor)))

for text_id in range(len(query)):
    log("")
    log("query text id: {}".format(text_id))
    log(query[text_id])

    log("answer text id: {}".format(text_id))
    log(res[text_id][0]["generated_text"])

    log("")

    log("gradients text id: {}".format(text_id))

    for name, param in model.named_parameters():
        if param.requires_grad:
            log("name: {}, shape: {}".format(name, param.shape))
            visualisation_grad[name] = param.grad

    log_time("gradients text id: {}".format(text_id))

    log("activations text id: {}".format(text_id))

    # TODO: support batch_size > 1

    visualisation_act_dump = {}

    log("seq start: {}, end: {}".format(start_pos[text_id], start_pos[text_id + 1] - 1))

    for key in visualisation_act.keys():
        current_act = visualisation_act[key]
        # TODO: support lotary embedding
        output_tensors = torch.hstack(
            current_act[start_pos[text_id] : start_pos[text_id + 1]]
        )
        visualisation_act_dump[key] = output_tensors
        output_tensor_mean = torch.mean(output_tensors, dim=1).squeeze()
        output_tensor_vars = torch.var(output_tensors, dim=1).squeeze()
        log("key: {}, shape: {}".format(key, output_tensors.shape))
        # print status along axis 1 in var order
        output_tensor_sort_index = list(
            torch.argsort(output_tensor_vars, dim=-1, descending=True).squeeze()
        )
        # print top 10 status
        for idx in output_tensor_sort_index[:10]:
            log(
                "idx: {}, mean: {:.4f}, var: {:.8f}".format(
                    idx,
                    output_tensor_mean[idx],
                    output_tensor_vars[idx],
                )
            )

    log_time("activations text id: {}".format(text_id))

    log("save gradients text id: {}".format(text_id))

    with open("dolly-v2-12b_" + str(int(start_time)) + "_grad.plk", "wb") as f:
        pickle.dump(visualisation_grad, f)

    log_time("save gradients text id: {}".format(text_id))

    log("save activations text id: {}".format(text_id))

    with open(
        "dolly-v2-12b_" + str(int(start_time)) + "_" + str(text_id) + "_act.plk", "wb"
    ) as f:
        pickle.dump(visualisation_act_dump, f)

    log_time("save activations text id: {}".format(text_id))
