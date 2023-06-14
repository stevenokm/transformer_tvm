import pickle
import time
import sys
import os
import random

import numpy as np
from tqdm import tqdm
import argparse
import glob

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gmean
import pandas as pd

# from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_model",
    default="dolly-v2-3b",
    choices=["dolly-v2-3b", "dolly-v2-7b", "dolly-v2-12b"],
)
parser.add_argument(
    "--timestamp",
    type=str,
    default=time.time(),
)
args = parser.parse_args()

file_list = glob.glob(f"{args.input_model}_{args.timestamp}_*_act.npy.plk")

assert len(file_list) > 0, "No activation file found"

act = []

for file_name in file_list:
    with open(file_name, "rb") as f:
        act.append(pickle.load(f))

print(len(act))

# repeat_count = 2
# set repeat_count to the file count to perform full comparison
repeat_count = len(file_list)

data_dict = {}

for i in range(0, len(file_list), repeat_count):
    # diff 2 activation vector with cos-sim, feature by feature
    # summarize the difference with mean, var, max, min
    for layer in act[i].keys():
        print("query: {}".format(i))
        print("layer: {}".format(layer), end="")
        data_dict.setdefault("query", []).append(i)
        data_dict.setdefault("layer", []).append(layer)

        compare_timestamps = act[i][layer].shape[1]

        for j in range(repeat_count):
            current_id = i + j
            current_act = act[current_id][layer]
            print(", shape {}: {}".format(j, current_act.shape), end="")
            data_dict.setdefault("shape_{}".format(j), []).append(
                str(current_act.shape)
            )
            compare_timestamps = min(
                compare_timestamps, act[current_id][layer].shape[1]
            )
        print("")

        # batchsize, timestamps, features
        compare_tensors = act[i][layer][:, :compare_timestamps, :]
        for j in range(1, repeat_count, 1):
            current_id = i + j
            compare_tensors = np.concatenate(
                (compare_tensors, act[current_id][layer][:, :compare_timestamps, :]),
                axis=0,
            )
        # calculate pair-wise cos-sim for each time-series vectors
        # ref: https://stackoverflow.com/questions/17627219/whats-the-fastest-way-in-python-to-calculate-cosine-similarity-given-sparse-mat
        # transpose to (timestamps, batchsize, features)
        compare_tensors = np.transpose(compare_tensors, (1, 0, 2))
        similarities = None
        for t in range(compare_timestamps):
            # similarities shape: (1, C(batchsize, 2), C(batchsize, 2))
            similarity = np.expand_dims(cosine_similarity(compare_tensors[t]), axis=0)
            similarity = similarity + 1
            if similarities is None:
                similarities = similarity
            else:
                similarities = np.concatenate((similarities, similarity), axis=0)
        # calculate mean, var, max, min
        sim_mean = np.mean(similarities, axis=0)
        sim_var = np.var(similarities, axis=0)
        sim_max = np.max(similarities, axis=0)
        sim_min = np.min(similarities, axis=0)
        print("sim_mean:\n{}".format(sim_mean))
        print("sim_var:\n{}".format(sim_var))
        print("sim_max:\n{}".format(sim_max))
        print("sim_min:\n{}".format(sim_min))
        for offset in range(repeat_count):
            for j in range(offset + 1, repeat_count):
                source_id = offset
                target_id = j
                data_dict.setdefault(
                    "sim_mean_{}_{}".format(source_id, target_id), []
                ).append(sim_mean[offset, j])
                data_dict.setdefault(
                    "sim_var_{}_{}".format(source_id, target_id), []
                ).append(sim_var[offset, j])
                data_dict.setdefault(
                    "sim_max_{}_{}".format(source_id, target_id), []
                ).append(sim_max[offset, j])
                data_dict.setdefault(
                    "sim_min_{}_{}".format(source_id, target_id), []
                ).append(sim_min[offset, j])
        sim_triu_indeces = np.triu_indices(repeat_count, k=1)
        # NOTE: gmean(x + 1) - 1 to avoid negative values
        sim_mean_gmean = gmean(sim_mean[sim_triu_indeces] + 1) - 1
        sim_var_gmean = gmean(sim_var[sim_triu_indeces])
        sim_max_gmean = gmean(sim_max[sim_triu_indeces] + 1) - 1
        sim_min_gmean = gmean(sim_min[sim_triu_indeces] + 1) - 1
        print("sim_mean_gmean:\n{}".format(sim_mean_gmean))
        print("sim_var_gmean:\n{}".format(sim_var_gmean))
        print("sim_max_gmean:\n{}".format(sim_max_gmean))
        print("sim_min_gmean:\n{}".format(sim_min_gmean))
        data_dict.setdefault("sim_mean_gmean", []).append(sim_mean_gmean)
        data_dict.setdefault("sim_var_gmean", []).append(sim_var_gmean)
        data_dict.setdefault("sim_max_gmean", []).append(sim_max_gmean)
        data_dict.setdefault("sim_min_gmean", []).append(sim_min_gmean)

        # data-agnostic score: gmean(sim_mean + 1, , sim_max + 1, sim_min + 1)


# convert to pandas dataframe, export to csv
df = pd.DataFrame.from_dict(data_dict)
df.to_csv("{}_{}_act.csv".format(args.input_model, args.timestamp))
