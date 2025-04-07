import numpy as np
import os
import json
import pandas as pd
from Dataset_Label_Mapper import DatasetLabelMapper

DATA_DIR = "/data/ssl_wearable/data/downstream/"
OUT_DIR = DATA_DIR + "cross_dataset_mapping"
# read activity label mapping config json file
CONFIG_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(CONFIG_DIR, "../conf/")
with open(os.path.join(CONFIG_DIR, "cross_dataset_mapping/Activity_label_mapping v2 5c.json"), "r") as f:
    activity_labels = json.load(f)

mapper = DatasetLabelMapper(os.path.join(CONFIG_DIR, "cross_dataset_mapping/Activity_label_mapping v2 5c.json"))
training_datasets = activity_labels["training_datasets"]
testing_datasets = activity_labels["testing_datasets"]
print(f"Training datasets: {training_datasets}")
print(f"Testing datasets: {testing_datasets}")

training_P = np.array([])
training_X = np.array([])
training_Y = np.array([])

for dataset in training_datasets:
    dataset_dir = os.path.join(DATA_DIR, activity_labels["dataset_directories"][dataset])
    print(f"Dataset dir: {dataset_dir}")
    Classes_to_use = list(activity_labels["dataset_mappings"][dataset].keys())
    print(f"Classes to use: {Classes_to_use}")
    X = np.load(os.path.join(dataset_dir, "X.npy"))
    Y = np.load(os.path.join(dataset_dir, "Y.npy"))
    # T = np.load(os.path.join(dataset_dir, "time.npy"))
    P = np.load(os.path.join(dataset_dir, "pid.npy"))
    print(f"Loaded data with shape {X.shape}")
    print(f"Loaded label with shape {Y.shape}")
    mask = np.isin(Y, Classes_to_use)
    X = X[mask]
    Y = Y[mask]
    # T = T[mask]
    P = P[mask]
    # add dataset name to the pid
    P = [f"{dataset}_{pid}" for pid in P]
    print(f"Filtered data with shape {X.shape}")
    print(f"Filtered label with shape {Y.shape}")
    print(f"Label distribution: {pd.Series(Y).value_counts()}")
    Y = mapper.map_to_reference_labels(Y, dataset)
    training_P = np.concatenate((training_P, P))
    if training_X.size == 0:
        training_X = X
    else:
        training_X = np.concatenate((training_X, X))
    training_Y = np.concatenate((training_Y, Y))

training_X = np.asarray(training_X)
training_Y = np.asarray(training_Y)
# training_T = np.asarray(training_T)
training_P = np.asarray(training_P)
print(f"Training data shape: {training_X.shape}")
print(f"Training label shape: {training_Y.shape}")
print(f"Label distribution: {pd.Series(training_Y).value_counts()}")
print(f"Training pid shape: {training_P.shape}")
# save training data
os.system(f"mkdir -p {OUT_DIR}/training/")
np.save(os.path.join(OUT_DIR, "training/X"), training_X)
np.save(os.path.join(OUT_DIR, "training/Y"), training_Y)
# np.save(os.path.join(OUT_DIR, "training/time"), training_T)
np.save(os.path.join(OUT_DIR, "training/pid"), training_P)

# X = np.load(os.path.join(IN_DIR, "X.npy"))
# Y = np.load(os.path.join(IN_DIR, "Y.npy"))
# # T = np.load(os.path.join(IN_DIR, "time.npy"))
# P = np.load(os.path.join(IN_DIR, "pid.npy"))
# print(f"Loaded data with shape {X.shape}")
# print(f"Loaded label with shape {Y.shape}")

# mask = np.isin(Y, Classes_to_use)
# X = X[mask]
# Y = Y[mask]
# # T = T[mask]
# P = P[mask]
# print(f"Filtered data with shape {X.shape}")
# print(f"Filtered label with shape {Y.shape}")

# os.system(f"mkdir -p {OUTDIR}")
# np.save(os.path.join(OUTDIR, "X"), X)
# np.save(os.path.join(OUTDIR, "Y"), Y)
# # np.save(os.path.join(OUTDIR, "time"), T)
# np.save(os.path.join(OUTDIR, "pid"), P)