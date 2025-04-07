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

testing_datasets = activity_labels["testing_datasets"]
print(f"Testing datasets: {testing_datasets}")

def process_datasets(datasets, output_subdir):
    processed_P = np.array([])
    processed_X = np.array([])
    processed_Y = np.array([])

    for dataset in datasets:
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
        processed_P = np.concatenate((processed_P, P))
        if processed_X.size == 0:
            processed_X = X
        else:
            processed_X = np.concatenate((processed_X, X))
        processed_Y = np.concatenate((processed_Y, Y))

    processed_X = np.asarray(processed_X)
    processed_Y = np.asarray(processed_Y)
    # processed_T = np.asarray(processed_T)
    processed_P = np.asarray(processed_P)
    print(f"Processed data shape: {processed_X.shape}")
    print(f"Processed label shape: {processed_Y.shape}")
    print(f"Label distribution: {pd.Series(processed_Y).value_counts()}")
    print(f"Processed pid shape: {processed_P.shape}")
    # save processed data
    os.system(f"mkdir -p {OUT_DIR}/{output_subdir}/")
    np.save(os.path.join(OUT_DIR, f"{output_subdir}/X"), processed_X)
    np.save(os.path.join(OUT_DIR, f"{output_subdir}/Y"), processed_Y)
    # np.save(os.path.join(OUT_DIR, f"{output_subdir}/time"), processed_T)
    np.save(os.path.join(OUT_DIR, f"{output_subdir}/pid"), processed_P)

# Process training datasets
training_datasets = activity_labels["training_datasets"]
print(f"Training datasets: {training_datasets}")
process_datasets(training_datasets, "training")

# Process testing datasets
testing_datasets = activity_labels["testing_datasets"]
print(f"Testing datasets: {testing_datasets}")
process_datasets(testing_datasets, "testing")

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