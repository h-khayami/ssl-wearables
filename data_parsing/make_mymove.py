import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import resize

CHECK_DATA = False # Set to True to check the data and plot histograms
DEVICE_HZ = 50  # Hz
WINDOW_SEC = 5  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
DATAPATH = "/data/har/raw/"
# DATAFILES = DATAPATH + "mymove/mymove_data_5s_acc_p*.npy"
# LABELFILES = DATAPATH + "mymove/mymove_data_5s_label_p*.npy"
OUTDIR = f"/data/ssl_wearable/data/downstream/mymove_30Hz_{WINDOW_SEC}s/"
participants = [f'p{i}' for i in range(1, 14)]
DATAFILES = [DATAPATH + f"mymove/mymove_data_5s_acc_{p}.npy" for p in participants]
LABELFILES = [DATAPATH + f"mymove/mymove_label_5s_acc_{p}.npy" for p in participants]
#Accelerometer data, label, time, and pid
X, Y, T, P, = (
    [],
    [],
    [],
    [],
)

#TODO: write data loader for mymove data
for datafile, labelfile in tqdm(zip(DATAFILES, LABELFILES)):
    # Load data
    # Extract participant id from the filename
    pid = os.path.basename(datafile).split('_')[-1].split('.')[0]
    data = np.load(datafile)
    label = np.load(labelfile)
    print(f"Loaded {datafile} with shape {data.shape}")
    print(f"Loaded {labelfile} with shape {label.shape}")
    P.extend([pid] * data.shape[0])
    X.append(data)
    Y.append(label)

# Concatenate all data along the first axis
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
    
    
X = np.asarray(X)
Y = np.asarray(Y)
T = np.asarray(T)
P = np.asarray(P)
if (CHECK_DATA):
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("P shape:", P.shape)
    print("X min:", X.min(),"X max:", X.max())
    import matplotlib.pyplot as plt

    unique_labels = np.unique(Y)
    colors = ['b', 'g', 'r']
    for label in unique_labels:
        label_indices = np.where(Y == label)[0]
        label_data = X[label_indices]
        label_data = label_data.reshape(-1, 3)
        # Calculate the Euclidean norm of the 3 axes
        euclidean_norm = np.linalg.norm(label_data, axis=1)
        # Create a 2x3 grid for subplots
        plt.figure(figsize=(15, 10))
        # Plot the histogram of the Euclidean norm
        plt.subplot(2, 3, (1, 3))
        # Plot the histogram of the Euclidean norm
        plt.hist(euclidean_norm, bins=50, alpha=0.75,color='purple', edgecolor='black')
        plt.title(f'Histogram of absolute values of Euclidean norm for {label}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        for i, axis in enumerate(['X', 'Y', 'Z']):
            plt.subplot(2, 3, i + 4)
            plt.hist(np.abs(label_data[:, i]), bins=50, alpha=0.75, color=colors[i],edgecolor='black')
            plt.title(f'{axis} axis for {label}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        # Save the figure
        plot_dir = os.path.join(os.path.dirname(__file__), "../plots/imgs/")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"MyMove Accel hist_{label}.png"))

# fixing unit to g
X = X / 9.81
# downsample to 30 Hz
X = resize(X, TARGET_WINDOW_LEN)
print("X shape:", X.shape)

os.makedirs(OUTDIR, exist_ok=True)
np.save(os.path.join(OUTDIR, "X"), X)
np.save(os.path.join(OUTDIR, "Y"), Y)
# np.save(os.path.join(OUTDIR, "time"), T)
np.save(os.path.join(OUTDIR, "pid"), P)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())