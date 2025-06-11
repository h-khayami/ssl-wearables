import glob
"""
This script processes accelerometer data from multiple participants, downsampling it to a target frequency and saving the processed data.
The script performs the following steps:
1. Loads accelerometer data and labels from .npy files for each participant.
2. Concatenates the data from all participants.
3. Optionally checks the data and plots histograms of the Euclidean norm and individual axes.
4. Converts the data units to g (assuming the data is in m/s^2).
5. Downsamples the data to a target frequency.
6. Saves the processed data to the specified output directory.
Constants:
- CHECK_DATA: Boolean flag to check the data and plot histograms.
- DEVICE_HZ: Sampling frequency of the device in Hz.
- WINDOW_SEC: Window size in seconds.
- WINDOW_OVERLAP_SEC: Overlap between windows in seconds.
- WINDOW_LEN: Length of the window in device ticks.
- WINDOW_OVERLAP_LEN: Length of the window overlap in device ticks.
- WINDOW_STEP_LEN: Step length between windows in device ticks.
- WINDOW_TOL: Tolerance for windowing.
- TARGET_HZ: Target frequency for downsampling in Hz.
- TARGET_WINDOW_LEN: Length of the target window in ticks.
- DATAPATH: Path to the raw data directory.
- DATAFILES: List of paths to the accelerometer data files.
- LABELFILES: List of paths to the label files.
- OUTDIR: Output directory for the processed data.
- participants: List of participant identifiers.
Variables:
- X: List to store accelerometer data.
- Y: List to store labels.
- T: List to store timestamps (currently unused).
- P: List to store participant IDs.
Functions:
- main(): Parses command-line arguments and updates window parameters.
Example usage:
    python make_mymove.py --window_size 5 --check_data --device_hz 50
    python make_mymove.py --window_size 10 --device_hz 30
"""
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils import resize
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size in seconds")
    parser.add_argument("--check_data", action="store_true", help="Check the data and plot histograms")
    parser.add_argument("--device_hz", type=int, default=50, help="Sampling frequency of the device in Hz")
    parser.add_argument("--five_classes", action="store_true", help="Use the 5-class label set")
    parser.add_argument("--younger", action="store_true", help="Flag to indicate if the participant is younger")
    parser.add_argument("--all", action="store_true", help="Flag to include both younger and older participants")
    args = parser.parse_args()
    # if both --younger and --all are set, raise an error
    if args.younger and args.all:
        raise ValueError("Cannot use both --younger and --all flags together. Use --younger for younger participants only. Use --all for all participants.")
    
    global CHECK_DATA, DEVICE_HZ, WINDOW_SEC, WINDOW_LEN, WINDOW_STEP_LEN, TARGET_WINDOW_LEN
    CHECK_DATA = args.check_data
    DEVICE_HZ = args.device_hz
    WINDOW_SEC = args.window_size
    WINDOW_OVERLAP_SEC = 0  # seconds
    WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
    WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
    WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
    WINDOW_TOL = 0.01  # 1%
    TARGET_HZ = 30  # Hz
    TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
    Classes_to_use = ['sedentary_lying',
                      'sedentary_sitting_other', 
                      'sedentary_sitting_transport', 
                      'upright_standing', 
                      'upright_stepping_low', 
                      'upright_stepping_moderate', 
                      'upright_stepping_vigorous', 
                      'cycling'
                      ]
    DATAPATH = "/data/har/raw/mymove/"
    # DATAFILES = DATAPATH + "mymove/mymove_data_5s_acc_p*.npy"
    # LABELFILES = DATAPATH + "mymove/mymove_data_5s_label_p*.npy"
    OUTDIR = f"/data/ssl_wearable/data/downstream/mymove_30Hz_{WINDOW_SEC}s/"
    if args.five_classes:
        OUTDIR = OUTDIR.rstrip('/') + "_5c/"
    
    older_participants = [f'p{i}' for i in range(1, 14)] # older participants
    younger_participants = [f'y{i}' for i in range(1, 3)]  # younger participants
    if args.all:
        OUTDIR = OUTDIR.rstrip('/') + "_all/"
    elif args.younger and not args.all:
        OUTDIR = OUTDIR.rstrip('/') + "_y/"
    try:
        if (WINDOW_SEC == 5 and DEVICE_HZ == 50):
            DATAFILES = [DATAPATH + f"mymove_data_5s_acc_{p}.npy" for p in older_participants]
            LABELFILES = [DATAPATH + f"mymove_label_5s_acc_{p}.npy" for p in older_participants]
        elif (WINDOW_SEC == 5 and DEVICE_HZ == 30):
            DATAFILES = [DATAPATH + f"np_data/mymove_data_acc_5sec_30hz_{p}.npy" for p in older_participants]
            LABELFILES = [DATAPATH + f"np_data/mymove_label_acc_5sec_30hz_{p}.npy" for p in older_participants]
        elif (WINDOW_SEC == 10 and DEVICE_HZ == 30):
            if args.all:
                DATAFILES = [DATAPATH + f"np_data/mymove_data_acc_10sec_30hz_{p}.npy" for p in older_participants]
                LABELFILES = [DATAPATH + f"np_data/mymove_label_acc_10sec_30hz_{p}.npy" for p in older_participants]
                DATAFILES.extend([DATAPATH + f"np_data_young_adult/mymove_data_acc_10sec_30hz_{p}.npy" for p in younger_participants])
                LABELFILES.extend([DATAPATH + f"np_data_young_adult/mymove_label_acc_10sec_30hz_{p}.npy" for p in younger_participants])
            elif args.younger:
                DATAFILES = [DATAPATH + f"np_data_young_adult/mymove_data_acc_10sec_30hz_{p}.npy" for p in younger_participants]
                LABELFILES = [DATAPATH + f"np_data_young_adult/mymove_label_acc_10sec_30hz_{p}.npy" for p in younger_participants]
            else:
                DATAFILES = [DATAPATH + f"np_data/mymove_data_acc_10sec_30hz_{p}.npy" for p in older_participants]
                LABELFILES = [DATAPATH + f"np_data/mymove_label_acc_10sec_30hz_{p}.npy" for p in older_participants]
        else:
            raise ValueError(f"Unavailable window size (={WINDOW_SEC}) and device frequency (={DEVICE_HZ}) combination")
    except Exception as e:
        print(f"Error: {e}")
        return
    #Accelerometer data, label, time, and pid
    X, Y, T, P, = (
        [],
        [],
        [],
        [],
    )

    for datafile, labelfile in tqdm(zip(DATAFILES, LABELFILES)):
        # Load data
        # Extract participant id from the filename
        pid = os.path.basename(datafile).split('_')[-1].split('.')[0]
        data = np.load(datafile)
        label = np.load(labelfile)
        print(f"Loaded {datafile} with shape {data.shape}")
        print(f"Loaded {labelfile} with shape {label.shape}")
        # Remove the labels and the associated data that are not in the Classes_to_use
        if args.five_classes:
            mask = np.isin(label, Classes_to_use)
            data = data[mask]
            label = label[mask]
            print(f"Filtered data with shape {data.shape}")
            print(f"Filtered label with shape {label.shape}")
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
        print(np.unique(Y))
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

    # fixing unit to g (assuming that the data is in m/s^2)
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
    return
if __name__ == "__main__":
    main()