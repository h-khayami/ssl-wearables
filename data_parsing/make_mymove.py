import glob
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .utils import resize

DEVICE_HZ = 50  # Hz
WINDOW_SEC = 10  # seconds
WINDOW_OVERLAP_SEC = 0  # seconds
WINDOW_LEN = int(DEVICE_HZ * WINDOW_SEC)  # device ticks
WINDOW_OVERLAP_LEN = int(DEVICE_HZ * WINDOW_OVERLAP_SEC)  # device ticks
WINDOW_STEP_LEN = WINDOW_LEN - WINDOW_OVERLAP_LEN  # device ticks
WINDOW_TOL = 0.01  # 1%
TARGET_HZ = 30  # Hz
TARGET_WINDOW_LEN = int(TARGET_HZ * WINDOW_SEC)
DATAFILES = "/path/to/data/"
DATAFILES = DATAFILES + "mymove/path/to/accel/*.txt"
OUTDIR = f"mymove_30Hz_{WINDOW_SEC}s/"

#Accelerometer data, label, time, and pid
X, Y, T, P, = (
    [],
    [],
    [],
    [],
)

#TODO: write data loader for mymove data
for datafile in tqdm(glob.glob(DATAFILES)):
    # Load data
    data = pd.read_csv(datafile, delimiter='\t')
    X.append(data[['accel_x', 'accel_y', 'accel_z']].values)
    Y.append(data['label'].values)
    T.append(data['time'].values)
    P.append(data['pid'].values)

X = np.asarray(X)
Y = np.asarray(Y)
T = np.asarray(T)
P = np.asarray(P)

# fixing unit to g
X = X / 9.81
# downsample to 30 Hz
X = resize(X, TARGET_WINDOW_LEN)


os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X"), X)
np.save(os.path.join(OUTDIR, "Y"), Y)
np.save(os.path.join(OUTDIR, "time"), T)
np.save(os.path.join(OUTDIR, "pid"), P)

print(f"Saved in {OUTDIR}")
print("X shape:", X.shape)
print("Y distribution:", len(set(Y)))
print(pd.Series(Y).value_counts())
print("User distribution:", len(set(P)))
print(pd.Series(P).value_counts())