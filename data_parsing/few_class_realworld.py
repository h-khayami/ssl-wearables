import numpy as np
import os

DATA_DIR = "/data/ssl_wearable/data/downstream/"
IN_DIR = DATA_DIR + "realworld_30hz_clean/"
OUTDIR = DATA_DIR + "realworld_30hz_few_class/"

Classes_to_use = [
    # "jumping",
    # "climbingup",
    # "climbingdown",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]

X = np.load(os.path.join(IN_DIR, "X.npy"))
Y = np.load(os.path.join(IN_DIR, "Y.npy"))
# T = np.load(os.path.join(IN_DIR, "time.npy"))
P = np.load(os.path.join(IN_DIR, "pid.npy"))
print(f"Loaded data with shape {X.shape}")
print(f"Loaded label with shape {Y.shape}")

mask = np.isin(Y, Classes_to_use)
X = X[mask]
Y = Y[mask]
# T = T[mask]
P = P[mask]
print(f"Filtered data with shape {X.shape}")
print(f"Filtered label with shape {Y.shape}")

os.system(f"mkdir -p {OUTDIR}")
np.save(os.path.join(OUTDIR, "X"), X)
np.save(os.path.join(OUTDIR, "Y"), Y)
# np.save(os.path.join(OUTDIR, "time"), T)
np.save(os.path.join(OUTDIR, "pid"), P)