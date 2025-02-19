import numpy as np
import os

DATA_DIR = "/data/ssl_wearable/data/downstream/"
IN_DIR = DATA_DIR + "wisdm_30hz_clean/"
OUTDIR = DATA_DIR + "wisdm_30hz_w10_few_classes/"

Classes_to_use = ["walking",
                "jogging",
                # "stairs",
                "sitting",
                "standing",
                # "typing",
                # "teeth",
                # "soup",
                # "chips",
                # "pasta",
                # "drinking",
                # "sandwich",
                # "kicking",
                # "catch",
                # "dribbling",
                # "writing",
                # "clapping",
                # "folding"
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