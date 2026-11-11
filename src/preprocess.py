"""
preprocess.py  -  turn downloaded Kaggle ASL parquet files into training arrays.

Run from the same folder as train.csv, subset.csv and the data/ folder:
    python preprocess.py

Outputs (in ./processed):
    X.npy              (N, 32, 96)  float32  landmark sequences
    y.npy              (N,)         int64    class ids
    participants.npy   (N,)         int64    signer id (use this to split train/test!)
    labels.json        list of class names, index = class id (last one is "other")
"""
import os, json, warnings
import numpy as np
import pandas as pd

# ---- settings: keep SIGNS identical to the list in process.py ----
SIGNS = ["hello", "thankyou", "please", "yes", "no",
         "drink", "look", "wait", "cat", "pizza"]
T = 32                      # frames per sequence after resampling
DATA_DIR = "data"
OUT_DIR = "processed"
# -------------------------------------------------------------------

# Row layout of every Kaggle frame (543 landmarks, fixed order):
#   face 0-467 | left_hand 468-488 | pose 489-521 | right_hand 522-542
POSE_START = 489
KEEP = (list(range(468, 489))                         # left hand  (21)
        + list(range(522, 543))                       # right hand (21)
        + [POSE_START + i for i in range(11, 17)])    # shoulders, elbows, wrists (6)
L_SH, R_SH = POSE_START + 11, POSE_START + 12         # shoulder landmarks
N_LM = 543


def load_xy(path):
    """Read one parquet file -> (frames, 543, 2) array of x, y (NaN = not detected)."""
    df = pd.read_parquet(path, columns=["x", "y"])
    n = len(df) // N_LM
    if n == 0:
        return None
    return df[["x", "y"]].to_numpy(dtype=np.float32)[: n * N_LM].reshape(n, N_LM, 2)


def preprocess(xy):
    """Select landmarks, normalise, resample to T frames -> (T, 96)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")               # all-NaN slices are expected
        shoulders = xy[:, [L_SH, R_SH], :]
        center = np.nanmean(shoulders, axis=(0, 1))   # mid-shoulder point
        width = np.nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1))
    if np.isnan(center).any():
        center = np.array([0.5, 0.5], dtype=np.float32)
    if not np.isfinite(width) or width < 1e-3:
        width = 0.3

    pts = (xy[:, KEEP, :] - center) / width           # position/size invariant
    idx = np.linspace(0, len(pts) - 1, T).round().astype(int)
    pts = pts[idx]
    pts = np.nan_to_num(pts, nan=0.0)                 # undetected hand -> zeros
    return pts.reshape(T, -1).astype(np.float32)


def main():
    train = pd.read_csv("train.csv")
    # take only the path column from subset.csv, labels come from train.csv
    sub = pd.read_csv("subset.csv")[["path"]].drop_duplicates().merge(train, on="path")

    classes = SIGNS + ["other"]
    class_id = {c: i for i, c in enumerate(classes)}

    X, y, pid = [], [], []
    missing = bad = 0
    for k, row in enumerate(sub.itertuples(index=False)):
        f = os.path.join(DATA_DIR, row.path)
        if not os.path.exists(f):
            missing += 1
            continue
        try:
            xy = load_xy(f)
            if xy is None:
                bad += 1
                continue
            X.append(preprocess(xy))
        except Exception as e:
            bad += 1
            print("skipping", row.path, type(e).__name__)
            continue
        y.append(class_id[row.sign if row.sign in class_id else "other"])
        pid.append(row.participant_id)
        if k % 200 == 0:
            print(k, "/", len(sub))

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    pid = np.array(pid, dtype=np.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "participants.npy"), pid)
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as fh:
        json.dump(classes, fh)

    print(f"\nsaved X {X.shape}, y {y.shape}")
    print(f"skipped: {missing} not downloaded, {bad} unreadable")
    print("examples per class:")
    for i, c in enumerate(classes):
        print(f"  {c:12s} {int((y == i).sum())}")
    print("signers:", len(np.unique(pid)))


if __name__ == "__main__":
    main()