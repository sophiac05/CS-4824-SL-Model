"""
download_data.py  -  download ONLY the Kaggle ASL landmark files we need.

Needs train.csv (from the Kaggle asl-signs data page) in the same folder, and a
Kaggle API token (see README). Safe to re-run: files already on disk are skipped.

    python download_data.py

Writes:
    data/train_landmark_files/<participant_id>/<sequence_id>.parquet
    subset.csv      list of files selected (used by preprocess.py)
    failed.csv      only if some downloads failed; just re-run to retry them
"""
import os
import time
import zipfile

import pandas as pd

# ---- settings (keep SIGNS identical to preprocess.py) ----
SIGNS = ["hello", "thankyou", "please", "yes", "no",
         "drink", "look", "wait", "cat", "pizza"]
PER_SIGN = 400          # examples per target word (dataset has ~300-415 each)
OTHER_PER_SIGN = 5      # examples from every OTHER sign, for the "other" class
DATA_DIR = "data"
PAUSE = 0.3             # seconds between requests (avoids Kaggle rate limits)
# -----------------------------------------------------------


def select_files(df):
    """Pick the files to download. Fixed seed -> same selection on every machine."""
    main = df[df["sign"].isin(SIGNS)]
    main = pd.concat([g.sample(min(len(g), PER_SIGN), random_state=0)
                      for _, g in main.groupby("sign")])
    other = df[~df["sign"].isin(SIGNS)]
    other = pd.concat([g.sample(min(len(g), OTHER_PER_SIGN), random_state=0)
                       for _, g in other.groupby("sign")])
    sub = pd.concat([main, other])[["path", "participant_id", "sequence_id", "sign"]]
    return sub.reset_index(drop=True), len(main), len(other)


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    df = pd.read_csv("train.csv")
    missing = [s for s in SIGNS if s not in set(df["sign"])]
    if missing:
        raise SystemExit(f"These signs are not in train.csv (check spelling): {missing}")

    sub, n_main, n_other = select_files(df)
    sub.to_csv("subset.csv", index=False)
    print(f"{n_main} target-word files + {n_other} 'other' files selected")

    api = KaggleApi()
    api.authenticate()

    failed = []
    for i, p in enumerate(sub["path"]):
        final = os.path.join(DATA_DIR, p)
        if os.path.exists(final):
            continue
        time.sleep(PAUSE)

        for attempt in range(5):
            try:
                # Kaggle recreates train_landmark_files/<id>/ inside DATA_DIR itself
                api.competition_download_file("asl-signs", p, path=DATA_DIR, quiet=True)
                z = final + ".zip"
                if os.path.exists(z):
                    zipfile.ZipFile(z).extractall(os.path.dirname(final))
                    os.remove(z)
                if os.path.exists(final):
                    break
                raise FileNotFoundError("file not at expected path")
            except Exception as e:
                for leftover in (final, final + ".zip"):
                    if os.path.exists(leftover):
                        os.remove(leftover)
                print(f"retry {attempt + 1}/5 for {p}: {type(e).__name__}: {str(e)[:150]}")
                time.sleep(15 * (attempt + 1))
        else:
            failed.append(p)

        if i % 50 == 0:
            print(i, "checked")

    have = sum(os.path.exists(os.path.join(DATA_DIR, p)) for p in sub["path"])
    print(f"finished: {have}/{len(sub)} files present, {len(failed)} failed")
    if failed:
        pd.Series(failed, name="path").to_csv("failed.csv", index=False)
        print("failed paths saved to failed.csv; re-run this script to retry them")


if __name__ == "__main__":
    main()
