# ASL Word Recognition (CS 4824 Machine Learning Project)

Real-time American Sign Language (ASL) word recognition from a webcam. A webcam feed is turned
into hand and body landmarks with MediaPipe, and a machine learning model classifies the sign into
a word shown as text.

**Current scope:** a small fixed vocabulary (10 words) plus an `other` class for anything else.
Open-vocabulary, sentence-level translation is future work, not a goal of this phase.

```
webcam video -> MediaPipe landmarks -> preprocessing -> classifier -> word on screen
```

## Team

| Name | Role |
|---|---|
| Daania Abbas | Research lead, model lead |
| Sophia Clark | Data collection, research lead |
| Giovanni Garcia | Model lead |
| Sid Gvalani | Project manager, data collection |
| Srikar Nagoti | Model lead |

## Timeline

| Date | Milestone |
|---|---|
| Nov 20 | Finalize model: test, tune and evaluate models, analyze errors |
| Dec 3 | Course project presentation and live demo |
| Dec 8 | Final report: problem, methods, results and quantitative analysis |

## Repository contents

| File | What it does |
|---|---|
| `download_data.py` | Downloads only the Kaggle files we need (not the full 57 GB) and writes `subset.csv` |
| `preprocess.py` | Turns the downloaded parquet files into fixed-size arrays in `processed/` |
| `train_baseline.py` | Random forest baseline (the number to beat) |
| `train_gru.py` | Bidirectional GRU with augmentation |
| `.gitignore` | Keeps data, models and tokens out of git |

Not committed (created locally by the scripts): `train.csv`, `subset.csv`, `data/`, `processed/`,
`gru_model.pt`, `*_confusion_matrix.csv`.

## Setup (Windows PowerShell)

Requires Python 3.10 or newer.

```powershell
python -m pip install pandas numpy pyarrow kaggle scikit-learn torch
```

`torch` is a large download. If it fails to install, run the training scripts in Google Colab instead.

### 1. Kaggle account and token (every teammate does this separately)

1. Create a Kaggle account and verify your phone number (Kaggle requires this for competition data).
2. Open <https://www.kaggle.com/competitions/asl-signs/rules> and accept the rules while logged in.
3. Go to <https://www.kaggle.com/settings/api>, click **Generate New Token**, and save it:

```powershell
mkdir $HOME\.kaggle -Force
Set-Content -Path $HOME\.kaggle\access_token -Value "PASTE_YOUR_TOKEN_HERE"
```

**Never commit a token, paste it in chat or a screenshot, or share it between teammates.** If one leaks,
revoke it on the same Kaggle page and generate a new one.

### 2. Get `train.csv`

On the competition's Data tab (<https://www.kaggle.com/competitions/asl-signs/data>), click the download
arrow on the `train.csv` card (6 MB). **Do not click "Download All"** (about 57 GB). Put `train.csv`
in the repo folder.

### 3. Download the data we use

```powershell
python download_data.py
```

With the default settings this fetches about 3,400 files for our 10 words plus 1,200 `other` files, roughly
3 GB. It takes a while, retries on dropped connections, and is safe to re-run: it skips files it already
has. Check progress from a second window:

```powershell
python -c "import glob; print(len(glob.glob('data/train_landmark_files/*/*.parquet')), 'files')"
```

### 4. Preprocess

```powershell
python preprocess.py
```

Writes `processed/X.npy`, `y.npy`, `participants.npy` and `labels.json`. Check that the printed
"examples per class" looks right.

### 5. Train

```powershell
python train_baseline.py
python train_gru.py
```

## The data

We use the Kaggle competition **Google - Isolated Sign Language Recognition**
(<https://www.kaggle.com/competitions/asl-signs>): about 94,000 recordings of 250 single ASL signs from
21 signers, already converted to MediaPipe landmarks (no video). Check the competition's rules for the
license and allowed use.

- `train.csv` has one row per recording: `path`, `participant_id`, `sequence_id`, `sign`.
- The label is **not** in the file name. `train_landmark_files/<participant_id>/<sequence_id>.parquet`
  is looked up in `train.csv` to get the sign.
- Each parquet file has one row per landmark per frame (`frame`, `row_id`, `type`, `landmark_index`,
  `x`, `y`, `z`), 543 landmarks per frame. Undetected landmarks are NaN.
- Landmark order within a frame: face 0-467, left hand 468-488, pose 489-521, right hand 522-542.

### Our vocabulary

`hello, thankyou, please, yes, no, drink, look, wait, cat, pizza` plus `other`.

`other` is built from a small random sample of the remaining 240 signs and is one lumped class. The model
cannot say *which* other word was signed, only that it is not one of ours.

**To change the vocabulary,** edit `SIGNS` in **both** `download_data.py` and `preprocess.py` (they must be
identical), then re-run the download and preprocessing. Avoid near-identical pairs (for example
`listen`/`hear`, `wake`/`awake`) and watch the confusion matrix for look-alikes.

## What preprocessing does

For each recording, `preprocess.py`:

1. Keeps both hands (21 points each) and 6 upper-body pose points (shoulders, elbows, wrists), using x and y
   only. The 468 face points are dropped.
2. Centers on the mid-shoulder point and scales by shoulder width, so position and distance from the camera
   do not matter.
3. Resamples to a fixed 32 frames.
4. Replaces undetected points with zeros.

Output shapes: `X` is `(N, 32, 96)`, where 96 = 48 landmarks x 2 coordinates.

## Models

- **Baseline** (`train_baseline.py`): random forest on per-feature mean, std, min and max over time. It
  ignores frame order, so it is a floor, not the goal.
- **GRU** (`train_gru.py`): 2-layer bidirectional GRU on the full sequence. Adds hand-shape features
  (landmarks relative to each wrist) and motion features (frame differences). Each epoch applies
  augmentation: left/right mirroring, rotation, scale, shift, time stretch and noise. Also prints a
  confidence-threshold table (low-confidence predictions become `other`).

## Evaluation rules (please follow these so results are comparable)

- **Always split by signer** (`participants.npy`), never by random row. Random splits put the same person
  in train and test and make scores look better than they are. Both training scripts already do this
  with `SEED = 0`.
- **Report macro-F1 and per-word recall, not only accuracy.** `other` is a large class, so accuracy alone
  can look fine when the target words are weak.
- Test sets are small (a handful of signers). Rerun with a different `SEED` or use signer-grouped
  cross-validation before drawing conclusions from a difference of a few points.
- Do not tune on the test signers. The GRU picks its best epoch on the validation signers only.

## Results

Fill this table in as experiments finish. Keep the settings next to each number.

| Model | Data per word | Test accuracy | Test macro-F1 | Notes |
|---|---|---|---|---|
| Random forest baseline | 50 | 0.345 | 0.353 | 5 held-out signers; `other` recall was low |
| GRU | 50 | TBD | TBD | |
| GRU | ~400 | TBD | TBD | |

## Live demo notes

The model only works if live input matches the training format exactly:

- Build each frame as a 543-landmark array in the **same order** as above (face, left hand, pose, right hand),
  with NaN for anything MediaPipe does not detect, then apply the same steps as `preprocess.py`.
- Check left/right: webcams are often mirrored ("selfie view"). Verify on a few signs that left and right hands
  are not swapped relative to the training data.
- Keep a 32-frame rolling window and only show a word when the model's confidence is above a threshold.
- Public data rarely matches our own camera, lighting and background. Record some of our own examples of
  each word and test on them. That number tells us whether the demo will work.

## Troubleshooting

| Problem | Fix |
|---|---|
| `kaggle` is not recognized | Not needed. The scripts use the Python library, not the command. |
| Authentication required | Token not saved. Re-run the `Set-Content` step, or set `$env:KAGGLE_API_TOKEN` in the same window. |
| 403 / join competition | Accept the rules at the competition's Rules page, on the same account that made the token. |
| `SSLError` / `HTTPError` during download | Usually a dropped connection or rate limit. Just re-run `download_data.py`. Try a different network or turn off VPN. |
| Files nested as `...\16069\train_landmark_files\16069\...` | Old download scripts caused this. Use `download_data.py` from this repo. |
| `ImportError` while preprocessing | Run `python -m pip install pyarrow`. |
| `PermissionError` on `subset.csv` | Another program (Excel, OneDrive sync) has it open. Close it, or work outside OneDrive. |
| OneDrive slows or locks files | Keep the project outside OneDrive, for example `C:\asl`. |

## Working together

- Each person downloads their own copy of the data with their own Kaggle token, or one person zips
  `data/` and `subset.csv` and shares it on the team drive. The selection uses a fixed seed, so every
  machine gets the same files.
- Do not commit `data/`, `processed/`, model files or any token. `.gitignore` covers them, but check
  `git status` before you push.
- Use branches, one per experiment (for example `model/transformer`, `demo/webcam`), and put results in
  the table above with a short note on settings.
- If you change preprocessing, tell everyone: all saved arrays and models must be regenerated.

## References

- Google - Isolated Sign Language Recognition (Kaggle): <https://www.kaggle.com/competitions/asl-signs>
- MediaPipe Holistic: <https://github.com/google-ai-edge/mediapipe>
- Project proposal references, including WLASL (Li et al., WACV 2020) and Sign Language Transformers
  (Camgoz et al., CVPR 2020).
