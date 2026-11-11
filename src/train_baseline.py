"""
train_baseline.py  -  random forest baseline on the preprocessed ASL landmarks.

Run after preprocess.py, from the same folder:
    python -m pip install scikit-learn      (once)
    python train_baseline.py

The split is BY SIGNER: some people are held out entirely for testing, so the
score reflects how the model does on a person it has never seen.
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 0
TEST_SIGNER_FRACTION = 0.25     # share of signers held out for the final test

X = np.load("processed/X.npy")                  # (N, 32, 96)
y = np.load("processed/y.npy")
pid = np.load("processed/participants.npy")
labels = json.load(open("processed/labels.json"))


def summarize(X):
    """Collapse the time axis: mean, std, min, max of every feature -> (N, 384)."""
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


# ---- signer-based split ----
rng = np.random.default_rng(SEED)
signers = rng.permutation(np.unique(pid))
n_test = max(1, int(round(len(signers) * TEST_SIGNER_FRACTION)))
test_signers = signers[:n_test]
test_mask = np.isin(pid, test_signers)

Xtr, ytr = summarize(X[~test_mask]), y[~test_mask]
Xte, yte = summarize(X[test_mask]), y[test_mask]
print(f"train: {len(ytr)} examples from {len(signers) - n_test} signers")
print(f"test : {len(yte)} examples from {n_test} unseen signers\n")

# ---- train ----
clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",     # stops the big "other" class from dominating
    random_state=SEED,
    n_jobs=-1,
)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte)

# ---- results ----
present = sorted(set(yte) | set(pred))
names = [labels[i] for i in present]
print(f"accuracy : {accuracy_score(yte, pred):.3f}")
print(f"macro F1 : {f1_score(yte, pred, average='macro'):.3f}  (treats every class equally)\n")
print(classification_report(yte, pred, labels=present, target_names=names, zero_division=0))

cm = confusion_matrix(yte, pred, labels=present)
print("confusion matrix (rows = true, columns = predicted):")
print("             " + " ".join(f"{n[:6]:>6}" for n in names))
for n, row in zip(names, cm):
    print(f"{n[:12]:12s} " + " ".join(f"{v:6d}" for v in row))

np.savetxt("baseline_confusion_matrix.csv", cm, fmt="%d", delimiter=",",
           header=",".join(names), comments="")
print("\nsaved baseline_confusion_matrix.csv")