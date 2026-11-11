"""
train_gru.py  -  bidirectional GRU on ASL landmark sequences, with augmentation.

Run after preprocess.py, from the same folder:
    python -m pip install torch scikit-learn      (once; torch is a large download)
    python train_gru.py

What is better than the random-forest baseline:
  * keeps the order of frames (sequence model)
  * extra features: hand shape relative to each wrist, frame-to-frame motion
  * augmentation every epoch: mirror (left-handed signers), scale, rotate, shift,
    time-stretch, noise
  * signer-based train / validation / test split; the best epoch is chosen on the
    validation signers and scored once on the test signers
"""
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 0
EPOCHS = 80
BATCH = 64
LR = 2e-3

# Layout of the 48 landmarks saved by preprocess.py (each has x, y):
#   0-20 left hand | 21-41 right hand | 42-47 pose:
#   L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist


# ------------------------------------------------------------------ data helpers
def split_signers(pid, seed, test_frac=0.2, val_frac=0.15):
    rng = np.random.default_rng(seed)
    s = rng.permutation(np.unique(pid))
    n_test = max(1, int(round(len(s) * test_frac)))
    n_val = max(1, int(round(len(s) * val_frac)))
    test_mask = np.isin(pid, s[:n_test])
    val_mask = np.isin(pid, s[n_test:n_test + n_val])
    return ~(test_mask | val_mask), val_mask, test_mask


def mirror(x4):
    """Flip left<->right. x4 has shape (..., 48, 2)."""
    out = x4.copy()
    out[..., 0] *= -1
    lh = out[..., 0:21, :].copy()
    rh = out[..., 21:42, :].copy()
    out[..., 0:21, :] = rh
    out[..., 21:42, :] = lh
    for a, b in ((42, 43), (44, 45), (46, 47)):
        t = out[..., a, :].copy()
        out[..., a, :] = out[..., b, :]
        out[..., b, :] = t
    return out


def augment_one(x4, rng):
    """x4: (T, 48, 2) -> augmented copy. Undetected points (all zero) stay zero."""
    T = x4.shape[0]
    if rng.random() < 0.5:
        x4 = mirror(x4)
    # time stretch / crop
    L = int(rng.integers(int(0.7 * T), T + 1))
    start = int(rng.integers(0, T - L + 1))
    x4 = x4[np.linspace(start, start + L - 1, T).round().astype(int)]

    mask = (np.abs(x4).sum(-1, keepdims=True) > 0).astype(np.float32)
    ang = np.deg2rad(rng.uniform(-10, 10))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float32)
    x4 = (x4 @ R.T) * rng.uniform(0.85, 1.15)
    x4 = x4 + rng.uniform(-0.1, 0.1, size=(1, 1, 2)).astype(np.float32)
    x4 = x4 + rng.normal(0, 0.01, size=x4.shape).astype(np.float32)
    return (x4 * mask).astype(np.float32)


def features(x4):
    """x4: (N, T, 48, 2) -> (N, T, 278) features."""
    N, T = x4.shape[:2]
    pos = x4.reshape(N, T, 96)
    present = (np.abs(x4[:, :, [0, 21], :]).sum(-1) > 0)               # wrist detected? (N,T,2)
    lh, rh = x4[:, :, 0:21, :], x4[:, :, 21:42, :]
    rel_l = (lh - lh[:, :, :1, :]) * present[:, :, 0, None, None]
    rel_r = (rh - rh[:, :, :1, :]) * present[:, :, 1, None, None]
    rel = np.concatenate([rel_l, rel_r], axis=2).reshape(N, T, 84)     # hand shape
    vel = np.zeros_like(pos)
    vel[:, 1:] = pos[:, 1:] - pos[:, :-1]                              # motion
    return np.concatenate([pos, rel, vel, present.astype(np.float32)], axis=2).astype(np.float32)


# ------------------------------------------------------------------ model / training
def build_model(n_in, n_classes):
    import torch.nn as nn
    import torch

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.2))
            self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=0.3)
            self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, n_classes))

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.head(torch.cat([h.mean(1), h.max(1).values], dim=1))

    return Net()


def main():
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)

    X = np.load("processed/X.npy").reshape(-1, 32, 48, 2)
    y = np.load("processed/y.npy")
    pid = np.load("processed/participants.npy")
    labels = json.load(open("processed/labels.json"))
    n_cls = len(labels)

    tr, va, te = split_signers(pid, SEED)
    print(f"train {tr.sum()} | val {va.sum()} | test {te.sum()} examples "
          f"(split by signer: {len(np.unique(pid[tr]))}/{len(np.unique(pid[va]))}/{len(np.unique(pid[te]))})")

    Ftr_clean = features(X[tr])
    mu = Ftr_clean.mean((0, 1), keepdims=True)
    sd = Ftr_clean.std((0, 1), keepdims=True) + 1e-6
    norm = lambda F: torch.tensor((F - mu) / sd, dtype=torch.float32)
    Fva, Fte = norm(features(X[va])), norm(features(X[te]))

    counts = np.bincount(y[tr], minlength=n_cls).astype(np.float32)
    w = (counts.max() / np.maximum(counts, 1)) ** 0.5            # mild class balancing
    model = build_model(Ftr_clean.shape[2], n_cls).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev), label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * int(np.ceil(tr.sum() / BATCH)))

    def predict(F):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(F), 256):
                out.append(torch.softmax(model(F[i:i + 256].to(dev)), 1).cpu())
        return torch.cat(out).numpy()

    Xtr, ytr = X[tr], y[tr]
    best_f1, best_state = -1, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        Faug = norm(features(np.stack([augment_one(x, rng) for x in Xtr])))
        yt = torch.tensor(ytr)
        perm = torch.randperm(len(yt))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Faug[b].to(dev)), yt[b].to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        p = predict(Fva).argmax(1)
        f1 = f1_score(y[va], p, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss {loss.item():.3f}  val acc {accuracy_score(y[va], p):.3f}  "
                  f"val macro-F1 {f1:.3f}")

    model.load_state_dict(best_state)
    prob = predict(Fte)
    pred = prob.argmax(1)
    yte = y[te]
    present = sorted(set(yte) | set(pred))
    names = [labels[i] for i in present]
    print(f"\nTEST  accuracy {accuracy_score(yte, pred):.3f}   macro-F1 {f1_score(yte, pred, average='macro'):.3f}"
          f"   (best val macro-F1 {best_f1:.3f})\n")
    print(classification_report(yte, pred, labels=present, target_names=names, zero_division=0))
    cm = confusion_matrix(yte, pred, labels=present)
    print("confusion matrix (rows = true, columns = predicted):")
    print("             " + " ".join(f"{n[:6]:>6}" for n in names))
    for n, row in zip(names, cm):
        print(f"{n[:12]:12s} " + " ".join(f"{v:6d}" for v in row))
    np.savetxt("gru_confusion_matrix.csv", cm, fmt="%d", delimiter=",", header=",".join(names), comments="")

    # confidence threshold: low-confidence predictions become "other"
    other = labels.index("other")
    print("\nconfidence threshold (below it -> 'other'):")
    for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        q = np.where(prob.max(1) < t, other, pred)
        print(f"  {t:.1f}   accuracy {accuracy_score(yte, q):.3f}   macro-F1 {f1_score(yte, q, average='macro'):.3f}")

    torch.save({"state": best_state, "mu": mu, "sd": sd, "labels": labels}, "gru_model.pt")
    print("\nsaved gru_model.pt, gru_confusion_matrix.csv")


if __name__ == "__main__":
    main()