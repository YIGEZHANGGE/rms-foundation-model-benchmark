"""
v3 Diagnosis Benchmark: 5 encoders, frozen n=390 cohort, fixed folds.

Design:
- Reads rms_output_v3/cohort_and_folds.json as the single source of truth.
- For each encoder, trains ABMIL+MAMMOTH per fold.
- Uses an inner stratified split of the training fold for early stopping,
  then evaluates on the frozen outer validation fold (no double dipping).
- Deterministic patch sampling with per-encoder/per-fold RandomState.
- Saves checkpoints, OOF predictions/probabilities, and per-encoder JSON.
"""

import os, sys, json, glob, time, logging, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
EPOCHS = 60
PATIENCE = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 2.0
MAX_PATCHES = 4096
N_FOLDS = 5
INNER_VAL_FRAC = 0.2

PROJECT = r"C:\Users\onekey\WorkBuddy\20260330120149"
OUT_DIR = os.path.join(PROJECT, "rms_output_v3")
COHORT_PATH = os.path.join(OUT_DIR, "cohort_and_folds.json")
LOG_PATH = os.path.join(OUT_DIR, "train_diagnosis_v3.log")

os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("v3_diag")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

ENCODERS = {
    "conch":     (r"F:\Pathology\features_rms_conch", 768),
    "conch15":   (r"F:\Pathology\features\features_cog_conch15", 768),
    "phikon":    (r"F:\Pathology\features\features_cog_phikon", 1024),
    "hoptimus0": (r"F:\Pathology\features\features_cog_hoptimus0", 1536),
    "gigapath":  (r"F:\Pathology\features\features_cog_gigapath", 1536),
}


class MAMMOTH(nn.Module):
    def __init__(self, dim, slots=8, heads=4, experts=4, hidden=384):
        super().__init__()
        self.slot_proj = nn.Linear(dim, slots * heads)
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden, hidden)) for _ in range(experts)]
        )
        self.proj = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        S, H = 8, 4
        logits = self.slot_proj(x).view(B, N, S, H)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), -1e9)
        w = F.softmax(logits, dim=1)
        slots = torch.einsum("bnsh,bnd->bshd", w, x).mean(2)
        r = F.softmax(self.router(slots), dim=-1)
        out = sum(e(slots) * r[:, :, i:i+1] for i, e in enumerate(self.experts)).mean(1)
        return self.norm(self.proj(out))


class ABMIL(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        a = self.attn(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        a = F.softmax(a, dim=1)
        return self.proj((a.unsqueeze(-1) * x).sum(1)), a


class Classifier(nn.Module):
    def __init__(self, dim, n_classes=3):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.mammoth = MAMMOTH(dim)
        self.abmil = ABMIL(dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x, mask=None, return_feat=False):
        x = self.proj_in(x)
        x = x + self.mammoth(x, mask).unsqueeze(1)
        feat, attn = self.abmil(x, mask)
        out = self.head(feat)
        if return_feat:
            return out, feat, attn
        return out, attn


class FeatureBagDataset(Dataset):
    def __init__(self, pids, labels, feat_files, dim, max_p=MAX_PATCHES, seed=SEED):
        self.pids = pids
        self.labels = labels
        self.feat_files = feat_files
        self.dim = dim
        self.max_p = max_p
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        pid = self.pids[i]
        with h5py.File(self.feat_files[pid], "r") as hf:
            feats = hf["features"][:].astype(np.float32)
        if feats.shape[0] == 0:
            raise ValueError(f"Empty features: {pid}")
        if feats.shape[0] > self.max_p:
            idxs = self.rng.choice(feats.shape[0], self.max_p, replace=False)
            feats = feats[idxs]
        return torch.FloatTensor(feats), torch.LongTensor([self.labels[i]]), pid


def collate(batch):
    feats, labels, pids = zip(*batch)
    max_n = max(f.shape[0] for f in feats)
    d = feats[0].shape[1]
    padded = torch.zeros(len(feats), max_n, d)
    mask = torch.zeros(len(feats), max_n, dtype=torch.bool)
    for i, f in enumerate(feats):
        padded[i, :f.shape[0]] = f
        mask[i, :f.shape[0]] = True
    return padded, mask, torch.stack(labels), list(pids)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_fold(encoder, dim, feat_files, train_pids, val_pids, labels, fold_idx):
    seed = SEED * 1000 + fold_idx * 10 + sum(ord(c) for c in encoder) % 100
    set_seed(seed)

    tr_labels = [labels[p] for p in train_pids]
    inner_tr, inner_va = train_test_split(
        np.arange(len(train_pids)), test_size=INNER_VAL_FRAC, stratify=tr_labels, random_state=seed
    )
    inner_tr_pids = [train_pids[i] for i in inner_tr]
    inner_va_pids = [train_pids[i] for i in inner_va]
    inner_tr_labels = [labels[p] for p in inner_tr_pids]
    inner_va_labels = [labels[p] for p in inner_va_pids]

    train_ds = FeatureBagDataset(inner_tr_pids, inner_tr_labels, feat_files, dim, seed=seed)
    early_ds = FeatureBagDataset(inner_va_pids, inner_va_labels, feat_files, dim, seed=seed + 1)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=False,
    )
    early_loader = DataLoader(early_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=False)

    model = Classifier(dim, n_classes=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    counts = np.bincount(np.array(inner_tr_labels), minlength=3).astype(np.float32)
    w = torch.tensor(counts.max() / (counts + 1e-6), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)

    best_acc = -1.0
    best_state = None
    patience = 0

    for ep in range(EPOCHS):
        model.train()
        tloss = 0.0
        for x, mask, lbls, _ in train_loader:
            if x is None:
                continue
            x, mask, lbls = x.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE).squeeze(1)
            logits, _ = model(x, mask)
            loss = crit(logits, lbls)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tloss += loss.item()
        sched.step()

        model.eval()
        ep_preds, ep_trues = [], []
        with torch.no_grad():
            for x, mask, lbls, _ in early_loader:
                logits, _ = model(x.to(DEVICE), mask.to(DEVICE))
                ep_preds.extend(logits.argmax(1).cpu().numpy())
                ep_trues.extend(lbls.squeeze(1).cpu().numpy())
        acc = accuracy_score(ep_trues, ep_preds)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            log.info(f"  {encoder} fold{fold_idx} early stop at epoch {ep+1}")
            break

    model.load_state_dict(best_state)
    model.eval()

    val_ds = FeatureBagDataset(val_pids, [labels[p] for p in val_pids], feat_files, dim, seed=seed + 2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=False)

    all_preds, all_probs, all_trues, all_pids = [], [], [], []
    with torch.no_grad():
        for x, mask, lbls, pids in val_loader:
            logits, _ = model(x.to(DEVICE), mask.to(DEVICE))
            probs = F.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_trues.extend(lbls.squeeze(1).cpu().numpy())
            all_pids.extend(pids)

    preds = np.array(all_preds)
    trues = np.array(all_trues)
    probs = np.array(all_probs)

    torch.save(best_state, os.path.join(OUT_DIR, f"benchmark_v3_{encoder}_fold{fold_idx}.pt"))

    return {
        "fold": fold_idx + 1,
        "seed": seed,
        "train_n": len(inner_tr_pids),
        "early_n": len(inner_va_pids),
        "val_n": len(val_pids),
        "acc": float(accuracy_score(trues, preds)),
        "f1w": float(f1_score(trues, preds, average="weighted", zero_division=0)),
        "f1m": float(f1_score(trues, preds, average="macro", zero_division=0)),
        "cm": confusion_matrix(trues, preds, labels=[0, 1, 2]).tolist(),
        "preds": preds.tolist(),
        "probs": probs.tolist(),
        "trues": trues.tolist(),
        "pids": all_pids,
    }


def main():
    with open(COHORT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)
    canonical = cohort["canonical_patient_ids"]
    labels = {str(k): int(v) for k, v in cohort["labels"].items()}
    folds = cohort["folds"]
    log.info(f"Canonical cohort: {len(canonical)} patients, {len(folds)} folds")

    all_results = {}

    for enc_idx, (encoder, (feat_dir, dim)) in enumerate(ENCODERS.items()):
        log.info(f"\n{'='*60}\nEncoder: {encoder} (dim={dim})\n{'='*60}")
        feat_files = {os.path.basename(f).replace(".h5", ""): f for f in glob.glob(os.path.join(feat_dir, "*.h5"))}
        missing = [p for p in canonical if p not in feat_files]
        if missing:
            log.error(f"{encoder}: missing features for {len(missing)} canonical patients: {missing[:10]}")
            continue
        summary_path = os.path.join(OUT_DIR, f"diagnosis_{encoder}_v3.json")
        ckpts = [os.path.join(OUT_DIR, f"benchmark_v3_{encoder}_fold{i}.pt") for i in range(1, 6)]
        if all(os.path.exists(c) for c in ckpts) and os.path.exists(summary_path):
            log.info(f"{encoder}: all 5 folds already done, skipping")
            with open(summary_path, "r", encoding="utf-8") as f:
                all_results[encoder] = json.load(f)
            continue

        fold_results = []
        oof_rows = []
        for fold_idx in range(1, N_FOLDS + 1):
            fold_key = f"fold_{fold_idx}"
            fd = folds[fold_key]
            train_pids = [p for p in fd["train_pids"] if p in feat_files]
            val_pids = [p for p in fd["val_pids"] if p in feat_files]
            log.info(f"\n{encoder} {fold_key}: train={len(train_pids)}, val={len(val_pids)}")
            t0 = time.time()
            res = train_one_fold(encoder, dim, feat_files, train_pids, val_pids, labels, fold_idx)
            res["seconds"] = round(time.time() - t0, 1)
            log.info(
                f"  -> acc={res['acc']:.4f} f1w={res['f1w']:.4f} f1m={res['f1m']:.4f} "
                f"({res['seconds']}s)"
            )
            fold_results.append({k: v for k, v in res.items() if k not in ("preds", "probs", "trues", "pids")})
            for pid, t, p, pr in zip(res["pids"], res["trues"], res["preds"], res["probs"]):
                oof_rows.append({"patient_id": pid, "fold": fold_idx, "true": int(t), "pred": int(p),
                                 "p0": float(pr[0]), "p1": float(pr[1]), "p2": float(pr[2])})

        pred_df = pd.DataFrame(oof_rows)
        pred_df.to_csv(os.path.join(OUT_DIR, f"diagnosis_oof_{encoder}.csv"), index=False)

        accs = [r["acc"] for r in fold_results]
        f1ws = [r["f1w"] for r in fold_results]
        f1ms = [r["f1m"] for r in fold_results]
        per_class = {}
        for cls, name in [(0, "ERMS"), (1, "ARMS"), (2, "Spindle")]:
            vals = []
            for r in fold_results:
                cm = np.array(r["cm"])
                tp = cm[cls, cls]
                fp = cm[:, cls].sum() - tp
                fn = cm[cls, :].sum() - tp
                vals.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
            per_class[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        summary = {
            "encoder": encoder,
            "dim": dim,
            "n_patients": len(canonical),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1w_mean": float(np.mean(f1ws)),
            "f1w_std": float(np.std(f1ws)),
            "f1m_mean": float(np.mean(f1ms)),
            "f1m_std": float(np.std(f1ms)),
            "per_class": per_class,
            "folds": fold_results,
        }
        all_results[encoder] = summary
        with open(os.path.join(OUT_DIR, f"diagnosis_{encoder}_v3.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log.info(f"{encoder} done: acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}")

    with open(os.path.join(OUT_DIR, "diagnosis_results_v3.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info("\n=== DIAGNOSIS V3 DONE ===")


if __name__ == "__main__":
    main()
