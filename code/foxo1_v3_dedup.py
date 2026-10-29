"""v3 FOXO1 prediction: BCH single-center 5-fold CV + decision curve."""

import os, sys, json, glob, random, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

SEED = 42
EPOCHS = 60
PATIENCE = 15
LR = 5e-4
MAX_PATCHES = 4096
PROJECT = r"C:\Users\onekey\WorkBuddy\20260330120149"
OUT_DIR = os.path.join(PROJECT, "rms_output_v3")
CHN_XLSX = r"F:\Pathology\Pathology_patch_Normalize_quality\RMS_BCH_Pathology_updated.xlsx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODERS = {
    "conch": (r"F:\Pathology\features_conch_768d", 768, "_conch"),
    "conch15": (r"F:\Pathology\features\features_chn_conch15", 768, "_conch15"),
    "phikon": (r"F:\Pathology\features\features_chn_phikon", 1024, "_phikon"),
    "hoptimus0": (r"F:\Pathology\features\features_chn_hoptimus0", 1536, "_hoptimus0"),
}


class ABMIL(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x, mask=None):
        a = self.attn(x).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, -1e9)
        a = F.softmax(a, dim=1)
        feat = self.proj((a.unsqueeze(-1) * x).sum(1))
        return self.head(feat), a


def collate(batch):
    fs, ls, ps = zip(*batch)
    mn = max(f.shape[0] for f in fs)
    d = fs[0].shape[1]
    p = torch.zeros(len(fs), mn, d)
    m = torch.zeros(len(fs), mn, dtype=torch.bool)
    for i, f in enumerate(fs):
        p[i, :f.shape[0]] = f
        m[i, :f.shape[0]] = True
    return p, m, torch.stack(ls), list(ps)


def load_chn():
    chn = pd.read_excel(CHN_XLSX)
    fusion_col = [c for c in chn.columns if "融合" in str(c)][0]
    chn["pid_str"] = chn["编号"].astype(int).astype(str).str.zfill(7)
    chn["foxo1"] = chn[fusion_col].map({1.0: 1, 2.0: 0}).fillna(-1).astype(int)
    out = chn[chn["foxo1"] >= 0].copy()
    out = out.drop_duplicates("pid_str", keep="first")   # 1002311 在 Excel 中重复且标签一致
    return out


def train_fold(tr_pids, tr_lbl, va_pids, va_lbl, feat_files, dim, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    inner_tr_idx, inner_va_idx = train_test_split(np.arange(len(tr_pids)), test_size=0.2, stratify=tr_lbl, random_state=seed)
    inner_tr_pids = [tr_pids[i] for i in inner_tr_idx]
    inner_va_pids = [tr_pids[i] for i in inner_va_idx]
    inner_tr_lbl = [tr_lbl[i] for i in inner_tr_idx]
    inner_va_lbl = [tr_lbl[i] for i in inner_va_idx]

    class DS(Dataset):
        def __init__(self, p, l, s):
            self.p = p; self.l = l; self.rng = np.random.RandomState(s)
        def __len__(self):
            return len(self.p)
        def __getitem__(self, i):
            with h5py.File(feat_files[self.p[i]], "r") as hf:
                feats = hf["features"][:].astype(np.float32)
            if feats.shape[0] > MAX_PATCHES:
                idxs = self.rng.choice(feats.shape[0], MAX_PATCHES, replace=False)
                feats = feats[idxs]
            return torch.FloatTensor(feats), torch.LongTensor([self.l[i]]), self.p[i]

    train_loader = DataLoader(DS(inner_tr_pids, inner_tr_lbl, seed), batch_size=1, shuffle=True, collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(DS(inner_va_pids, inner_va_lbl, seed + 1), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    model = ABMIL(dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    pos = sum(inner_tr_lbl); neg = len(inner_tr_lbl) - pos
    w = torch.tensor([1.0, neg / max(pos, 1)], dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)

    best_auc = -1.0; best_state = None; patience = 0
    for ep in range(EPOCHS):
        model.train()
        for x, mask, lbls, _ in train_loader:
            x, mask, lbls = x.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE).squeeze(1)
            logits, _ = model(x, mask)
            loss = crit(logits, lbls)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for x, mask, lbls, _ in val_loader:
                logits, _ = model(x.to(DEVICE), mask.to(DEVICE))
                probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                trues.extend(lbls.squeeze(1).cpu().numpy())
        if len(np.unique(trues)) > 1:
            auc = roc_auc_score(trues, probs)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
        if patience >= PATIENCE:
            break

    model.load_state_dict(best_state); model.eval()
    test_loader = DataLoader(DS(va_pids, va_lbl, seed + 2), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    probs, trues, ids = [], [], []
    with torch.no_grad():
        for x, mask, lbls, pids_b in test_loader:
            logits, _ = model(x.to(DEVICE), mask.to(DEVICE))
            probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            trues.extend(lbls.squeeze(1).cpu().numpy())
            ids.extend(pids_b)
    return np.array(probs), np.array(trues), ids


def decision_curve(trues, probs, thresholds):
    n = len(trues)
    rows = []
    for pt in thresholds:
        preds = (probs >= pt).astype(int)
        tp = int(((preds == 1) & (trues == 1)).sum())
        fp = int(((preds == 1) & (trues == 0)).sum())
        nb = tp / n - (1 - pt) / pt * fp / n
        rows.append({"threshold": pt, "net_benefit": nb})
    return rows


def main():
    chn = load_chn()
    all_results = {}
    for enc, (feat_dir, dim, suffix) in ENCODERS.items():
        feat_files = {}
        for f in glob.glob(os.path.join(feat_dir, "*.h5")):
            pid = os.path.basename(f).replace(".h5", "").replace(suffix, "")
            if pid.isdigit() and len(pid) == 7:
                feat_files[pid] = f
        pids, labels = [], []
        for _, row in chn.iterrows():
            if row["pid_str"] in feat_files:
                pids.append(row["pid_str"]); labels.append(int(row["foxo1"]))

        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        fold_probs = np.zeros(len(pids)); fold_trues = np.zeros(len(pids), dtype=int)
        fold_meta = []
        for fi, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            tr_pids = [pids[i] for i in tr]; tr_lbl = [labels[i] for i in tr]
            seed = SEED * 10 + fi
            va_pids = [pids[i] for i in va]; va_lbl = [labels[i] for i in va]
            probs, trues, ids = train_fold(tr_pids, tr_lbl, va_pids, va_lbl, feat_files, dim, seed)
            for pid, pr, t in zip(ids, probs, trues):
                idx = pids.index(pid)
                fold_probs[idx] = pr; fold_trues[idx] = t
            auc = roc_auc_score(trues, probs) if len(np.unique(trues)) > 1 else float("nan")
            preds = (probs >= 0.5).astype(int)
            fold_meta.append({
                "fold": fi + 1, "auc": float(auc), "acc": float(accuracy_score(trues, preds)),
                "sens": float(recall_score(trues, preds, zero_division=0)),
                "spec": float(recall_score(1 - trues, 1 - preds, zero_division=0)),
            })

        auc = roc_auc_score(fold_trues, fold_probs)
        preds = (fold_probs >= 0.5).astype(int)
        res = {
            "encoder": enc, "dim": dim, "n": len(pids), "n_pos": int(fold_trues.sum()),
            "auc": float(auc), "acc": float(accuracy_score(fold_trues, preds)),
            "sens": float(recall_score(fold_trues, preds, zero_division=0)),
            "spec": float(recall_score(1 - fold_trues, 1 - preds, zero_division=0)),
            "f1": float(f1_score(fold_trues, preds, zero_division=0)),
            "folds": fold_meta,
            "decision_curve": decision_curve(fold_trues, fold_probs, np.arange(0.05, 0.96, 0.05).round(2).tolist()),
        }
        all_results[enc] = res
        pd.DataFrame({"patient_id": pids, "true": fold_trues, "prob": fold_probs}).to_csv(
            os.path.join(OUT_DIR, f"foxo1_probs_v3_dedup_{enc}.csv"), index=False)
        print(json.dumps(res, indent=2, ensure_ascii=False))

    with open(os.path.join(OUT_DIR, "foxo1_results_v3_dedup.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("=== FOXO1 V3 DONE ===")


if __name__ == "__main__":
    main()
