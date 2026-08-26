"""v3 Architecture ablation on frozen n=390 cohort (honest custom variants)."""

import os, sys, json, glob, time, logging, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
EPOCHS = 40
PATIENCE = 12
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_PATCHES = 4096
PROJECT = r"C:\Users\onekey\WorkBuddy\20260330120149"
OUT_DIR = os.path.join(PROJECT, "rms_output_v3")
COHORT_PATH = os.path.join(OUT_DIR, "cohort_and_folds.json")
LOG_PATH = os.path.join(OUT_DIR, "architecture_ablation_v3.log")

os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8", mode="a"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("v3_ablation")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODERS = {
    "conch": (r"F:\Pathology\features_rms_conch", 768),
    "conch15": (r"F:\Pathology\features\features_cog_conch15", 768),
    "hoptimus0": (r"F:\Pathology\features\features_cog_hoptimus0", 1536),
}
ARCHS = ["AttentionMIL", "MultiClusterAttention", "TransformerInstanceAttention", "TransformerPooling"]


class AttentionMIL(nn.Module):
    def __init__(self, dim, n_cls, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, n_cls)
    def forward(self, x, mask=None):
        a = self.attn(x).squeeze(-1)
        if mask is not None: a = a.masked_fill(~mask, -1e9)
        a = F.softmax(a, dim=1)
        feat = self.proj((a.unsqueeze(-1) * x).sum(1))
        return self.head(feat), a


class MultiClusterAttention(nn.Module):
    def __init__(self, dim, n_cls, hidden=256, n_clusters=8):
        super().__init__()
        self.n_clusters = n_clusters
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(hidden, n_clusters))
        self.head = nn.Linear(dim * n_clusters, n_cls)
    def forward(self, x, mask=None):
        B, N, D = x.shape
        l = self.attn(x)
        if mask is not None: l = l.masked_fill(~mask.unsqueeze(-1), -1e9)
        a = F.softmax(l, dim=1)
        feat = torch.einsum("bnc,bnd->bcd", a, x).view(B, -1)
        return self.head(feat), a.mean(-1)


class TransformerInstanceAttention(nn.Module):
    def __init__(self, dim, n_cls, hidden=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True, dim_feedforward=dim*4), num_layers=2)
        self.head = nn.Linear(dim, n_cls)
        self.critic = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x, mask=None):
        B, N, D = x.shape
        scores = self.critic(x).squeeze(-1)
        if mask is not None: scores = scores.masked_fill(~mask, -1e9)
        max_idx = scores.argmax(1)
        max_feat = x[torch.arange(B, device=x.device), max_idx].unsqueeze(1)
        bag = torch.cat([self.cls_token.expand(B, -1, -1), self.proj(x)], dim=1)
        out = self.transformer(bag)[:, 0, :]
        return self.head(out + max_feat.squeeze(1)), F.softmax(scores, dim=1)


class TransformerPooling(nn.Module):
    def __init__(self, dim, n_cls, reduced=512):
        super().__init__()
        self.reduce = nn.Linear(dim, reduced)
        self.pos = nn.Parameter(torch.randn(1, 5000, reduced) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, reduced) * 0.02)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=reduced, nhead=8, batch_first=True, dim_feedforward=reduced*4), num_layers=2)
        self.head = nn.Sequential(nn.Linear(reduced, 256), nn.GELU(), nn.Linear(256, n_cls))
    def forward(self, x, mask=None):
        B, N, D = x.shape
        x = self.reduce(x)
        if N > 5000:
            keep = torch.randperm(N, device=x.device)[:5000]
            x = x[:, keep, :]
            N = 5000
        x = x + self.pos[:, :N, :]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        out = self.transformer(x)[:, 0, :]
        return self.head(out), torch.ones(B, N, device=x.device)


MODELS = {
    "AttentionMIL": AttentionMIL,
    "MultiClusterAttention": MultiClusterAttention,
    "TransformerInstanceAttention": TransformerInstanceAttention,
    "TransformerPooling": TransformerPooling,
}


class FeatureBagDataset(Dataset):
    def __init__(self, pids, labels, feat_files, dim, max_p=MAX_PATCHES, seed=SEED):
        self.pids = pids; self.labels = labels; self.feat_files = feat_files
        self.dim = dim; self.max_p = max_p; self.rng = np.random.RandomState(seed)
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, i):
        pid = self.pids[i]
        with h5py.File(self.feat_files[pid], "r") as hf:
            feats = hf["features"][:].astype(np.float32)
        if feats.shape[0] > self.max_p:
            idxs = self.rng.choice(feats.shape[0], self.max_p, replace=False)
            feats = feats[idxs]
        return torch.FloatTensor(feats), torch.LongTensor([self.labels[i]]), pid


def collate(batch):
    fs, ls, ps = zip(*batch)
    mn = max(f.shape[0] for f in fs); d = fs[0].shape[1]
    p = torch.zeros(len(fs), mn, d); m = torch.zeros(len(fs), mn, dtype=torch.bool)
    for i, f in enumerate(fs):
        p[i, :f.shape[0]] = f; m[i, :f.shape[0]] = True
    return p, m, torch.stack(ls), list(ps)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one(arch, dim, feat_files, train_pids, val_pids, labels, fold_idx):
    seed = SEED * 1000 + fold_idx * 10 + sum(ord(c) for c in arch) % 50
    set_seed(seed)
    tr_labels = [labels[p] for p in train_pids]
    inner_tr, inner_va = train_test_split(np.arange(len(train_pids)), test_size=0.2, stratify=tr_labels, random_state=seed)
    tr_pids = [train_pids[i] for i in inner_tr]; va_pids = [train_pids[i] for i in inner_va]
    tr_lbl = [labels[p] for p in tr_pids]; va_lbl = [labels[p] for p in va_pids]
    train_loader = DataLoader(FeatureBagDataset(tr_pids, tr_lbl, feat_files, dim, seed=seed), batch_size=1, shuffle=True,
                              collate_fn=collate, generator=torch.Generator().manual_seed(seed), num_workers=0)
    early_loader = DataLoader(FeatureBagDataset(va_pids, va_lbl, feat_files, dim, seed=seed+1), batch_size=1, shuffle=False,
                              collate_fn=collate, num_workers=0)
    model = MODELS[arch](dim, 3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    counts = np.bincount(np.array(tr_lbl), minlength=3).astype(np.float32)
    w = torch.tensor(counts.max() / (counts + 1e-6), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    best_acc, best_state, patience = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for x, mask, lbls, _ in train_loader:
            x, mask, lbls = x.to(DEVICE), mask.to(DEVICE), lbls.to(DEVICE).squeeze(1)
            logits, _ = model(x, mask); loss = crit(logits, lbls)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, mask, lbls, _ in early_loader:
                logits, _ = model(x.to(DEVICE), mask.to(DEVICE))
                preds.extend(logits.argmax(1).cpu().numpy()); trues.extend(lbls.squeeze(1).cpu().numpy())
        acc = accuracy_score(trues, preds)
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state); model.eval()
    val_loader = DataLoader(FeatureBagDataset(val_pids, [labels[p] for p in val_pids], feat_files, dim, seed=seed+2),
                            batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    preds, probs, trues, ids = [], [], [], []
    with torch.no_grad():
        for x, mask, lbls, pids in val_loader:
            logits, _ = model(x.to(DEVICE), mask.to(DEVICE))
            pr = F.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().numpy()); probs.extend(pr.cpu().numpy())
            trues.extend(lbls.squeeze(1).cpu().numpy()); ids.extend(pids)
    return {"acc": float(accuracy_score(trues, preds)), "f1w": float(f1_score(trues, preds, average="weighted", zero_division=0)),
            "f1m": float(f1_score(trues, preds, average="macro", zero_division=0)),
            "cm": confusion_matrix(trues, preds, labels=[0,1,2]).tolist(),
            "preds": preds, "trues": trues, "probs": probs, "pids": ids}


def main():
    with open(COHORT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)
    canonical = cohort["canonical_patient_ids"]
    labels = {str(k): int(v) for k, v in cohort["labels"].items()}
    folds = cohort["folds"]
    results = {}
    for enc, (feat_dir, dim) in ENCODERS.items():
        feat_files = {os.path.basename(f).replace(".h5", ""): f for f in glob.glob(os.path.join(feat_dir, "*.h5"))}
        enc_res = {}
        for arch in ARCHS:
            fold_res = []
            for fi in range(1, 6):
                fd = folds[f"fold_{fi}"]
                train_pids = [p for p in fd["train_pids"] if p in feat_files]
                val_pids = [p for p in fd["val_pids"] if p in feat_files]
                log.info(f"{enc} {arch} fold{fi}")
                r = train_one(arch, dim, feat_files, train_pids, val_pids, labels, fi)
                fold_res.append({k: v for k, v in r.items() if k not in ("preds", "trues", "probs", "pids")})
            accs = [r["acc"] for r in fold_res]; f1ms = [r["f1m"] for r in fold_res]
            enc_res[arch] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                             "f1m_mean": float(np.mean(f1ms)), "f1m_std": float(np.std(f1ms)), "folds": fold_res}
            log.info(f"{enc} {arch}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}")
        results[enc] = enc_res
    with open(os.path.join(OUT_DIR, "architecture_ablation_v3.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("=== ARCHITECTURE ABLATION V3 DONE ===")


if __name__ == "__main__":
    main()
