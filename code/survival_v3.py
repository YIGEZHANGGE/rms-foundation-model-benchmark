"""v3 Survival: true OOF features from v3 diagnosis checkpoints + clinical Cox."""

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
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42
MAX_PATCHES = 4096
N_PCA = 10
PROJECT = r"C:\Users\onekey\WorkBuddy\20260330120149"
OUT_DIR = os.path.join(PROJECT, "rms_output_v3")
COHORT_PATH = os.path.join(OUT_DIR, "cohort_and_folds.json")
FEAT_DIR = r"F:\Pathology\features\features_cog_conch15"
DIM = 768

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MAMMOTH(nn.Module):
    def __init__(self, dim, slots=8, heads=4, experts=4, hidden=384):
        super().__init__()
        self.slot_proj = nn.Linear(dim, slots * heads)
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15), nn.Linear(hidden, hidden)) for _ in range(experts)])
        self.proj = nn.Linear(hidden, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        B, N, D = x.shape
        l = self.slot_proj(x).view(B, N, 8, 4)
        if mask is not None:
            l = l.masked_fill(~mask.unsqueeze(-1).unsqueeze(-1), -1e9)
        w = F.softmax(l, dim=1)
        s = torch.einsum("bnsh,bnd->bshd", w, x).mean(2)
        r = F.softmax(self.router(s), dim=-1)
        return self.norm(self.proj(sum(e(s) * r[:, :, i:i+1] for i, e in enumerate(self.experts)).mean(1)))


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

    def forward(self, x, mask=None):
        x = self.proj_in(x)
        x = x + self.mammoth(x, mask).unsqueeze(1)
        feat, attn = self.abmil(x, mask)
        return self.head(feat), feat, attn


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


def extract_oof_features(folds, feat_files):
    oof = {}
    for fi in range(1, 6):
        fk = f"fold_{fi}"
        ck = os.path.join(OUT_DIR, f"benchmark_v3_conch15_fold{fi}.pt")
        if not os.path.exists(ck):
            print(f"Missing checkpoint {ck}")
            continue
        val_pids = folds[fk]["val_pids"]

        class DS(Dataset):
            def __init__(self):
                self.rng = np.random.RandomState(SEED * 10 + fi)
            def __len__(self):
                return len(val_pids)
            def __getitem__(self, i):
                pid = val_pids[i]
                with h5py.File(feat_files[pid], "r") as hf:
                    feats = hf["features"][:].astype(np.float32)
                if feats.shape[0] > MAX_PATCHES:
                    idxs = self.rng.choice(feats.shape[0], MAX_PATCHES, replace=False)
                    feats = feats[idxs]
                return torch.FloatTensor(feats), torch.LongTensor([0]), pid

        loader = DataLoader(DS(), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=False)
        model = Classifier(DIM).to(DEVICE)
        model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True))
        model.eval()
        with torch.no_grad():
            for x, mask, _, pids in loader:
                _, feat, _ = model(x.to(DEVICE), mask.to(DEVICE))
                for pid, fvec in zip(pids, feat.cpu().numpy()):
                    oof[pid] = fvec.astype(np.float32)
        del model; torch.cuda.empty_cache()
        with open(os.path.join(OUT_DIR, f"oof_features_v3_fold_{fi}.json"), "w") as f:
            json.dump({p: oof[p].tolist() for p in val_pids}, f)
        print(f"Extracted fold {fi}: {len(val_pids)}")
    return oof


def time_dependent_auc(times, events, risk, horizons=(1, 3, 5)):
    out = {}
    risk = np.asarray(risk)
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    for t in horizons:
        case_idx = np.where((events == 1) & (times <= t))[0]
        ctrl_idx = np.where(times > t)[0]
        if len(case_idx) < 2 or len(ctrl_idx) < 2:
            out[t] = float("nan")
            continue
        vals = []
        for i in case_idx:
            for j in ctrl_idx:
                vals.append(1.0 if risk[i] > risk[j] else (0.5 if risk[i] == risk[j] else 0.0))
        out[t] = float(np.mean(vals))
    return out


def run_cox_cv(df, feat_lookup, folds, tcol, ecol, use_pca=False, clin_cols=None):
    all_risk, all_time, all_event, all_pid = [], [], [], []
    train_cindices = []
    for fi in range(1, 6):
        fk = f"fold_{fi}"
        val_set = set(folds[fk]["val_pids"])
        train_df = df[~df["pid"].isin(val_set)].copy()
        val_df = df[df["pid"].isin(val_set)].copy()
        train_df = train_df.dropna(subset=[tcol, ecol] + (clin_cols or []))
        val_df = val_df.dropna(subset=[tcol, ecol] + (clin_cols or []))
        if len(train_df) < 20 or len(val_df) < 5 or train_df[ecol].nunique() < 2:
            continue
        train_feat = np.stack([feat_lookup[p] for p in train_df["pid"]])
        val_feat = np.stack([feat_lookup[p] for p in val_df["pid"]])
        if use_pca:
            scaler = StandardScaler().fit(train_feat)
            pca = PCA(n_components=N_PCA, random_state=SEED).fit(scaler.transform(train_feat))
            tr_X = pca.transform(scaler.transform(train_feat))
            va_X = pca.transform(scaler.transform(val_feat))
            cols = [f"PC{i+1}" for i in range(N_PCA)]
        else:
            tr_X = train_feat
            va_X = val_feat
            cols = [f"F{i}" for i in range(train_feat.shape[1])]
        tr_df = pd.DataFrame(tr_X, columns=cols, index=train_df.index)
        va_df = pd.DataFrame(va_X, columns=cols, index=val_df.index)
        if clin_cols:
            for c in clin_cols:
                tr_df[c] = train_df[c].values
                va_df[c] = val_df[c].values
        tr_ready = pd.concat([train_df[[tcol, ecol]].rename(columns={tcol: "duration", ecol: "event"}), tr_df], axis=1)
        va_ready = pd.concat([val_df[[tcol, ecol]].rename(columns={tcol: "duration", ecol: "event"}), va_df], axis=1)
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(tr_ready, duration_col="duration", event_col="event")
            train_cindices.append(cph.concordance_index_)
            risk = cph.predict_partial_hazard(va_ready.drop(columns=["duration", "event"]))
            all_risk.extend(risk.values)
            all_time.extend(va_ready["duration"].values)
            all_event.extend(va_ready["event"].values)
            all_pid.extend(val_df["pid"].values)
        except Exception:
            continue
    if len(all_risk) < 20:
        return None
    c = concordance_index(all_time, -np.asarray(all_risk), all_event)
    rng = np.random.RandomState(SEED)
    boot = []
    idx = np.arange(len(all_risk))
    for _ in range(1000):
        bi = rng.choice(idx, len(idx), replace=True)
        boot.append(concordance_index(np.asarray(all_time)[bi], -np.asarray(all_risk)[bi], np.asarray(all_event)[bi]))
    return {
        "cv_cindex": float(c),
        "cindex_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n": int(len(all_risk)),
        "events": int(sum(all_event)),
        "train_cindex_mean": float(np.mean(train_cindices)) if train_cindices else None,
        "time_auc": time_dependent_auc(all_time, all_event, all_risk),
    }


def main():
    with open(COHORT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)
    folds = cohort["folds"]
    surv = cohort["survival"]
    canonical = cohort["canonical_patient_ids"]

    feat_files = {os.path.basename(f).replace(".h5", ""): f for f in glob.glob(os.path.join(FEAT_DIR, "*.h5"))}
    oof = extract_oof_features(folds, feat_files)

    rows = []
    for p in canonical:
        if p not in oof:
            continue
        s = surv[p]
        rows.append({
            "pid": p,
            "efs_time": s["efs_years"], "efs_event": s["efs_event"],
            "os_time": s["os_years"], "os_event": s["os_event"],
            "age": s["age"], "stage": s["stage"],
            "risk_high": 1 if s["risk_group"] == "High" else 0,
        })
    df = pd.DataFrame(rows)
    feat_lookup = {p: oof[p] for p in df["pid"]}

    results = {}
    for outcome, tcol, ecol in [("efs", "efs_time", "efs_event"), ("os", "os_time", "os_event")]:
        sub = df.dropna(subset=[tcol, ecol]).copy()
        sub = sub[sub[tcol] > 0].copy()
        m1 = run_cox_cv(sub, feat_lookup, folds, tcol, ecol, use_pca=True)
        clin_cols = ["age", "stage", "risk_high"]
        sub_c = sub.dropna(subset=clin_cols).copy()
        m2 = run_cox_cv(sub_c, feat_lookup, folds, tcol, ecol, use_pca=False, clin_cols=clin_cols)
        m3 = run_cox_cv(sub_c, feat_lookup, folds, tcol, ecol, use_pca=True, clin_cols=clin_cols)
        results[outcome] = {"M1_morphology": m1, "M2_clinical": m2, "M3_combined": m3}
        print(outcome, json.dumps(results[outcome], indent=2, ensure_ascii=False))

    with open(os.path.join(OUT_DIR, "survival_results_v3.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("=== SURVIVAL V3 DONE ===")


if __name__ == "__main__":
    main()
