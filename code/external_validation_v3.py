"""v3 External validation: 4 encoders on BCH, seeded repeats, bootstrap CI."""

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
N_REPEATS = 5
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
        return self.head(feat), attn


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
    hist_col = [c for c in chn.columns if "病理类型" in str(c) and "组织" not in str(c)][0]
    spec_col = [c for c in chn.columns if "穿刺" in str(c) or "活检" in str(c)][0]
    chn = chn[chn[hist_col].notna() & (chn[hist_col] != "未确")]
    chn["hist_label"] = chn[hist_col].astype(float).astype(int) - 1
    chn = chn[chn["hist_label"].isin([0, 1, 2])]
    chn["pid_str"] = chn["编号"].astype(int).astype(str).str.zfill(7)
    chn["spec_type"] = chn[spec_col].map({1.0: 0, 2.0: 1}).fillna(-1).astype(int)
    return chn


def run_encoder(enc_name, feat_dir, dim, suffix, chn):
    feat_files = {}
    for f in glob.glob(os.path.join(feat_dir, "*.h5")):
        pid = os.path.basename(f).replace(".h5", "").replace(suffix, "")
        if pid.isdigit() and len(pid) == 7:
            feat_files[pid] = f

    pids, labels, specs = [], [], []
    for _, row in chn.iterrows():
        pid = row["pid_str"]
        if pid in feat_files:
            pids.append(pid)
            labels.append(int(row["hist_label"]))
            specs.append(int(row["spec_type"]))

    ckpts = [os.path.join(OUT_DIR, f"benchmark_v3_{enc_name}_fold{i}.pt") for i in range(1, 6)]
    if not all(os.path.exists(c) for c in ckpts):
        print(f"{enc_name}: checkpoints missing, skip")
        return None

    repeat_preds = []
    repeat_probs = []
    for rep in range(N_REPEATS):
        seed = SEED + rep * 100

        class DS(Dataset):
            def __init__(self):
                self.rng = np.random.RandomState(seed)
            def __len__(self):
                return len(pids)
            def __getitem__(self, i):
                with h5py.File(feat_files[pids[i]], "r") as hf:
                    feats = hf["features"][:].astype(np.float32)
                if feats.shape[0] > MAX_PATCHES:
                    idxs = self.rng.choice(feats.shape[0], MAX_PATCHES, replace=False)
                    feats = feats[idxs]
                return torch.FloatTensor(feats), torch.LongTensor([labels[i]]), pids[i]

        loader = DataLoader(DS(), batch_size=1, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=False)
        all_probs = np.zeros((len(pids), 3), dtype=np.float32)
        for ck in ckpts:
            model = Classifier(dim).to(DEVICE)
            model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=True))
            model.eval()
            idx = 0
            with torch.no_grad():
                for x, mask, _, _ in loader:
                    logits, _ = model(x.to(DEVICE), mask.to(DEVICE))
                    all_probs[idx:idx+1] += F.softmax(logits, dim=1).cpu().numpy()
                    idx += 1
            del model
            torch.cuda.empty_cache()
        all_probs /= len(ckpts)
        preds = all_probs.argmax(1)
        repeat_preds.append(preds)
        repeat_probs.append(all_probs)

    mean_probs = np.mean(repeat_probs, axis=0)
    preds = mean_probs.argmax(1)
    trues = np.array(labels)
    acc = accuracy_score(trues, preds)
    f1m = f1_score(trues, preds, average="macro", zero_division=0)

    rng = np.random.RandomState(SEED)
    boot_acc, boot_f1m = [], []
    for _ in range(1000):
        idx = rng.choice(len(pids), len(pids), replace=True)
        boot_acc.append(accuracy_score(trues[idx], preds[idx]))
        boot_f1m.append(f1_score(trues[idx], preds[idx], average="macro", zero_division=0))

    cm = confusion_matrix(trues, preds, labels=[0, 1, 2])
    per_class = {}
    for cls, name in [(0, "ERMS"), (1, "ARMS"), (2, "Spindle")]:
        tp = cm[cls, cls]; fp = cm[:, cls].sum() - tp; fn = cm[cls, :].sum() - tp
        per_class[name] = {"n": int((trues == cls).sum()), "f1": float(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0)}

    result = {
        "encoder": enc_name,
        "n_patients": len(pids),
        "acc": float(acc),
        "acc_ci": [float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5))],
        "f1m": float(f1m),
        "f1m_ci": [float(np.percentile(boot_f1m, 2.5)), float(np.percentile(boot_f1m, 97.5))],
        "per_class": per_class,
        "cm": cm.tolist(),
        "repeat_acc": [float(accuracy_score(trues, p)) for p in repeat_preds],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_rows = []
    for i, pid in enumerate(pids):
        out_rows.append({"patient_id": pid, "true": int(trues[i]), "pred": int(preds[i]),
                         "prob_ERMS": float(mean_probs[i, 0]), "prob_ARMS": float(mean_probs[i, 1]),
                         "prob_Spindle": float(mean_probs[i, 2]), "specimen": int(specs[i])})
    pd.DataFrame(out_rows).to_csv(os.path.join(OUT_DIR, f"bch_predictions_v3_{enc_name}.csv"), index=False)
    return result


def main():
    chn = load_chn()
    results = {}
    for enc, (fd, dim, suffix) in ENCODERS.items():
        r = run_encoder(enc, fd, dim, suffix, chn)
        if r:
            results[enc] = r
    with open(os.path.join(OUT_DIR, "external_validation_v3.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("=== EXTERNAL V3 DONE ===")


if __name__ == "__main__":
    main()
