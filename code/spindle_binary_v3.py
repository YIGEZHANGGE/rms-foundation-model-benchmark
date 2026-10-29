# -*- coding: utf-8 -*-
"""Experiment (a): is the ERMS vs spindle/sclerosing boundary learnable inside COG?

Two-class task (ERMS = 0, spindle/sclerosing = 1), ARMS excluded, five stratified folds,
same MAMMOTH classifier and training schedule as the three-class benchmark.
Two conditions: plain training, and spindle oversampled x5 in the training fold.
"""
import os, sys, json, glob, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_diagnosis_v3 import Classifier, FeatureBagDataset, collate, COHORT_PATH, OUT_DIR, EPOCHS, PATIENCE, LR
DIM = 768

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIR = r"F:////Pathology////features_rms_conch"
SUFFIX = ""


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_fold(tr_pids, tr_lbl, va_pids, va_lbl, feat_files, fold, oversample=1):
    set_seed(SEED * 10 + fold)
    if oversample > 1:                      # repeat spindle cases in the training list
        idx = [i for i, l in enumerate(tr_lbl) if l == 1]
        for _ in range(oversample - 1):
            tr_pids = tr_pids + [tr_pids[i] for i in idx]
            tr_lbl = tr_lbl + [1] * len(idx)

    itr, iva = train_test_split(np.arange(len(tr_pids)), test_size=0.2, stratify=tr_lbl, random_state=SEED * 10 + fold)
    p_tr = [tr_pids[i] for i in itr]; l_tr = [tr_lbl[i] for i in itr]
    p_va = [tr_pids[i] for i in iva]; l_va = [tr_lbl[i] for i in iva]

    dl_tr = DataLoader(FeatureBagDataset(p_tr, l_tr, feat_files, DIM, seed=SEED * 10 + fold),
                       batch_size=1, shuffle=True, collate_fn=collate,
                       generator=torch.Generator().manual_seed(SEED * 10 + fold), num_workers=0)
    dl_ev = DataLoader(FeatureBagDataset(p_va, l_va, feat_files, DIM, seed=SEED * 10 + fold + 1),
                       batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    model = Classifier(DIM, 2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    counts = np.bincount(np.array(l_tr), minlength=2).astype(np.float32)
    w = torch.tensor(counts.max() / (counts + 1e-6), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    best, best_state, patience = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for x, mask, lb, _ in dl_tr:
            x, mask, lb = x.to(DEVICE), mask.to(DEVICE), lb.to(DEVICE).squeeze(1)
            logits, _ = model(x, mask)
            loss = crit(logits, lb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
        model.eval(); pr, tr_ = [], []
        with torch.no_grad():
            for x, mask, lb, _ in dl_ev:
                lo, _ = model(x.to(DEVICE), mask.to(DEVICE))
                pr.extend(lo.argmax(1).cpu().numpy()); tr_.extend(lb.squeeze(1).numpy())
        acc = accuracy_score(tr_, pr)
        if acc > best: best, best_state, patience = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= PATIENCE: break
    model.load_state_dict(best_state); model.eval()

    dl_va = DataLoader(FeatureBagDataset(va_pids, va_lbl, feat_files, DIM, seed=SEED * 10 + fold + 2),
                       batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)
    probs, preds, trues = [], [], []
    with torch.no_grad():
        for x, mask, lb, _ in dl_va:
            lo, _ = model(x.to(DEVICE), mask.to(DEVICE))
            probs.extend(F.softmax(lo, 1)[:, 1].cpu().numpy())
            preds.extend(lo.argmax(1).cpu().numpy()); trues.extend(lb.squeeze(1).numpy())
    return {"acc": float(accuracy_score(trues, preds)),
            "f1_macro": float(f1_score(trues, preds, average="macro", zero_division=0)),
            "spindle_recall": float(recall_score(trues, preds, pos_label=1, zero_division=0)),
            "auroc": float(roc_auc_score(trues, probs)) if len(set(trues)) > 1 else float("nan")}, trues, probs


def main():
    cohort = json.load(open(COHORT_PATH, encoding="utf-8"))
    labels = {str(k): int(v) for k, v in cohort["labels"].items()}
    feat_files = {os.path.basename(f).replace(".h5", "").replace(SUFFIX, ""): f
                  for f in glob.glob(os.path.join(FEAT_DIR, "*.h5"))}
    pids = [p for p in labels if p in feat_files and labels[p] in (0, 2)]
    y = [1 if labels[p] == 2 else 0 for p in pids]
    print(f"two-class cohort: n={len(pids)}  ERMS={y.count(0)}  spindle={y.count(1)}", flush=True)
    print(f"majority-class baseline accuracy = {max(y.count(0), y.count(1))/len(y):.3f}", flush=True)

    out = {}
    for oversample in (1, 5):
        key = f"plain" if oversample == 1 else f"oversample_x{oversample}"
        folds, all_t, all_p = [], [], []
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for fi, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            r, tt, pp = train_fold([pids[i] for i in tr], [y[i] for i in tr],
                                   [pids[i] for i in va], [y[i] for i in va], feat_files, fi, oversample)
            folds.append(r); all_t += tt; all_p += pp
            print(f"  {key} fold{fi}: acc={r['acc']:.3f} macroF1={r['f1_macro']:.3f} "
                  f"spindle_recall={r['spindle_recall']:.3f} auroc={r['auroc']:.3f}", flush=True)
        pooled = {"auroc": float(roc_auc_score(all_t, all_p)) if len(set(all_t)) > 1 else float("nan")}
        out[key] = {"folds": folds, "pooled": pooled,
                    "mean": {k: float(np.nanmean([f[k] for f in folds])) for k in folds[0]}}
        print(f"== {key}: acc={out[key]['mean']['acc']:.3f} macroF1={out[key]['mean']['f1_macro']:.3f} "
              f"spindle_recall={out[key]['mean']['spindle_recall']:.3f} "
              f"AUROC(mean fold)={out[key]['mean']['auroc']:.3f} pooled={pooled['auroc']:.3f}", flush=True)

    json.dump(out, open(os.path.join(OUT_DIR, "spindle_binary_v3.json"), "w"), indent=2)
    print("=== SPINDLE BINARY DONE ===")


if __name__ == "__main__":
    main()
