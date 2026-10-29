# -*- coding: utf-8 -*-
"""Experiment (a, external arm): does a two-class ERMS vs spindle model trained on COG transfer to BCH?

COG: ERMS + spindle only (ARMS excluded), five stratified folds.
BCH: ERMS + spindle only (155 patients: 145 + 10).
Five fold models are averaged (same convention as the main pipeline). Threshold fixed at 0.5.
"""
import os, sys, json, glob, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np, h5py, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score,
                             average_precision_score, roc_curve)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_diagnosis_v3 import Classifier, FeatureBagDataset, collate, COHORT_PATH, OUT_DIR, EPOCHS, PATIENCE, LR

SEED = 42; DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COG_DIR = r"F:/Pathology/features_rms_conch"
BCH_DIR = r"F:/Pathology/features_conch_768d"
BCH_PRED = r"C:/Users/onekey/WorkBuddy/20260330120149/rms_output_v3/bch_predictions_v3_conch.csv"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def fit(tr_pids, tr_lbl, feat_files, fold, oversample=1):
    set_seed(SEED * 10 + fold)
    if oversample > 1:
        idx = [i for i, l in enumerate(tr_lbl) if l == 1]
        for _ in range(oversample - 1):
            tr_pids = tr_pids + [tr_pids[i] for i in idx]; tr_lbl = tr_lbl + [1] * len(idx)
    itr, iva = train_test_split(np.arange(len(tr_pids)), test_size=0.2, stratify=tr_lbl, random_state=SEED * 10 + fold)
    p_tr = [tr_pids[i] for i in itr]; l_tr = [tr_lbl[i] for i in itr]
    p_va = [tr_pids[i] for i in iva]; l_va = [tr_lbl[i] for i in iva]

    dl = DataLoader(FeatureBagDataset(p_tr, l_tr, feat_files, DIM, seed=SEED * 10 + fold), batch_size=1,
                    shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(SEED * 10 + fold), num_workers=0)
    dlev = DataLoader(FeatureBagDataset(p_va, l_va, feat_files, DIM, seed=SEED * 10 + fold + 1), batch_size=1,
                      shuffle=False, collate_fn=collate, num_workers=0)
    model = Classifier(DIM, 2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    counts = np.bincount(np.array(l_tr), minlength=2).astype(np.float32)
    w = torch.tensor(counts.max() / (counts + 1e-6), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    best, state, pat = -1.0, None, 0
    for _ in range(EPOCHS):
        model.train()
        for x, mask, lb, _ in dl:
            x, mask, lb = x.to(DEVICE), mask.to(DEVICE), lb.to(DEVICE).squeeze(1)
            loss = crit(model(x, mask)[0], lb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
        model.eval(); pr, tr_ = [], []
        with torch.no_grad():
            for x, mask, lb, _ in dlev:
                pr.extend(model(x.to(DEVICE), mask.to(DEVICE))[0].argmax(1).cpu().numpy())
                tr_.extend(lb.squeeze(1).numpy())
        acc = accuracy_score(tr_, pr)
        if acc > best: best, state, pat = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= PATIENCE: break
    model.load_state_dict(state); model.eval()
    return model


def predict(model, pids, lbl, feat_files):
    dl = DataLoader(FeatureBagDataset(pids, lbl, feat_files, DIM, seed=0), batch_size=1, shuffle=False,
                    collate_fn=collate, num_workers=0)
    out = []
    with torch.no_grad():
        for x, mask, lb, _ in dl:
            out.extend(F.softmax(model(x.to(DEVICE), mask.to(DEVICE))[0], 1)[:, 1].cpu().numpy())
    return np.array(out)


def metrics(y, p, tag):
    pred = (p >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    spec = 1 - fpr
    rec90 = float(tpr[np.argmin(np.abs(spec - 0.90))]) if len(tpr) else float("nan")
    m = {"tag": tag, "n": int(len(y)), "n_spindle": int(y.sum()),
         "accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
         "spindle_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
         "n_predicted_spindle": int(pred.sum()),
         "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
         "auprc": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
         "recall_at_90pct_specificity": rec90}
    print(f"  {tag}: n={m['n']} (spindle {m['n_spindle']}) acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
          f"recall={m['spindle_recall']:.3f} predicted_spindle={m['n_predicted_spindle']} "
          f"AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}", flush=True)
    return m


def main():
    cohort = json.load(open(COHORT_PATH, encoding="utf-8"))
    lbl = {str(k): int(v) for k, v in cohort["labels"].items()}
    cog_files = {os.path.basename(f)[:-3]: f for f in glob.glob(os.path.join(COG_DIR, "*.h5"))}
    bch_files = {os.path.basename(f).replace("_conch", "").replace(".h5", ""): f
                 for f in glob.glob(os.path.join(BCH_DIR, "*.h5"))}

    cog_pids = [p for p in lbl if p in cog_files and lbl[p] in (0, 2)]
    cog_y = [1 if lbl[p] == 2 else 0 for p in cog_pids]
    print(f"COG two-class: n={len(cog_pids)} ERMS={cog_y.count(0)} spindle={cog_y.count(1)}", flush=True)

    import pandas as pd
    bp = pd.read_csv(BCH_PRED, dtype={"patient_id": str})
    bp["patient_id"] = bp["patient_id"].str.zfill(7)
    bch_pids = [p for p in bp["patient_id"] if p in bch_files and int(bp.loc[bp.patient_id == p, "true"].iloc[0]) in (0, 2)]
    bch_y = np.array([1 if int(bp.loc[bp.patient_id == p, "true"].iloc[0]) == 2 else 0 for p in bch_pids])
    bch_lbl = [int(x) for x in bch_y]
    print(f"BCH two-class: n={len(bch_pids)} ERMS={int((bch_y==0).sum())} spindle={int(bch_y.sum())}", flush=True)
    print(f"BCH majority-class baseline accuracy = {max(bch_y.mean(), 1-bch_y.mean()):.3f}", flush=True)

    res = {}
    for oversample in (1, 5):
        key = "plain" if oversample == 1 else "oversample_x5"
        folds, probs = [], []
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for fi, (tr, va) in enumerate(skf.split(np.zeros(len(cog_y)), cog_y), 1):
            model = fit([cog_pids[i] for i in tr], [cog_y[i] for i in tr], cog_files, fi, oversample)
            # internal check
            pi = predict(model, [cog_pids[i] for i in va], [cog_y[i] for i in va], cog_files)
            vi = np.array([cog_y[i] for i in va])
            folds.append({"fold": fi, "internal_auroc": float(roc_auc_score(vi, pi)),
                          "internal_recall": float(recall_score(vi, (pi >= 0.5).astype(int), zero_division=0))})
            probs.append(predict(model, bch_pids, bch_lbl, bch_files))
            del model; torch.cuda.empty_cache()
            print(f"  {key} fold{fi}: internal AUROC={folds[-1]['internal_auroc']:.3f} "
                  f"recall={folds[-1]['internal_recall']:.3f}", flush=True)
        ens = np.mean(np.vstack(probs), axis=0)
        res[key] = {"folds": folds, "ensemble_bch": metrics(bch_y, ens, f"{key}_ensemble_BCH")}
        res[key]["mean_internal_auroc"] = float(np.mean([f["internal_auroc"] for f in folds]))

    json.dump(res, open(os.path.join(OUT_DIR, "spindle_binary_external_v3.json"), "w"), indent=2)
    print("=== SPINDLE BINARY EXTERNAL DONE ===")


if __name__ == "__main__":
    main()
