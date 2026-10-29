"""v3 Architecture ablation, second seed (run 2) for run-to-run stability check.

Reuses the model definitions and training loop from architecture_ablation_v3.py
but runs on the frozen public cohort with a different training seed, and writes
to architecture_ablation_v3_run2.json so run 1 (seed 42) and run 2 (seed 2026)
can be compared. Only the two encoders used in the manuscript (CONCH,
H-optimus-0) are run.
"""

import os, sys, glob, json, importlib.util, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ab",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture_ablation_v3.py"))
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ab.OUT_DIR = os.path.join(ROOT, "outputs", "data_snapshot", "data", "results")
ab.COHORT_PATH = os.path.join(ROOT, "outputs", "data_snapshot", "data", "cohort_and_folds_public.json")
ab.SEED = 2026
ab.MAX_PATCHES = 4096
ab.ENCODERS = {
    "conch": (r"F:\Pathology\features_rms_conch", 768),
    "hoptimus0": (r"F:\Pathology\features\features_cog_hoptimus0", 1536),
}
OUT_NAME = "architecture_ablation_v3_run2.json"


def run():
    with open(ab.COHORT_PATH, "r", encoding="utf-8") as f:
        cohort = json.load(f)
    labels = {str(k): int(v) for k, v in cohort["labels"].items()}
    folds = cohort["folds"]
    results = {}
    t_start = time.time()
    for enc, (feat_dir, dim) in ab.ENCODERS.items():
        feat_files = {os.path.basename(f).replace(".h5", ""): f for f in glob.glob(os.path.join(feat_dir, "*.h5"))}
        enc_res = {}
        for arch in ab.ARCHS:
            fold_res = []
            for fi in range(1, 6):
                fd = folds[f"fold_{fi}"]
                train_pids = [p for p in fd["train_pids"] if p in feat_files]
                val_pids = [p for p in fd["val_pids"] if p in feat_files]
                ab.log.info("RUN2 %s %s fold%d", enc, arch, fi)
                r = ab.train_one(arch, dim, feat_files, train_pids, val_pids, labels, fi)
                fold_res.append({k: v for k, v in r.items() if k not in ("preds", "trues", "probs", "pids")})
            accs = [r["acc"] for r in fold_res]; f1ms = [r["f1m"] for r in fold_res]
            enc_res[arch] = {"acc_mean": float(__import__("numpy").mean(accs)),
                             "acc_std": float(__import__("numpy").std(accs)),
                             "f1m_mean": float(__import__("numpy").mean(f1ms)),
                             "f1m_std": float(__import__("numpy").std(f1ms)),
                             "folds": fold_res}
            ab.log.info("RUN2 %s %s: acc=%.4f +/- %.4f  elapsed=%.0fs",
                        enc, arch, float(__import__("numpy").mean(accs)),
                        float(__import__("numpy").std(accs)), time.time() - t_start)
        results[enc] = enc_res
    with open(os.path.join(ab.OUT_DIR, OUT_NAME), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    ab.log.info("=== ARCHITECTURE ABLATION RUN2 DONE (%s) ===", OUT_NAME)
    print("RUN2_COMPLETE")


if __name__ == "__main__":
    run()
