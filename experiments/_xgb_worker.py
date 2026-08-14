"""Standalone XGBoost worker, run as its own OS process, NEVER importing
torch. See NOTES.md: xgboost.fit() and torch (once torch has actually run a
compute op) cannot safely coexist in the same process on this machine --
whichever imports/runs first, the other segfaults (torch-first: xgboost.fit
segfaults; xgboost-first: torch.softmax's first call segfaults unless
pre-warmed). The only fully robust fix is process isolation: this script
touches numpy/pandas/sklearn-ecosystem only, writes its result to a CSV, and
the parent process (which does import torch) reads that file back rather
than trusting anything about this process's internal state or exit code.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, "/Users/tripti/Deakin_QML")
from cqgt.data.pipeline import build_paired_dataset, features_with_shock_indicator  # noqa: E402
from cqgt.metrics import full_report  # noqa: E402

N_MC = 20


def _flatten(ds, t_indices, n_mc):
    Xs = [features_with_shock_indicator(ds, t, m) for t in t_indices for m in range(n_mc)]
    ys = [ds["y_mc"][t, m] for t in t_indices for m in range(n_mc)]
    masks = [~ds["shocked_mask_mc"][t, m] for t in t_indices for m in range(n_mc)]
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(masks, axis=0)


def main(seed, out_path):
    ds = build_paired_dataset("mindensity", n_mc=N_MC, seed=seed)
    X_tr, y_tr, _ = _flatten(ds, ds["train_t"], N_MC)
    pos_frac = y_tr.mean()
    scale_pos_weight = (1 - pos_frac) / max(pos_frac, 1e-6)
    clf = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                         n_estimators=200, max_depth=4, random_state=seed)
    clf.fit(X_tr, y_tr)

    X_te, y_te, mask_te = _flatten(ds, ds["test_t"], N_MC)
    p_te = clf.predict_proba(X_te)[:, 1]
    report = full_report(y_te[mask_te], p_te[mask_te])
    n_params = int(len(clf.get_booster().trees_to_dataframe()))

    row = {"table": "T1", "model": "xgboost", "seed": seed, "n_params": n_params, **report}
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"[xgb worker seed={seed}] wrote {out_path}: auprc={report['auprc']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    main(args.seed, args.out)
