"""Convergence-curve diagnostic at FULL M=20 (no training-budget reduction),
run on mindensity only (the Phase 3 ablation-primary dataset). Used to
empirically justify the epochs-per-stage budget for Phase 3 -- per explicit
instruction, we do not cut epochs below what this curve shows is needed.
"""
import time
import numpy as np
from sklearn.metrics import average_precision_score

from cqgt.data.pipeline import attach_macro_factors, build_paired_dataset
from cqgt.train import predict, train_cqgt

EPOCHS_PER_STAGE = 20

ds = build_paired_dataset("mindensity", n_mc=20, seed=0)
attach_macro_factors(ds, seed=0)
n = ds["W_panel"].shape[1]
edges = list(map(tuple, np.argwhere(ds["W_panel"][0] > 0)))
print(f"n_edges={len(edges)}", flush=True)

start = time.time()
model, full_history, per_stage = train_cqgt(
    ds, n_qubits=n, n_features=7, edges=edges, layer_schedule=(1, 2, 3),
    epochs_per_stage=EPOCHS_PER_STAGE, seed=0, n_mc_train=None)
elapsed = time.time() - start

print(f"\nTotal wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
print(f"CQGT parameter count: {model.n_parameters()}", flush=True)
for L, hist in per_stage.items():
    print(f"stage L={L} ({len(hist)} epochs): {[round(h,4) for h in hist]}", flush=True)

y_val, p_val, shocked_val = predict(model, ds, ds["val_t"])
spillover_mask = ~shocked_val
prevalence = y_val[spillover_mask].mean()
auprc_spillover = average_precision_score(y_val[spillover_mask], p_val[spillover_mask])
print(f"\nvalidation: prevalence(spillover)={prevalence:.4f}  AUPRC(spillover)={auprc_spillover:.4f}", flush=True)
print("PASS" if auprc_spillover > prevalence else "FAIL", flush=True)
