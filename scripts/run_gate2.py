"""GATE 2 (BRIEF.md Sec 6): training loss decreases monotonically-ish over
epochs; CQGT beats the prevalence floor on validation. Report the learning
curve and parameter count (for the parity requirement on later baseline
comparisons).
"""
import numpy as np
from sklearn.metrics import average_precision_score

from cqgt.data.pipeline import attach_macro_factors, build_paired_dataset
from cqgt.train import predict, train_cqgt

N_MC = 20          # full paired M draws -- used for evaluation
N_MC_TRAIN = 4     # compute-budget knob for training only, see cqgt/train.py
EPOCHS_PER_STAGE = 8


def run_gate2_for(name, source, seed=0):
    ds = build_paired_dataset(source, n_mc=N_MC, seed=seed)
    attach_macro_factors(ds, seed=seed)
    n = ds["W_panel"].shape[1]
    edges = list(map(tuple, np.argwhere(ds["W_panel"][0] > 0)))

    model, full_history, per_stage = train_cqgt(
        ds, n_qubits=n, n_features=7, edges=edges, layer_schedule=(1, 2, 3),
        epochs_per_stage=EPOCHS_PER_STAGE, seed=seed, n_mc_train=N_MC_TRAIN)

    y_val, p_val, shocked_val = predict(model, ds, ds["val_t"])
    spillover_mask = ~shocked_val
    prevalence = y_val[spillover_mask].mean()
    auprc_spillover = average_precision_score(y_val[spillover_mask], p_val[spillover_mask])
    auprc_all = average_precision_score(y_val, p_val)

    print(f"\n=== {name} ===")
    print(f"  n_edges={len(edges)}  CQGT parameter count={model.n_parameters()}")
    for L, hist in per_stage.items():
        print(f"  stage L={L}: loss {hist[0]:.4f} -> {hist[-1]:.4f} "
              f"(first 3: {[round(h,4) for h in hist[:3]]}, last 3: {[round(h,4) for h in hist[-3:]]})")
    print(f"  validation: prevalence(spillover)={prevalence:.4f}  "
          f"AUPRC(spillover)={auprc_spillover:.4f}  AUPRC(all)={auprc_all:.4f}")
    verdict = "PASS (beats prevalence)" if auprc_spillover > prevalence else "FAIL (does not beat prevalence)"
    print(f"  -> {verdict}")
    return {
        "name": name, "n_params": model.n_parameters(), "n_edges": len(edges),
        "full_history": full_history, "prevalence": prevalence,
        "auprc_spillover": auprc_spillover, "auprc_all": auprc_all,
    }


if __name__ == "__main__":
    results = [run_gate2_for(name, source) for name, source in
               [("real_maxent", "maxent"), ("real_mindensity", "mindensity"),
                ("fallback_core_periphery", "fallback")]]

    print("\n=== GATE 2 SUMMARY ===")
    for r in results:
        print(f"{r['name']:<25s} params={r['n_params']:4d} edges={r['n_edges']:3d} "
              f"prevalence={r['prevalence']:.4f} AUPRC(spillover)={r['auprc_spillover']:.4f}  "
              f"loss {r['full_history'][0]:.4f} -> {r['full_history'][-1]:.4f}")
