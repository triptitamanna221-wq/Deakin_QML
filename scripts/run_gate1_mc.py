"""GATE 1 refined diagnostic (see NOTES.md 'User decision on GATE 1'):
M=20 independent shock realizations per snapshot, each a separate
scenario-example with a binary "shocked this draw" feature appended to the
6-dim feature vector, and the primary AUPRC metric restricted to the
NON-directly-shocked (spillover) subset -- the only place network structure
can causally matter.

Pre-registered stopping rule (recorded in NOTES.md before this was run): if
the GCN still does not beat logistic regression on the spillover subset
here, accept the negative finding and proceed to Phase 2 regardless. No
further redesign.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from cqgt.data.cascade import generate_mc_labels_for_panel
from cqgt.data.pipeline import EXTRA_FIELDS, build_fallback_dataset, build_real_dataset
from cqgt.seeding import set_seed
from scripts.run_gate1 import QuickGCN, _edges_from_W

N_MC = 20


def build_mc_dataset(base_ds, n_mc=N_MC, seed=0):
    y, loss_frac, shocked_mask, diag = generate_mc_labels_for_panel(
        base_ds["W_panel"], base_ds["y832_panel"], base_ds["m362_panel"],
        p_shock=base_ds["p_shock"], seed=seed, n_mc=n_mc)
    base_ds.update(y_mc=y, loss_frac_mc=loss_frac, shocked_mask_mc=shocked_mask, mc_diag=diag)
    return base_ds


def _features_with_shock_indicator(ds, t, m):
    x = ds["X_std"][t]  # (n, 6)
    shocked = ds["shocked_mask_mc"][t, m].astype(np.float32)[:, None]  # (n, 1)
    return np.concatenate([x, shocked], axis=1)  # (n, 7)


def train_logreg_mc(ds, train_t, n_mc):
    Xs, ys = [], []
    for t in train_t:
        for m in range(n_mc):
            Xs.append(_features_with_shock_indicator(ds, t, m))
            ys.append(ds["y_mc"][t, m])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def train_quick_gcn_mc(ds, train_t, n_mc, epochs=60, lr=0.02, seed=0):
    set_seed(seed)
    model = QuickGCN(in_dim=7)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y_train_flat = np.concatenate([ds["y_mc"][t, m] for t in train_t for m in range(n_mc)])
    pos_frac = y_train_flat.mean()
    pos_weight = (1 - pos_frac) / max(pos_frac, 1e-6)

    # cache edge_index/edge_weight per t (identical across m draws)
    edges_by_t = {t: _edges_from_W(ds["W_panel"][t]) for t in train_t}

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        losses = []
        for t in train_t:
            edge_index, edge_weight = edges_by_t[t]
            for m in range(n_mc):
                x = torch.tensor(_features_with_shock_indicator(ds, t, m), dtype=torch.float32)
                yt = torch.tensor(ds["y_mc"][t, m], dtype=torch.float32)
                p = model(x, edge_index, edge_weight)
                w = torch.where(yt > 0, torch.tensor(pos_weight), torch.tensor(1.0))
                losses.append(torch.nn.functional.binary_cross_entropy(p, yt, weight=w))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
    return model


def evaluate_spillover_auprc(ds, test_t, n_mc, predict_fn):
    """predict_fn(t, m) -> array of per-node predicted probabilities (n,)."""
    y_all, p_all, mask_all = [], [], []
    for t in test_t:
        for m in range(n_mc):
            y_all.append(ds["y_mc"][t, m])
            p_all.append(predict_fn(t, m))
            mask_all.append(~ds["shocked_mask_mc"][t, m])
    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    mask_all = np.concatenate(mask_all)

    auprc_all = average_precision_score(y_all, p_all)
    auprc_spillover = average_precision_score(y_all[mask_all], p_all[mask_all])
    prevalence_spillover = y_all[mask_all].mean()
    return auprc_all, auprc_spillover, prevalence_spillover, mask_all.sum()


def run_refined_gate1_for(name, base_ds, n_mc=N_MC, seed=0):
    ds = build_mc_dataset(base_ds, n_mc=n_mc, seed=seed)
    train_t, test_t = ds["train_t"], ds["test_t"]

    clf = train_logreg_mc(ds, train_t, n_mc)

    def logreg_predict(t, m):
        x = _features_with_shock_indicator(ds, t, m)
        return clf.predict_proba(x)[:, 1]

    auprc_all_lr, auprc_sp_lr, prev_sp, n_sp = evaluate_spillover_auprc(ds, test_t, n_mc, logreg_predict)

    gcn = train_quick_gcn_mc(ds, train_t, n_mc, seed=seed)
    gcn.eval()
    edges_by_t_test = {t: _edges_from_W(ds["W_panel"][t]) for t in test_t}

    def gcn_predict(t, m):
        x = torch.tensor(_features_with_shock_indicator(ds, t, m), dtype=torch.float32)
        ei, ew = edges_by_t_test[t]
        with torch.no_grad():
            return gcn(x, ei, ew).numpy()

    auprc_all_gcn, auprc_sp_gcn, _, _ = evaluate_spillover_auprc(ds, test_t, n_mc, gcn_predict)

    print(f"\n=== {name} (M={n_mc} MC draws/snapshot) ===")
    print(f"  MC label diagnostics: base_rate={ds['mc_diag']['base_rate']:.4f} "
          f"direct={ds['mc_diag']['n_direct_positive']} spillover={ds['mc_diag']['n_spillover_positive']}")
    print(f"  spillover-subset prevalence: {prev_sp:.4f}  (n={n_sp} non-shocked test examples)")
    print(f"  AUPRC (all nodes)      logreg={auprc_all_lr:.4f}  gcn={auprc_all_gcn:.4f}")
    print(f"  AUPRC (spillover only) logreg={auprc_sp_lr:.4f}  gcn={auprc_sp_gcn:.4f}  "
          f"margin={auprc_sp_gcn - auprc_sp_lr:+.4f}")
    return {
        "name": name, "prevalence_spillover": prev_sp,
        "auprc_spillover_logreg": auprc_sp_lr, "auprc_spillover_gcn": auprc_sp_gcn,
        "margin_spillover": auprc_sp_gcn - auprc_sp_lr,
    }


if __name__ == "__main__":
    results = []
    for name, builder in [
        ("real_maxent", lambda: build_real_dataset("maxent", seed=0)),
        ("real_mindensity", lambda: build_real_dataset("mindensity", seed=0)),
        ("fallback_core_periphery", lambda: build_fallback_dataset(seed=0)),
    ]:
        ds = builder()
        results.append(run_refined_gate1_for(name, ds))

    print("\n=== GATE 1 REFINED SUMMARY (spillover subset only) ===")
    for r in results:
        verdict = "GCN beats logreg" if r["margin_spillover"] > 0 else "GCN does NOT beat logreg"
        print(f"{r['name']:<25s} prevalence={r['prevalence_spillover']:.4f} "
              f"logreg={r['auprc_spillover_logreg']:.4f} gcn={r['auprc_spillover_gcn']:.4f} "
              f"margin={r['margin_spillover']:+.4f}  -> {verdict}")
