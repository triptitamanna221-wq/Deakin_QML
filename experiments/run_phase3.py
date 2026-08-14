"""Phase 3 main sweep (BRIEF.md Sec 2.4/2.5/2.7, cut to the minimum
publishable experiment per the user's explicit STEP 2 scope): mindensity
dataset only, 5 baselines (logreg, XGBoost, GCN, GAT, generic VQC), CQGT
full model, 3 ablations (no_hamiltonian, classical_attention, random_edges),
3 seeds, spillover-only primary metric, paired evaluation (identical splits
+ shock draws per seed across every model, from build_paired_dataset),
parameter counts in every row.

Layer-wise growth is DROPPED for every quantum-circuit model here (CQGT,
generic VQC, all 3 ablations) -- see NOTES.md's Phase 2/3 finding: the
Attempt-1 gradient-norm diagnostic found no barren-plateau signature at
N=12, so growth (specified as C4 mitigation) is empirically unnecessary and
every quantum-circuit model instead trains directly at a fixed depth for a
matched, confound-free protocol. Full growth is Future Work.

Produces results/T1_raw.csv and results/T2_raw.csv (one row per
(model, seed)); results/T1.csv and results/T2.csv (aggregated mean +/- 95%
bootstrap CI across seeds -- disclosed in NOTES.md as a weak estimate at
n=3 seeds, reported because it's what the protocol specifies, not because
3 points give a trustworthy CI).
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GATConv, GCNConv

from baselines.gnn import GNNBaseline
from baselines.vqc import GenericVQC
from cqgt.counterfactual import exact_counterfactual
from cqgt.data.cascade import DEFAULT_CAPITAL_RATIO, DEFAULT_SHOCK_FRAC, ground_truth_delta_r
from cqgt.data.pipeline import attach_macro_factors, build_paired_dataset, features_with_shock_indicator
from cqgt.metrics import counterfactual_precision_at_k, counterfactual_spearman, full_report
from cqgt.model import CQGTModel
from cqgt.seeding import set_seed
from cqgt.train import predict as cqgt_predict
from cqgt.train import train_stage

SEEDS = (0, 1, 2)
N_MC = 20
QUANTUM_EPOCHS = 25
QUANTUM_LR = 0.01
GNN_EPOCHS = 150
GNN_LR = 0.02
N_FEATURES = 7  # 6 standardized proxies + shocked-this-draw indicator
T3_N_SNAPSHOTS = 3  # pooled test snapshots for the counterfactual comparison


# ---------------------------------------------------------------- data ----

def _edges_from_W(W_t):
    idx = np.argwhere(W_t > 0)
    edge_index = torch.tensor(idx.T, dtype=torch.long)
    edge_weight = torch.tensor(W_t[idx[:, 0], idx[:, 1]], dtype=torch.float32)
    if edge_weight.numel() > 0:
        edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight


def _flatten_scenarios(ds, t_indices, n_mc):
    Xs = [features_with_shock_indicator(ds, t, m) for t in t_indices for m in range(n_mc)]
    ys = [ds["y_mc"][t, m] for t in t_indices for m in range(n_mc)]
    masks = [~ds["shocked_mask_mc"][t, m] for t in t_indices for m in range(n_mc)]
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(masks, axis=0)


def _spillover_report(y, p, mask):
    return full_report(y[mask], p[mask])


# ---------------------------------------------------------- classical -----

def run_logreg(ds, train_t, test_t, n_mc):
    X_tr, y_tr, _ = _flatten_scenarios(ds, train_t, n_mc)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    X_te, y_te, mask_te = _flatten_scenarios(ds, test_t, n_mc)
    p_te = clf.predict_proba(X_te)[:, 1]
    n_params = clf.coef_.size + clf.intercept_.size
    return _spillover_report(y_te, p_te, mask_te), n_params


def run_xgb(seed):
    """xgboost runs in a completely separate OS process (experiments/
    _xgb_worker.py), never importing torch. NOTES.md: xgboost.fit() and any
    prior torch compute op cannot safely coexist in one process on this
    machine, in EITHER import order (torch-first: xgboost.fit segfaults;
    xgboost-first: torch.softmax's first-ever call segfaults). Reordering
    imports only moves which op breaks -- the only fully robust fix is
    process isolation, so this reads the worker's result back from a file
    rather than trusting anything about a shared process/exit code."""
    out_path = f"results/_xgb_seed{seed}.csv"
    result = subprocess.run(
        [sys.executable, "-m", "experiments._xgb_worker", "--seed", str(seed), "--out", out_path],
        capture_output=True, text=True)
    try:
        row = pd.read_csv(out_path).iloc[0]
    except (FileNotFoundError, pd.errors.EmptyDataError, IndexError) as e:
        raise RuntimeError(
            f"xgboost worker (seed={seed}) produced no usable output file. "
            f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        ) from e
    report = {k: float(row[k]) for k in ("prevalence", "auprc", "auroc", "f1_youden", "brier", "ece")}
    return report, int(row["n_params"])


# ----------------------------------------------------------------- gnn ----

def train_gnn(ds, train_t, n_mc, conv_cls, seed, epochs=GNN_EPOCHS, lr=GNN_LR, **conv_kwargs):
    set_seed(seed)
    model = GNNBaseline(conv_cls, in_dim=N_FEATURES, **conv_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y_train_flat = np.concatenate([ds["y_mc"][t, m] for t in train_t for m in range(n_mc)])
    pos_frac = y_train_flat.mean()
    pos_weight = (1 - pos_frac) / max(pos_frac, 1e-6)
    edges_by_t = {t: _edges_from_W(ds["W_panel"][t]) for t in train_t}

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        losses = []
        for t in train_t:
            edge_index, edge_weight = edges_by_t[t]
            for m in range(n_mc):
                x = torch.tensor(features_with_shock_indicator(ds, t, m), dtype=torch.float32)
                yt = torch.tensor(ds["y_mc"][t, m], dtype=torch.float32)
                p = model(x, edge_index, edge_weight)
                w = torch.where(yt > 0, torch.tensor(pos_weight), torch.tensor(1.0))
                losses.append(torch.nn.functional.binary_cross_entropy(p, yt, weight=w))
        torch.stack(losses).mean().backward()
        opt.step()
    return model


def eval_gnn(model, ds, test_t, n_mc):
    model.eval()
    y_all, p_all, mask_all = [], [], []
    with torch.no_grad():
        for t in test_t:
            edge_index, edge_weight = _edges_from_W(ds["W_panel"][t])
            for m in range(n_mc):
                x = torch.tensor(features_with_shock_indicator(ds, t, m), dtype=torch.float32)
                p_all.append(model(x, edge_index, edge_weight).numpy())
                y_all.append(ds["y_mc"][t, m])
                mask_all.append(~ds["shocked_mask_mc"][t, m])
    y_all, p_all, mask_all = np.concatenate(y_all), np.concatenate(p_all), np.concatenate(mask_all)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _spillover_report(y_all, p_all, mask_all), n_params


# ------------------------------------------------------------- quantum ----

def train_and_eval_quantum(model, ds, train_t, test_t, n_mc, seed, epochs=QUANTUM_EPOCHS, lr=QUANTUM_LR):
    model, hist = train_stage(model, ds, epochs=epochs, lr=lr, seed=seed, n_mc_train=None)
    y_te, p_te, shocked_te = cqgt_predict(model, ds, test_t)
    mask_te = ~shocked_te
    n_params = model.n_parameters() if hasattr(model, "n_parameters") else \
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    return _spillover_report(y_te, p_te, mask_te), n_params, hist, model


def random_edges_matching_density(n, n_edges, seed):
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = rng.choice(len(all_pairs), size=min(n_edges, len(all_pairs)), replace=False)
    return [all_pairs[k] for k in idx]


# --------------------------------------------------------- counterfactual (T3)

def _gcn_gradient_saliency_delta_r(gcn_model, W_t, x_t, edges):
    """Phase-1-style screening baseline (BRIEF.md Sec 2.3/T3): dR/dw_ij via
    autograd on the trained GCN's mean output, first-order Taylor estimate
    of Delta-R = R0 - R(w_ij=0) ~= (dR/dw_ij) * w_ij. This is the
    "gradient-saliency baseline on the GCN" T3 explicitly requires -- without
    it the causal module is unevaluated (BRIEF.md Sec 4)."""
    idx = np.argwhere(W_t > 0)
    edge_index = torch.tensor(idx.T, dtype=torch.long)
    w_raw = torch.tensor(W_t[idx[:, 0], idx[:, 1]], dtype=torch.float32)
    edge_weight = (w_raw / w_raw.max()).clone().requires_grad_(True)
    gcn_model.eval()
    p = gcn_model(torch.tensor(x_t, dtype=torch.float32), edge_index, edge_weight)
    R = p.mean()
    grad_w = torch.autograd.grad(R, edge_weight)[0]
    pred = {}
    for k, (i, j) in enumerate(zip(idx[:, 0], idx[:, 1])):
        if (int(i), int(j)) in edges:
            pred[(int(i), int(j))] = float(grad_w[k].item() * edge_weight[k].item())
    return pred


def _cqgt_predict_fn(model, macro_t):
    def predict_fn(W, X):
        x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, n, F)
        with torch.no_grad():
            p = model(x_t, W, macro_t)
        return p.squeeze(0).numpy()
    return predict_fn


def run_t3(ds, edges, cqgt_trained, gcn_trained, test_t, seed, n_snapshots=T3_N_SNAPSHOTS, m=0):
    """Pools edge-level (ground_truth_delta_r, predicted_delta_r) pairs over
    the first `n_snapshots` test-set snapshots' m=0 MC draw (the edge SET is
    fixed across the whole panel -- see cqgt/model.py's docstring -- so
    pooling snapshots adds independent feature/shock states over the same
    edges rather than duplicating them)."""
    macro_factors = torch.tensor(ds["macro_factors"], dtype=torch.float32)
    true_all, cqgt_pred_all, gcn_pred_all = [], [], []
    for t in test_t[:n_snapshots]:
        W_t = ds["W_panel"][t]
        y832_t, m362_t = ds["y832_panel"][t], ds["m362_panel"][t]
        shocked_idx = np.where(ds["shocked_mask_mc"][t, m])[0].tolist()
        true_dr = ground_truth_delta_r(W_t, y832_t, m362_t, shocked_idx, DEFAULT_SHOCK_FRAC,
                                        DEFAULT_CAPITAL_RATIO)

        x_t = features_with_shock_indicator(ds, t, m)
        cqgt_dr = exact_counterfactual(_cqgt_predict_fn(cqgt_trained, macro_factors[t]), W_t, x_t, edges)
        gcn_dr = _gcn_gradient_saliency_delta_r(gcn_trained, W_t, x_t, edges)

        common_edges = [e for e in edges if e in true_dr and e in cqgt_dr and e in gcn_dr]
        true_all.extend(true_dr[e] for e in common_edges)
        cqgt_pred_all.extend(cqgt_dr[e] for e in common_edges)
        gcn_pred_all.extend(gcn_dr[e] for e in common_edges)

    return [
        {"table": "T3", "model": "cqgt_full", "seed": seed, "n_edges_pooled": len(true_all),
         "spearman_rho": counterfactual_spearman(true_all, cqgt_pred_all),
         "precision_at_10": counterfactual_precision_at_k(true_all, cqgt_pred_all, k=10)},
        {"table": "T3", "model": "gcn_gradient_saliency", "seed": seed, "n_edges_pooled": len(true_all),
         "spearman_rho": counterfactual_spearman(true_all, gcn_pred_all),
         "precision_at_10": counterfactual_precision_at_k(true_all, gcn_pred_all, k=10)},
    ]


# ------------------------------------------------------------- runner -----

def run_seed(seed, log=print):
    t0 = time.time()
    ds = build_paired_dataset("mindensity", n_mc=N_MC, seed=seed)
    attach_macro_factors(ds, seed=seed)
    n = ds["W_panel"].shape[1]
    edges = list(map(tuple, np.argwhere(ds["W_panel"][0] > 0)))
    train_t, test_t = ds["train_t"], ds["test_t"]
    log(f"[seed {seed}] dataset built in {time.time()-t0:.1f}s, n_edges={len(edges)}")

    rows_t1, rows_t2 = [], []

    def add(rows, table, model_name, report, n_params, extra=None):
        row = {"table": table, "model": model_name, "seed": seed, "n_params": n_params, **report}
        if extra:
            row.update(extra)
        rows.append(row)
        log(f"[seed {seed}] {table}/{model_name}: n_params={n_params} "
            f"prevalence={report['prevalence']:.4f} auprc={report['auprc']:.4f}")

    # --- T1 baselines ---
    rep, np_ = run_logreg(ds, train_t, test_t, N_MC)
    add(rows_t1, "T1", "logreg", rep, np_)

    rep, np_ = run_xgb(seed)
    add(rows_t1, "T1", "xgboost", rep, np_)

    gcn = train_gnn(ds, train_t, N_MC, GCNConv, seed)
    rep, np_ = eval_gnn(gcn, ds, test_t, N_MC)
    add(rows_t1, "T1", "gcn", rep, np_)

    gat = train_gnn(ds, train_t, N_MC, GATConv, seed, heads=2, concat=False)
    rep, np_ = eval_gnn(gat, ds, test_t, N_MC)
    add(rows_t1, "T1", "gat", rep, np_)

    vqc = GenericVQC(n_qubits=n, n_features=N_FEATURES, n_layers=3)
    rep, np_, hist, _ = train_and_eval_quantum(vqc, ds, train_t, test_t, N_MC, seed)
    add(rows_t1, "T1", "generic_vqc", rep, np_, {"loss_start": hist[0], "loss_end": hist[-1]})

    cqgt_full = CQGTModel(n_qubits=n, n_features=N_FEATURES, edges=edges, n_layers=3)
    rep, np_, hist, cqgt_trained = train_and_eval_quantum(cqgt_full, ds, train_t, test_t, N_MC, seed)
    add(rows_t1, "T1", "cqgt_full", rep, np_, {"loss_start": hist[0], "loss_end": hist[-1]})
    # also the T2 anchor row
    add(rows_t2, "T2", "full_model", rep, np_, {"loss_start": hist[0], "loss_end": hist[-1]})

    # --- T2 ablations (all fixed L=3, identical protocol to cqgt_full) ---
    no_ham = CQGTModel(n_qubits=n, n_features=N_FEATURES, edges=edges, n_layers=3, use_hamiltonian=False)
    rep, np_, hist, _ = train_and_eval_quantum(no_ham, ds, train_t, test_t, N_MC, seed)
    add(rows_t2, "T2", "no_hamiltonian", rep, np_, {"loss_start": hist[0], "loss_end": hist[-1]})

    classical_attn = CQGTModel(n_qubits=n, n_features=N_FEATURES, edges=edges, n_layers=3,
                                use_quantum_attention=False)
    rep, np_, hist, _ = train_and_eval_quantum(classical_attn, ds, train_t, test_t, N_MC, seed)
    add(rows_t2, "T2", "classical_attention", rep, np_, {"loss_start": hist[0], "loss_end": hist[-1]})

    rand_edges = random_edges_matching_density(n, len(edges), seed=seed)
    random_topo = CQGTModel(n_qubits=n, n_features=N_FEATURES, edges=rand_edges, n_layers=3)
    rep, np_, hist, _ = train_and_eval_quantum(random_topo, ds, train_t, test_t, N_MC, seed)
    add(rows_t2, "T2", "random_edges", rep, np_, {"loss_start": hist[0], "loss_end": hist[-1]})

    # --- T3 counterfactual attribution: CQGT vs. GCN gradient-saliency ---
    rows_t3 = run_t3(ds, edges, cqgt_trained, gcn, test_t, seed)
    for r in rows_t3:
        log(f"[seed {seed}] T3/{r['model']}: spearman_rho={r['spearman_rho']:.4f} "
            f"P@10={r['precision_at_10']:.4f} (n_pooled={r['n_edges_pooled']})")

    log(f"[seed {seed}] done in {(time.time()-t0)/60:.1f} min")
    return rows_t1, rows_t2, rows_t3, cqgt_trained, ds, edges


def bootstrap_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.any(np.isnan(values)):
        return float(np.nanmean(values)), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def aggregate(rows, table_name):
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("table", "model", "seed")]
    out = []
    for model_name, g in df.groupby("model"):
        row = {"table": table_name, "model": model_name, "n_seeds": len(g)}
        for c in metric_cols:
            mean, lo, hi = bootstrap_ci(g[c].values)
            row[f"{c}_mean"] = mean
            row[f"{c}_ci_lo"] = lo
            row[f"{c}_ci_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)


def run_single_seed_to_disk(seed):
    """Entry point for one subprocess (see run_parallel below): runs one
    seed end to end and writes its own partial CSVs, so seeds can be
    executed as independent OS processes (real parallelism -- qsim is
    compute-bound and does not benefit much from torch's own intra-op
    threading at this circuit size, see NOTES.md's thread-contention
    benchmark) rather than fighting over Python's GIL in one process."""
    rows_t1, rows_t2, rows_t3, *_ = run_seed(seed)
    pd.DataFrame(rows_t1).to_csv(f"results/T1_raw_seed{seed}.csv", index=False)
    pd.DataFrame(rows_t2).to_csv(f"results/T2_raw_seed{seed}.csv", index=False)
    pd.DataFrame(rows_t3).to_csv(f"results/T3_raw_seed{seed}.csv", index=False)


def main(seeds=SEEDS):
    """Serial fallback / single-seed mode. For the real multi-seed sweep,
    use run_parallel (spawns one OS process per seed)."""
    all_t1, all_t2, all_t3 = [], [], []
    for seed in seeds:
        rows_t1, rows_t2, rows_t3, *_ = run_seed(seed)
        all_t1.extend(rows_t1)
        all_t2.extend(rows_t2)
        all_t3.extend(rows_t3)
    _write_outputs(all_t1, all_t2, all_t3)


def _write_outputs(all_t1, all_t2, all_t3):
    pd.DataFrame(all_t1).to_csv("results/T1_raw.csv", index=False)
    pd.DataFrame(all_t2).to_csv("results/T2_raw.csv", index=False)
    pd.DataFrame(all_t3).to_csv("results/T3_raw.csv", index=False)

    aggregate(all_t1, "T1").to_csv("results/T1.csv", index=False)
    aggregate(all_t2, "T2").to_csv("results/T2.csv", index=False)
    aggregate(all_t3, "T3").to_csv("results/T3.csv", index=False)
    print("\nWrote results/T1.csv, T2.csv, T3.csv (+ _raw.csv per-seed detail)")


def aggregate_from_disk(seeds=SEEDS):
    """Read the per-seed partial CSVs written by run_single_seed_to_disk
    (i.e. by `python -m experiments.run_phase3 --seed N` subprocesses) and
    produce the final aggregated tables."""
    all_t1, all_t2, all_t3 = [], [], []
    for seed in seeds:
        all_t1.append(pd.read_csv(f"results/T1_raw_seed{seed}.csv"))
        all_t2.append(pd.read_csv(f"results/T2_raw_seed{seed}.csv"))
        all_t3.append(pd.read_csv(f"results/T3_raw_seed{seed}.csv"))
    all_t1, all_t2, all_t3 = pd.concat(all_t1), pd.concat(all_t2), pd.concat(all_t3)
    _write_outputs(all_t1.to_dict("records"), all_t2.to_dict("records"), all_t3.to_dict("records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--seed", type=int, default=None,
                         help="Run exactly one seed and write partial CSVs (subprocess mode).")
    parser.add_argument("--aggregate-only", action="store_true",
                         help="Skip training; read existing per-seed partial CSVs and aggregate.")
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate_from_disk(seeds=tuple(args.seeds))
    elif args.seed is not None:
        run_single_seed_to_disk(args.seed)
    else:
        main(seeds=tuple(args.seeds))
