"""End-to-end dataset assembly for one data source ('maxent', 'mindensity',
'fallback'), used by GATE 1 diagnostics and later training. Ties together
every module built for Phase 1: loader -> panel selection -> reconstruction
-> temporal panel -> features -> cascade labels -> treewidth.
"""
import networkx as nx
import numpy as np
import pandas as pd

from cqgt.data.cascade import (external_assets_from_marginals, generate_labels_for_panel,
                                generate_mc_labels_for_panel, tune_p_shock)
from cqgt.data.fallback import (generate_core_periphery, synthetic_extra_fields,
                                 synthetic_marginals_from_panel)
from cqgt.data.features import build_raw_feature_panel, standardize_train_split
from cqgt.data.fry15_loader import NETWORK_ANCHOR_YEARS, load_all_years
from cqgt.data.panel import select_panel
from cqgt.data.reconstruction import maxent_reconstruct, mindensity_reconstruct, rescale_marginals
from cqgt.data.splits import rolling_splits
from cqgt.data.temporal import CRISIS_WINDOW, T_DEFAULT, build_fallback_panel, build_real_anchor_panel

EXTRA_FIELDS = ("m376", "m390", "m405", "m408", "m411", "m422", "m426")


def _marginal_df_list(y832_panel, m362_panel, m370_panel, extra):
    T, n = y832_panel.shape
    out = []
    for t in range(T):
        d = {"y832": y832_panel[t], "m362": m362_panel[t], "m370": m370_panel[t]}
        for k in EXTRA_FIELDS:
            d[k] = extra[k][t]
        out.append(pd.DataFrame(d))
    return out


def build_real_dataset(method, T=T_DEFAULT, seed=0):
    """method: 'maxent' or 'mindensity'."""
    recon_fn = {"maxent": maxent_reconstruct, "mindensity": mindensity_reconstruct}[method]
    panel_sel, panel_report = select_panel()
    ids = panel_sel["id_rssd"].tolist()
    data = load_all_years()

    W_anchors, y832_anchors, m362_anchors, m370_anchors, extra_anchors = [], [], [], [], {f: [] for f in EXTRA_FIELDS}
    for y in NETWORK_ANCHOR_YEARS:
        df = data[y].set_index("id_rssd").loc[ids]
        a, l, imbalance_report = rescale_marginals(df["m362"].values, df["m370"].values)
        W_anchors.append(recon_fn(a, l))
        y832_anchors.append(df["y832"].values.astype(float))
        m362_anchors.append(a)
        m370_anchors.append(l)
        for f in EXTRA_FIELDS:
            extra_anchors[f].append(df[f].values.astype(float))

    marginal_anchors = [np.stack([m362_anchors[i], m370_anchors[i], y832_anchors[i]] +
                                  [extra_anchors[f][i] for f in EXTRA_FIELDS], axis=1)
                         for i in range(len(NETWORK_ANCHOR_YEARS))]

    W_panel, m_panel, anchor_t = build_real_anchor_panel(W_anchors, marginal_anchors, T=T, seed=seed)
    m362_panel, m370_panel, y832_panel = m_panel[:, :, 0], m_panel[:, :, 1], m_panel[:, :, 2]
    extra = {f: m_panel[:, :, 3 + i] for i, f in enumerate(EXTRA_FIELDS)}

    return _assemble(W_panel, y832_panel, m362_panel, m370_panel, extra, seed,
                      meta={"method": method, "panel_report": panel_report,
                            "imbalance_report_last_year": imbalance_report, "anchor_t": anchor_t,
                            "institution_ids": ids, "institution_names": panel_sel["name"].tolist()})


def build_fallback_dataset(T=T_DEFAULT, seed=0):
    W_panel = build_fallback_panel(T=T, seed=seed)
    y832_panel, m362_panel, m370_panel = synthetic_marginals_from_panel(W_panel, seed=seed)
    extra = synthetic_extra_fields(y832_panel, seed=seed)
    return _assemble(W_panel, y832_panel, m362_panel, m370_panel, extra, seed,
                      meta={"method": "fallback"})


def _assemble(W_panel, y832_panel, m362_panel, m370_panel, extra, seed, meta):
    T, n, _ = W_panel.shape
    marginal_dfs = _marginal_df_list(y832_panel, m362_panel, m370_panel, extra)

    X_raw = build_raw_feature_panel(marginal_dfs, W_panel)
    train_t, val_t, test_t = rolling_splits(T)
    X_std, mu, sd = standardize_train_split(X_raw, train_end_t=len(train_t))

    p_shock, shock_diag = tune_p_shock(W_panel, y832_panel, m362_panel, seed=seed)
    y, loss_frac, label_diag = generate_labels_for_panel(W_panel, y832_panel, m362_panel,
                                                          p_shock=p_shock, seed=seed)

    treewidths = []
    for t in range(T):
        G = nx.from_numpy_array((W_panel[t] > 0).astype(int), create_using=nx.DiGraph)
        tw, _ = nx.algorithms.approximation.treewidth_min_fill_in(G.to_undirected())
        treewidths.append(tw)

    return {
        "W_panel": W_panel, "y832_panel": y832_panel, "m362_panel": m362_panel,
        "m370_panel": m370_panel, "X_raw": X_raw, "X_std": X_std,
        "y": y, "loss_frac": loss_frac,
        "train_t": train_t, "val_t": val_t, "test_t": test_t,
        "treewidths": np.array(treewidths),
        "p_shock": p_shock, "label_diag": label_diag, "meta": meta,
    }


N_MC_DEFAULT = 20


def add_mc_labels(ds, n_mc=N_MC_DEFAULT, seed=0):
    """Attach M=20 (default) independent Monte Carlo shock draws per
    snapshot to an existing dataset dict, in place -- see NOTES.md 'User
    decision on GATE 1'. `seed` controls the shock draws; the SAME (source,
    seed, n_mc) triple always yields the SAME scenario set, which is what
    makes cross-model comparisons paired (identical splits from `_assemble`,
    identical shock draws from here) rather than independently resampled."""
    y, loss_frac, shocked_mask, diag = generate_mc_labels_for_panel(
        ds["W_panel"], ds["y832_panel"], ds["m362_panel"],
        p_shock=ds["p_shock"], seed=seed, n_mc=n_mc)
    ds.update(y_mc=y, loss_frac_mc=loss_frac, shocked_mask_mc=shocked_mask, mc_diag=diag)
    return ds


def build_paired_dataset(source, n_mc=N_MC_DEFAULT, seed=0, T=T_DEFAULT):
    """The single entry point every model (CQGT and every baseline) should
    use for a fair, paired comparison: source in {'maxent', 'mindensity',
    'fallback'}, same seed => identical rolling splits AND identical M=20
    shock draws across every model trained against the result."""
    base = (build_fallback_dataset(T=T, seed=seed) if source == "fallback"
            else build_real_dataset(source, T=T, seed=seed))
    return add_mc_labels(base, n_mc=n_mc, seed=seed)


def features_with_shock_indicator(ds, t, m):
    """(n, 7) feature matrix for scenario (t, m): the 6 standardized
    proxy/real features plus the binary "shocked this draw" scenario
    indicator (input, not outcome -- see NOTES.md)."""
    x = ds["X_std"][t]
    shocked = ds["shocked_mask_mc"][t, m].astype(np.float32)[:, None]
    return np.concatenate([x, shocked], axis=1)


ANCHOR_DATE_START, ANCHOR_DATE_END = "2020-12-31", "2023-12-31"


def attach_macro_factors(ds, seed=0):
    """Attach ds['macro_factors'], shape (T, K_FACTORS). Real datasets get
    real FRED-derived factors mapped onto the synthetic weekly calendar
    (BRIEF.md's real-data path); the fallback dataset -- which has no real
    calendar to anchor to -- gets a seeded synthetic AR(1) 3-factor process
    instead, per BRIEF.md Sec 2.1's fallback spec. Both are disclosed."""
    from cqgt.data.macro import K_FACTORS, build_macro_factors
    T = ds["W_panel"].shape[0]
    if ds["meta"]["method"] == "fallback":
        rng = np.random.default_rng(seed)
        factors = np.zeros((T, K_FACTORS))
        for k in range(K_FACTORS):
            for t in range(1, T):
                factors[t, k] = 0.95 * factors[t - 1, k] + rng.normal(0, 0.3)
        ds["macro_factors"] = factors
        ds["macro_report"] = {"source": "synthetic_ar1", "rho": 0.95}
    else:
        factors, loadings, dates, report = build_macro_factors(T, ANCHOR_DATE_START, ANCHOR_DATE_END)
        ds["macro_factors"] = factors
        ds["macro_report"] = {"source": "real_fred", **report}
    return ds
