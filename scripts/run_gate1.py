"""GATE 1 diagnostics (BRIEF.md Sec 6): for each dataset variant (maxent,
mindensity, fallback), report label base rate, per-snapshot treewidth, and
AUPRC of logistic-regression-on-features vs. a quick GCN on held-out data.

Pass/fail criterion: the GCN must beat logistic regression by a clear
margin. If it does not, labels do not depend on the network and everything
downstream is meaningless -- this script reports the honest result either
way, it does not tune anything to make the comparison look better.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GCNConv

from cqgt.data.pipeline import build_fallback_dataset, build_real_dataset
from cqgt.seeding import set_seed


class QuickGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden=16):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_weight):
        h = torch.relu(self.conv1(x, edge_index, edge_weight))
        h = torch.relu(self.conv2(h, edge_index, edge_weight))
        return torch.sigmoid(self.head(h)).squeeze(-1)


def _edges_from_W(W_t):
    idx = np.argwhere(W_t > 0)
    edge_index = torch.tensor(idx.T, dtype=torch.long)
    edge_weight = torch.tensor(W_t[idx[:, 0], idx[:, 1]], dtype=torch.float32)
    if edge_weight.numel() > 0:
        edge_weight = edge_weight / edge_weight.max()
    return edge_index, edge_weight


def train_quick_gcn(ds, epochs=150, lr=0.02, seed=0):
    set_seed(seed)
    n = ds["X_std"].shape[1]
    model = QuickGCN(in_dim=ds["X_std"].shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y_train = ds["y"][ds["train_t"]]
    pos_frac = y_train.mean()
    pos_weight = (1 - pos_frac) / max(pos_frac, 1e-6)

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        losses = []
        for t in ds["train_t"]:
            x = torch.tensor(ds["X_std"][t], dtype=torch.float32)
            edge_index, edge_weight = _edges_from_W(ds["W_panel"][t])
            yt = torch.tensor(ds["y"][t], dtype=torch.float32)
            p = model(x, edge_index, edge_weight)
            w = torch.where(yt > 0, torch.tensor(pos_weight), torch.tensor(1.0))
            loss = torch.nn.functional.binary_cross_entropy(p, yt, weight=w)
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for t in ds["test_t"]:
            x = torch.tensor(ds["X_std"][t], dtype=torch.float32)
            edge_index, edge_weight = _edges_from_W(ds["W_panel"][t])
            preds.append(model(x, edge_index, edge_weight).numpy())
    return np.concatenate(preds)


def train_logreg(ds):
    X_train = ds["X_std"][ds["train_t"]].reshape(-1, ds["X_std"].shape[-1])
    y_train = ds["y"][ds["train_t"]].reshape(-1)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    X_test = ds["X_std"][ds["test_t"]].reshape(-1, ds["X_std"].shape[-1])
    return clf.predict_proba(X_test)[:, 1]


def run_gate1_for(name, ds):
    y_test = ds["y"][ds["test_t"]].reshape(-1)
    prevalence = ds["y"].mean()

    p_logreg = train_logreg(ds)
    auprc_logreg = average_precision_score(y_test, p_logreg)

    p_gcn = train_quick_gcn(ds)
    auprc_gcn = average_precision_score(y_test, p_gcn)

    tw = ds["treewidths"]
    print(f"\n=== {name} ===")
    print(f"  label base rate (full panel): {ds['y'].mean():.4f}  "
          f"(direct={ds['label_diag']['n_direct_positive']}, "
          f"spillover={ds['label_diag']['n_spillover_positive']}, "
          f"p_shock tuned={ds['p_shock']:.4f})")
    print(f"  treewidth per snapshot: min={tw.min()} max={tw.max()} mean={tw.mean():.2f}")
    print(f"  prevalence (trivial AUPRC floor): {prevalence:.4f}")
    print(f"  AUPRC logreg-on-features: {auprc_logreg:.4f}")
    print(f"  AUPRC quick-GCN:          {auprc_gcn:.4f}")
    margin = auprc_gcn - auprc_logreg
    print(f"  GCN - logreg margin: {margin:+.4f}  "
          f"-> {'PASS (clear margin)' if margin > 0.03 else 'DOES NOT CLEARLY BEAT LOGREG'}")
    return {
        "name": name, "base_rate": ds["y"].mean(), "prevalence": prevalence,
        "treewidth_min": int(tw.min()), "treewidth_max": int(tw.max()), "treewidth_mean": float(tw.mean()),
        "auprc_logreg": auprc_logreg, "auprc_gcn": auprc_gcn, "margin": margin,
    }


if __name__ == "__main__":
    results = []
    for name, builder in [
        ("real_maxent", lambda: build_real_dataset("maxent", seed=0)),
        ("real_mindensity", lambda: build_real_dataset("mindensity", seed=0)),
        ("fallback_core_periphery", lambda: build_fallback_dataset(seed=0)),
    ]:
        ds = builder()
        results.append(run_gate1_for(name, ds))

    print("\n=== GATE 1 SUMMARY ===")
    for r in results:
        print(f"{r['name']:<25s} base_rate={r['base_rate']:.3f} prevalence={r['prevalence']:.3f} "
              f"tw=[{r['treewidth_min']},{r['treewidth_max']}] "
              f"logreg={r['auprc_logreg']:.3f} gcn={r['auprc_gcn']:.3f} margin={r['margin']:+.3f}")
