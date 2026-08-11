"""CQGT training loop: layer-by-layer growth L=1 -> 2 -> 3 (BRIEF.md Sec
2.3), Adam + cosine annealing + grad clipping + class-weighted BCE (Sec
2.7's training config), trained on the same paired-dataset scenario
examples (t, m) used throughout Phase 1/GATE 1 so results are directly
comparable to every baseline built against the same call to
cqgt.data.pipeline.build_paired_dataset.
"""
import numpy as np
import torch

from cqgt.data.pipeline import features_with_shock_indicator
from cqgt.model import CQGTModel
from cqgt.seeding import set_seed

GROWTH_STD = np.pi / 10
_SHARED_PREFIXES = ("embed", "tau0_raw", "alpha_raw", "beta_raw", "gamma_raw", "f_psi",
                    "macro_loadings", "query_params", "key_params", "attn_lambda", "head")


def grow_model(prev_model, n_layers_new, n_qubits, n_features, edges, n_macro=3, hidden_dim=8,
               growth_std=GROWTH_STD):
    """Build a CQGTModel with n_layers_new layers. If prev_model is given,
    copy every non-layer-specific parameter directly, copy already-trained
    layers as-is, and initialize any newly-added layer(s) as
    N(theta*_prev_layer, growth_std^2) -- centered at the most recently
    trained layer's parameters, per BRIEF.md Sec 2.3."""
    new_model = CQGTModel(n_qubits, n_features, edges, n_layers=n_layers_new,
                           n_macro=n_macro, hidden_dim=hidden_dim)
    if prev_model is None:
        return new_model

    prev_state, new_state = prev_model.state_dict(), new_model.state_dict()
    for k in new_state:
        if k.startswith(_SHARED_PREFIXES):
            new_state[k] = prev_state[k].clone()

    L_prev = prev_model.n_layers
    with torch.no_grad():
        new_state["phi"][:L_prev] = prev_state["phi"]
        new_state["zz_params"][:L_prev] = prev_state["zz_params"]
        new_state["delta"][:L_prev] = prev_state["delta"]
        if n_layers_new > L_prev:
            last_phi, last_zz, last_delta = prev_state["phi"][-1], prev_state["zz_params"][-1], prev_state["delta"][-1]
            for l in range(L_prev, n_layers_new):
                new_state["phi"][l] = last_phi + torch.randn_like(last_phi) * growth_std
                new_state["zz_params"][l] = last_zz + torch.randn_like(last_zz) * growth_std
                new_state["delta"][l] = last_delta + torch.randn(()) * growth_std
    new_model.load_state_dict(new_state)
    return new_model


def _batch_forward(model, ds, t, macro_t, n_mc=None):
    n_mc = ds["y_mc"].shape[1] if n_mc is None else n_mc
    xs = np.stack([features_with_shock_indicator(ds, t, m) for m in range(n_mc)])
    x = torch.tensor(xs, dtype=torch.float32)
    return model(x, ds["W_panel"][t], macro_t)


def train_stage(model, ds, epochs, lr=0.01, grad_clip=1.0, seed=0, n_mc_train=None):
    """n_mc_train: use only the first n_mc_train of the full M MC draws per
    training step (compute-budget knob for a full statevector simulator on
    CPU-only hardware -- see NOTES.md's Phase 2 performance discussion).
    Evaluation (cqgt.train.predict) always uses the full paired M draws, so
    the held-out comparison against baselines stays fair; only the gradient
    estimate per step is coarser at smaller n_mc_train."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    train_t = ds["train_t"]
    n_mc = ds["y_mc"].shape[1] if n_mc_train is None else n_mc_train
    y_train_flat = ds["y_mc"][train_t, :n_mc].reshape(-1)
    pos_frac = y_train_flat.mean()
    pos_weight = (1 - pos_frac) / max(pos_frac, 1e-6)
    macro_factors = torch.tensor(ds["macro_factors"], dtype=torch.float32)

    # Gradient accumulation via a backward() call PER timestep, not one
    # backward() over a graph spanning all len(train_t) timesteps at once.
    # The latter (the original implementation) held ~72 timesteps' worth of
    # computation graph in memory simultaneously before freeing any of it,
    # which pushed this machine into heavy swap (measured: 17.5/18GB swap
    # used, >1e9 page faults on the process) during an actual GATE 2 run --
    # a real bug caught by that run stalling for 2+ hours against a ~70min
    # profiled estimate, not just "compute is slow." Per-timestep backward
    # frees each timestep's graph immediately, bounding peak memory to one
    # timestep instead of len(train_t) of them.
    loss_history = []
    n_t = len(train_t)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        total_loss = 0.0
        for t in train_t:
            p = _batch_forward(model, ds, t, macro_factors[t], n_mc=n_mc)
            y = torch.tensor(ds["y_mc"][t, :n_mc], dtype=torch.float32)
            w = torch.where(y > 0, torch.tensor(pos_weight), torch.tensor(1.0))
            loss_t = torch.nn.functional.binary_cross_entropy(p, y, weight=w) / n_t
            loss_t.backward()
            total_loss += loss_t.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        loss_history.append(total_loss)
    return model, loss_history


def train_cqgt(ds, n_qubits, n_features, edges, layer_schedule=(1, 2, 3),
               epochs_per_stage=30, lr=0.01, seed=0, n_mc_train=None):
    """Full layer-by-layer growth training run. Returns (model, full_loss_history,
    per_stage_histories)."""
    model = None
    full_history = []
    per_stage = {}
    for n_layers in layer_schedule:
        model = grow_model(model, n_layers, n_qubits, n_features, edges)
        model, hist = train_stage(model, ds, epochs_per_stage, lr=lr, seed=seed, n_mc_train=n_mc_train)
        per_stage[n_layers] = hist
        full_history.extend(hist)
    return model, full_history, per_stage


def predict(model, ds, t_indices):
    """Returns (y_true, p_hat, shocked_mask) flattened over (t, m) for the
    given timestep indices, for evaluation."""
    model.eval()
    macro_factors = torch.tensor(ds["macro_factors"], dtype=torch.float32)
    y_all, p_all, mask_all = [], [], []
    with torch.no_grad():
        for t in t_indices:
            p = _batch_forward(model, ds, t, macro_factors[t])
            y_all.append(ds["y_mc"][t])
            p_all.append(p.numpy())
            mask_all.append(ds["shocked_mask_mc"][t])
    return (np.concatenate(y_all).reshape(-1), np.concatenate(p_all).reshape(-1),
            np.concatenate(mask_all).reshape(-1))
