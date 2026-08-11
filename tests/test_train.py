import copy

import numpy as np
import torch

from cqgt.model import CQGTModel
from cqgt.train import grow_model, predict, train_cqgt, train_stage


def _toy_dataset(T=6, n=4, F=5, n_mc=3, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 1, size=(n, n))
    np.fill_diagonal(W, 0)
    W_panel = np.stack([W] * T)
    X_std = rng.uniform(-1, 1, size=(T, n, F - 1)).astype(np.float32)
    shocked_mask_mc = rng.random((T, n_mc, n)) < 0.3
    y_mc = shocked_mask_mc.astype(int).copy()
    # give a little spillover-style signal so training has something to learn
    y_mc[rng.random((T, n_mc, n)) < 0.05] = 1
    macro_factors = rng.normal(size=(T, 3)).astype(np.float32)

    class FakeDS(dict):
        pass

    ds = FakeDS(W_panel=W_panel, X_std=X_std, shocked_mask_mc=shocked_mask_mc,
                y_mc=y_mc, macro_factors=macro_factors,
                train_t=np.arange(0, 4), val_t=np.arange(4, 5), test_t=np.arange(5, 6))
    return ds


def test_grow_model_copies_shared_params_exactly():
    n, F = 4, 5
    edges = [(0, 1), (1, 2), (2, 3)]
    m1 = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=1, n_macro=3)
    with torch.no_grad():
        m1.embed.weight.fill_(0.42)
    m2 = grow_model(m1, 2, n, F, edges)
    torch.testing.assert_close(m2.embed.weight, m1.embed.weight)
    torch.testing.assert_close(m2.tau0_raw, m1.tau0_raw)
    torch.testing.assert_close(m2.head[0].weight, m1.head[0].weight)


def test_grow_model_preserves_trained_layer_and_inits_new_layer_nearby():
    n, F = 4, 5
    edges = [(0, 1), (1, 2), (2, 3)]
    m1 = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=1, n_macro=3)
    with torch.no_grad():
        m1.phi[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    m2 = grow_model(m1, 2, n, F, edges, growth_std=1e-6)
    torch.testing.assert_close(m2.phi[0], m1.phi[0])  # trained layer preserved exactly
    # new layer initialized very close to the previous (last) layer given tiny growth_std
    torch.testing.assert_close(m2.phi[1], m1.phi[0], atol=1e-3, rtol=0)


def test_grow_model_from_scratch_when_prev_is_none():
    n, F = 4, 5
    edges = [(0, 1)]
    m = grow_model(None, 1, n, F, edges)
    assert m.n_layers == 1


def _compute_loss_terms(model, ds, train_t, n_mc, pos_weight):
    """Returns list of per-timestep BCE loss tensors (still attached to the
    graph), matching what both gradient paths below build from."""
    import cqgt.train as train_mod
    macro_factors = torch.tensor(ds["macro_factors"], dtype=torch.float32)
    losses = []
    for t in train_t:
        p = train_mod._batch_forward(model, ds, t, macro_factors[t], n_mc=n_mc)
        y = torch.tensor(ds["y_mc"][t, :n_mc], dtype=torch.float32)
        w = torch.where(y > 0, torch.tensor(pos_weight), torch.tensor(1.0))
        losses.append(torch.nn.functional.binary_cross_entropy(p, y, weight=w))
    return losses


def test_gradient_accumulation_matches_single_backward_over_stacked_mean():
    """The Phase 2 performance fix switched train_stage from one backward()
    over torch.stack(losses).mean() (holding all len(train_t) timesteps'
    graphs in memory at once -- what caused the swap-thrashing GATE 2 run)
    to a backward() per timestep on loss_t/n_t, accumulating into .grad.
    These must be mathematically identical (no shared state like BatchNorm
    exists between timesteps in CQGTModel, so linearity of differentiation
    guarantees sum-of-grads == grad-of-sum) -- verified here numerically to
    ~1e-6 on a tiny case rather than assumed, per explicit instruction:
    if this ever fails, every result downstream of the fix is suspect."""
    ds = _toy_dataset(T=5, n=3, F=4, n_mc=2, seed=1)
    edges = [(0, 1), (1, 2), (2, 0)]
    train_t = ds["train_t"][:3]  # 2-3 timesteps, per instruction
    n_mc = 2
    pos_weight = 1.7  # arbitrary fixed value; irrelevant to the equivalence claim

    base_model = CQGTModel(n_qubits=3, n_features=4, edges=edges, n_layers=2, n_macro=3)

    # Path A: single backward() over the stacked mean (the original design).
    model_a = copy.deepcopy(base_model)
    losses_a = _compute_loss_terms(model_a, ds, train_t, n_mc, pos_weight)
    torch.stack(losses_a).mean().backward()
    grads_a = {name: p.grad.clone() for name, p in model_a.named_parameters() if p.grad is not None}

    # Path B: backward() per timestep on loss_t / n_t, accumulated (the fix).
    model_b = copy.deepcopy(base_model)
    n_t = len(train_t)
    for t in train_t:
        losses_t = _compute_loss_terms(model_b, ds, [t], n_mc, pos_weight)
        (losses_t[0] / n_t).backward()
    grads_b = {name: p.grad.clone() for name, p in model_b.named_parameters() if p.grad is not None}

    assert set(grads_a.keys()) == set(grads_b.keys())
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], atol=1e-6, rtol=1e-5,
                                    msg=f"gradient mismatch for {name}")


def test_train_stage_runs_and_loss_history_has_expected_length():
    ds = _toy_dataset()
    n, F = 4, 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    model = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=1, n_macro=3)
    model, history = train_stage(model, ds, epochs=3, n_mc_train=2)
    assert len(history) == 3
    assert all(np.isfinite(h) for h in history)


def test_train_cqgt_layer_schedule_grows_and_returns_full_history():
    ds = _toy_dataset()
    n, F = 4, 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    model, full_history, per_stage = train_cqgt(
        ds, n_qubits=n, n_features=F, edges=edges,
        layer_schedule=(1, 2), epochs_per_stage=2, n_mc_train=2)
    assert model.n_layers == 2
    assert len(full_history) == 4
    assert set(per_stage.keys()) == {1, 2}

    y, p, mask = predict(model, ds, ds["val_t"])
    assert y.shape == p.shape == mask.shape
    assert ((p >= 0) & (p <= 1)).all()
