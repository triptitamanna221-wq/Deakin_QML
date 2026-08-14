import numpy as np
import torch

from cqgt.model import CQGTModel


def _toy_setup(n=4, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 1, size=(n, n))
    np.fill_diagonal(W, 0)
    edges = list(map(tuple, np.argwhere(W > 0)))
    return W, edges


def test_forward_output_shape_and_range():
    n, F = 4, 5
    W, edges = _toy_setup(n, F)
    model = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=2, n_macro=3)
    x = torch.randn(3, n, F)  # batch of 3 scenarios
    macro_t = torch.randn(3)
    p = model(x, W, macro_t)
    assert p.shape == (3, n)
    assert (p >= 0).all() and (p <= 1).all()


def test_forward_is_deterministic_given_fixed_params_and_input():
    n, F = 4, 5
    W, edges = _toy_setup(n, F)
    model = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=2, n_macro=3)
    x = torch.randn(2, n, F)
    macro_t = torch.randn(3)
    p1 = model(x, W, macro_t)
    p2 = model(x, W, macro_t)
    torch.testing.assert_close(p1, p2)


def test_gradients_flow_to_every_parameter():
    n, F = 4, 5
    W, edges = _toy_setup(n, F)
    model = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=2, n_macro=3)
    x = torch.randn(3, n, F)
    macro_t = torch.randn(3)
    y = torch.randint(0, 2, (3, n)).float()
    p = model(x, W, macro_t)
    loss = torch.nn.functional.binary_cross_entropy(p, y)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_n_parameters_matches_state_dict_scalar_count():
    n, F = 4, 5
    W, edges = _toy_setup(n, F)
    model = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=3, n_macro=3)
    expected = sum(p.numel() for p in model.state_dict().values())
    assert model.n_parameters() == expected


def test_no_hamiltonian_ablation_runs_and_differs():
    n, F = 4, 5
    W, edges = _toy_setup(n, F)
    torch.manual_seed(0)
    full = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=2, n_macro=3)
    torch.manual_seed(0)
    ablated = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=2, n_macro=3,
                         use_hamiltonian=False)
    x = torch.randn(3, n, F)
    macro_t = torch.randn(3)
    p_full = full(x, W, macro_t)
    p_ablated = ablated(x, W, macro_t)
    assert p_ablated.shape == (3, n)
    assert not torch.allclose(p_full, p_ablated)
    # gradient should NOT reach the disabled Hamiltonian's own coefficients
    # or its inputs (f_psi, macro_loadings) -- disclosed as dead-but-counted
    # parameters for this ablation, not a bug.
    loss = torch.nn.functional.binary_cross_entropy(
        p_ablated, torch.randint(0, 2, (3, n)).float())
    loss.backward()
    assert ablated.alpha_raw.grad is None or ablated.alpha_raw.grad.item() == 0.0
    assert ablated.phi.grad is not None  # the RY/RZZ variational path still trains


def test_classical_attention_ablation_runs_and_differs():
    n, F = 4, 5
    W, edges = _toy_setup(n, F)
    torch.manual_seed(0)
    full = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=2, n_macro=3)
    torch.manual_seed(0)
    ablated = CQGTModel(n_qubits=n, n_features=F, edges=edges, n_layers=2, n_macro=3,
                         use_quantum_attention=False)
    assert full.n_parameters() == ablated.n_parameters()  # mechanism-only ablation
    x = torch.randn(3, n, F)
    macro_t = torch.randn(3)
    p_full = full(x, W, macro_t)
    p_ablated = ablated(x, W, macro_t)
    assert p_ablated.shape == (3, n)
    assert not torch.allclose(p_full, p_ablated)
    loss = torch.nn.functional.binary_cross_entropy(
        p_ablated, torch.randint(0, 2, (3, n)).float())
    loss.backward()
    assert ablated.query_params.grad is not None
    assert torch.isfinite(ablated.query_params.grad).all()


def test_zero_edges_does_not_crash():
    """An isolated-node edge case: n_edges=0 must not break shape handling."""
    n, F = 3, 4
    W = np.zeros((n, n))
    model = CQGTModel(n_qubits=n, n_features=F, edges=[], n_layers=1, n_macro=2)
    x = torch.randn(2, n, F)
    macro_t = torch.randn(2)
    p = model(x, W, macro_t)
    assert p.shape == (2, n)
