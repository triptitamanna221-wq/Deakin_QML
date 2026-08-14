import torch

from baselines.vqc import GenericVQC


def test_forward_shape_and_range():
    n, F = 6, 7
    model = GenericVQC(n_qubits=n, n_features=F, n_layers=2)
    x = torch.randn(4, n, F)
    p = model(x, W_t=None, macro_t=None)
    assert p.shape == (4, n)
    assert (p >= 0).all() and (p <= 1).all()


def test_gradients_flow_and_are_finite():
    n, F = 6, 7
    model = GenericVQC(n_qubits=n, n_features=F, n_layers=2)
    x = torch.randn(4, n, F)
    y = torch.randint(0, 2, (4, n)).float()
    p = model(x, None, None)
    loss = torch.nn.functional.binary_cross_entropy(p, y)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(param.grad).all()


def test_single_qubit_no_entangling_edges_does_not_crash():
    model = GenericVQC(n_qubits=1, n_features=3, n_layers=1)
    x = torch.randn(2, 1, 3)
    p = model(x)
    assert p.shape == (2, 1)
