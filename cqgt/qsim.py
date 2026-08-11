"""Minimal batched PyTorch statevector simulator for the CQGT circuit.

State layout: complex tensor of shape (batch, 2, 2, ..., 2) with N trailing
axes, one per qubit. This mirrors Qiskit's little-endian Statevector reshaped
in C order: qubit 0 is the fastest-varying (last) axis, qubit N-1 is the
first qubit-axis (right after the batch axis). Gate parameters may be a
python/torch scalar (shared across the batch) or a (batch,) tensor (e.g. a
per-sample embedding angle); either way everything here is built from plain
torch ops, so autograd flows through exactly -- no parameter-shift needed.
"""
import torch

CDTYPE = torch.complex128


def zero_state(batch, n_qubits, device=None):
    state = torch.zeros((batch,) + (2,) * n_qubits, dtype=CDTYPE, device=device)
    state.reshape(batch, -1)[:, 0] = 1.0
    return state


def _axis(q, n_qubits):
    return 1 + (n_qubits - 1 - q)


def _bexpand(theta, batch, device):
    theta = torch.as_tensor(theta, dtype=torch.float64, device=device)
    return theta.expand(batch) if theta.ndim == 0 else theta


def _apply_1q(state, U, q, n_qubits):
    ax = _axis(q, n_qubits)
    state = torch.movedim(state, ax, -1)
    state = torch.einsum("b...i,bji->b...j", state, U)
    return torch.movedim(state, -1, ax)


def _apply_2q(state, U, q1, q2, n_qubits):
    ax1, ax2 = _axis(q1, n_qubits), _axis(q2, n_qubits)
    state = torch.movedim(state, (ax1, ax2), (-2, -1))
    shape = state.shape
    flat = torch.einsum("b...i,bji->b...j", state.reshape(*shape[:-2], 4), U)
    return torch.movedim(flat.reshape(shape), (-2, -1), (ax1, ax2))


def ry(state, theta, q, n_qubits):
    b = state.shape[0]
    th = _bexpand(theta, b, state.device) / 2
    c, s = torch.cos(th).to(CDTYPE), torch.sin(th).to(CDTYPE)
    U = torch.zeros(b, 2, 2, dtype=CDTYPE, device=state.device)
    U[:, 0, 0], U[:, 0, 1] = c, -s
    U[:, 1, 0], U[:, 1, 1] = s, c
    return _apply_1q(state, U, q, n_qubits)


def rz(state, theta, q, n_qubits):
    b = state.shape[0]
    th = (_bexpand(theta, b, state.device) / 2).to(CDTYPE)
    U = torch.zeros(b, 2, 2, dtype=CDTYPE, device=state.device)
    U[:, 0, 0], U[:, 1, 1] = torch.exp(-1j * th), torch.exp(1j * th)
    return _apply_1q(state, U, q, n_qubits)


def rzz(state, theta, q1, q2, n_qubits):
    b = state.shape[0]
    th = (_bexpand(theta, b, state.device) / 2).to(CDTYPE)
    e_m, e_p = torch.exp(-1j * th), torch.exp(1j * th)
    U = torch.zeros(b, 4, 4, dtype=CDTYPE, device=state.device)
    diag = torch.stack([e_m, e_p, e_p, e_m], dim=-1)
    U[:, torch.arange(4), torch.arange(4)] = diag
    return _apply_2q(state, U, q1, q2, n_qubits)


def rxx(state, theta, q1, q2, n_qubits):
    b = state.shape[0]
    th = _bexpand(theta, b, state.device) / 2
    c = torch.cos(th).to(CDTYPE)
    m = (-1j * torch.sin(th)).to(CDTYPE)
    U = torch.zeros(b, 4, 4, dtype=CDTYPE, device=state.device)
    U[:, 0, 0] = U[:, 1, 1] = U[:, 2, 2] = U[:, 3, 3] = c
    U[:, 0, 3] = U[:, 1, 2] = U[:, 2, 1] = U[:, 3, 0] = m
    return _apply_2q(state, U, q1, q2, n_qubits)


def ryy(state, theta, q1, q2, n_qubits):
    b = state.shape[0]
    th = _bexpand(theta, b, state.device) / 2
    c = torch.cos(th).to(CDTYPE)
    p = (1j * torch.sin(th)).to(CDTYPE)
    U = torch.zeros(b, 4, 4, dtype=CDTYPE, device=state.device)
    U[:, 0, 0] = U[:, 1, 1] = U[:, 2, 2] = U[:, 3, 3] = c
    U[:, 0, 3] = U[:, 3, 0] = p
    U[:, 1, 2] = U[:, 2, 1] = -p
    return _apply_2q(state, U, q1, q2, n_qubits)


def hopping(state, theta, q1, q2, n_qubits):
    """RXX(theta) . RYY(theta): the CTQW hopping term (X_iX_j + Y_iY_j)/2.
    RXX and RYY commute, so applying them in either order is exact."""
    state = rxx(state, theta, q1, q2, n_qubits)
    return ryy(state, theta, q1, q2, n_qubits)


def expect_z(state, q, n_qubits):
    ax = _axis(q, n_qubits)
    s = torch.movedim(state, ax, -1)
    probs = (s.abs() ** 2).reshape(s.shape[0], -1, 2).sum(dim=1)
    return probs[:, 0] - probs[:, 1]


def expect_all_z(state, n_qubits):
    return torch.stack([expect_z(state, q, n_qubits) for q in range(n_qubits)], dim=-1)
