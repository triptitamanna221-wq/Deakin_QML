"""Mandatory GATE 0 test: cqgt/qsim.py (PyTorch statevector simulator) must
match qiskit.quantum_info.Statevector exactly, since it replaces Qiskit in
the training loop and correctness here cannot be taken on faith."""
import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from cqgt import qsim

ATOL = 1e-6


def _random_circuit_case(seed, n, batch):
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]
    embed_theta = rng.uniform(-1.5, 1.5, size=(batch, n))
    hop_theta = rng.uniform(-1.0, 1.0, size=len(edges))
    zz_theta = rng.uniform(-1.0, 1.0, size=len(edges))
    rz_theta = rng.uniform(-1.0, 1.0, size=n)
    return edges, embed_theta, hop_theta, zz_theta, rz_theta


def _qsim_run(n, batch, edges, embed_theta, hop_theta, zz_theta, rz_theta):
    state = qsim.zero_state(batch, n)
    for i in range(n):
        state = qsim.ry(state, torch.tensor(embed_theta[:, i]), i, n)
    for e, (i, j) in enumerate(edges):
        state = qsim.hopping(state, torch.tensor(hop_theta[e]), i, j, n)
    for e, (i, j) in enumerate(edges):
        state = qsim.rzz(state, torch.tensor(zz_theta[e]), i, j, n)
    for i in range(n):
        state = qsim.rz(state, torch.tensor(rz_theta[i]), i, n)
    return state


def _qiskit_run(n, edges, embed_theta_row, hop_theta, zz_theta, rz_theta):
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(float(embed_theta_row[i]), i)
    for e, (i, j) in enumerate(edges):
        qc.rxx(float(hop_theta[e]), i, j)
        qc.ryy(float(hop_theta[e]), i, j)
    for e, (i, j) in enumerate(edges):
        qc.rzz(float(zz_theta[e]), i, j)
    for i in range(n):
        qc.rz(float(rz_theta[i]), i)
    return Statevector.from_instruction(qc)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_qsim_matches_qiskit_statevector_and_z(seed, n):
    batch = 3
    edges, embed_theta, hop_theta, zz_theta, rz_theta = _random_circuit_case(seed, n, batch)
    state = _qsim_run(n, batch, edges, embed_theta, hop_theta, zz_theta, rz_theta)
    qsim_amps = state.reshape(batch, -1).detach().numpy()
    qsim_z = qsim.expect_all_z(state, n).detach().numpy()

    for b in range(batch):
        sv = _qiskit_run(n, edges, embed_theta[b], hop_theta, zz_theta, rz_theta)
        np.testing.assert_allclose(qsim_amps[b], sv.data, atol=ATOL)

        ref_z = np.array([
            sv.expectation_value(SparsePauliOp("".join("Z" if k == i else "I" for k in range(n))[::-1])).real
            for i in range(n)
        ])
        np.testing.assert_allclose(qsim_z[b], ref_z, atol=ATOL)


def test_qsim_is_differentiable():
    n = 3
    theta = torch.tensor([0.3, -0.2, 0.7], dtype=torch.float64, requires_grad=True)
    hop = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    state = qsim.zero_state(1, n)
    for i in range(n):
        state = qsim.ry(state, theta[i], i, n)
    state = qsim.hopping(state, hop, 0, 1, n)
    loss = qsim.expect_all_z(state, n).sum()
    loss.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    assert torch.abs(theta.grad).sum() > 0


def test_hopping_conserves_total_z():
    """RXX(theta).RYY(theta) generates X_iX_j + Y_iY_j, which commutes with
    total Z -- sum(<Z_i>) must be invariant under a pure hopping layer. This
    is what distinguishes it from the old, broken RZZ 'CTQW' (which does move
    amplitude only in phase and never actually transfers excitation)."""
    n = 4
    state = qsim.zero_state(1, n)
    state = qsim.ry(state, torch.tensor(0.9), 0, n)
    z_before = qsim.expect_all_z(state, n).sum()
    state = qsim.hopping(state, torch.tensor(1.234), 0, 1, n)
    z_after = qsim.expect_all_z(state, n).sum()
    assert torch.allclose(z_before, z_after, atol=ATOL)
    # and it must actually move amplitude between the two qubits it acts on
    state2 = qsim.zero_state(1, n)
    state2 = qsim.ry(state2, torch.tensor(0.9), 0, n)
    z0_pre = qsim.expect_z(state2, 0, n)
    state2 = qsim.hopping(state2, torch.tensor(1.234), 0, 1, n)
    z0_post = qsim.expect_z(state2, 0, n)
    z1_post = qsim.expect_z(state2, 1, n)
    assert not torch.allclose(z0_pre, z0_post, atol=1e-3)
    assert not torch.allclose(z1_post, torch.tensor([1.0], dtype=torch.float64), atol=1e-3)
