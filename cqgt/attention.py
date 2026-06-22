import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def build_qk_state(n_a, params):
    qc = QuantumCircuit(n_a)
    for i in range(n_a):
        qc.ry(params[i], i)
    for i in range(n_a - 1):
        qc.rzz(params[n_a + i], i, i + 1)
    return Statevector.from_instruction(qc)

def fidelity_attention(query_params, key_params, n_a, lam=1.0):
    N = len(query_params)
    F = np.zeros((N, N))
    states_q = [build_qk_state(n_a, query_params[i]) for i in range(N)]
    states_k = [build_qk_state(n_a, key_params[j]) for j in range(N)]
    for i in range(N):
        for j in range(N):
            F[i, j] = np.abs(states_q[i].inner(states_k[j])) ** 2
    A = np.exp(lam * F)
    A /= A.sum(axis=1, keepdims=True)
    return A, F

def swap_test_circuit(n_a):
    """Explicit destructive swap test, for hardware/noisy-sim runs only."""
    qc = QuantumCircuit(2 * n_a + 1, 1)
    anc = 2 * n_a
    qc.h(anc)
    for k in range(n_a):
        qc.cswap(anc, k, n_a + k)
    qc.h(anc)
    qc.measure(anc, 0)
    return qc
