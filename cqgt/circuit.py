import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

def amplitude_embed(circuit, x_scalar, qubit):
    theta = np.arctan(x_scalar)
    circuit.ry(2 * theta, qubit)

def ctqw_layer(circuit, edges, tau0_param):
    # Trotterized approx of e^{-i L tau0}: RZZ on edges scaled by tau0
    for (i, j) in edges:
        circuit.rzz(tau0_param, i, j)

def variational_layer(circuit, edges, n_qubits, layer_idx):
    ry_params = [Parameter(f"phi_{layer_idx}_{i}") for i in range(n_qubits)]
    for i, p in enumerate(ry_params):
        circuit.ry(p, i)
    zz_params = [Parameter(f"theta_{layer_idx}_{e}") for e in range(len(edges))]
    for p, (i, j) in zip(zz_params, edges):
        circuit.rzz(p, i, j)
    return ry_params, zz_params

def hamiltonian_layer(circuit, h_diag_part, layer_idx):
    delta = Parameter(f"delta_{layer_idx}")
    for i, h_i in enumerate(h_diag_part):
        circuit.rz(2 * delta * h_i, i)
    return delta

def build_cqgt_circuit(n_qubits, edges, x_scalars, h_diag_part, n_layers=3):
    qc = QuantumCircuit(n_qubits)
    for i, x in enumerate(x_scalars):
        amplitude_embed(qc, x, i)
    tau0 = Parameter("tau0")
    ctqw_layer(qc, edges, tau0)
    all_params = {"tau0": tau0}
    for l in range(n_layers):
        ry_p, zz_p = variational_layer(qc, edges, n_qubits, l)
        delta = hamiltonian_layer(qc, h_diag_part, l)
        all_params[f"layer_{l}"] = (ry_p, zz_p, delta)
    return qc, all_params

def expectation_z(qc, bound_params, n_qubits):
    bound = qc.assign_parameters(bound_params)
    sv = Statevector.from_instruction(bound)
    return np.array([sv.expectation_value(("Z" + "I" * (n_qubits - 1 - i)).rjust(n_qubits, "I")[::-1] if False else _z_string(n_qubits, i)) for i in range(n_qubits)]).real

def _z_string(n, i):
    return "".join("Z" if k == i else "I" for k in range(n))[::-1]
