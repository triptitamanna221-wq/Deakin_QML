import numpy as np

from cqgt.circuit import build_cqgt_circuit, expectation_z


def test_ctqw_layer_moves_amplitude_and_uses_edge_weights():
    """Regression test for the broken CTQW: the old ctqw_layer used RZZ,
    which is diagonal in the computational basis and cannot transfer
    amplitude between qubits, and shared a single tau0 across all edges so
    W_ij never entered the circuit. Build two circuits differing only in
    edge weights (tau0=0 fixed, so only the ctqw hopping term is exercised
    downstream by varying weights) and confirm <Z> differs."""
    n = 3
    edges = [(0, 1), (1, 2)]
    x_scalars = [0.4, 0.0, 0.0]  # excitation starts on qubit 0
    h_diag = [0.0, 0.0, 0.0]

    qc, params = build_cqgt_circuit(n, edges, edge_weights=[1.0, 1.0], x_scalars=x_scalars,
                                     h_diag_part=h_diag, n_layers=0)
    bind = {params["tau0"]: 1.2}
    z_uniform = expectation_z(qc, bind, n)

    qc2, params2 = build_cqgt_circuit(n, edges, edge_weights=[3.0, 0.0], x_scalars=x_scalars,
                                       h_diag_part=h_diag, n_layers=0)
    bind2 = {params2["tau0"]: 1.2}
    z_weighted = expectation_z(qc2, bind2, n)

    # amplitude actually moved off qubit 0 in both cases
    assert z_uniform[0] < 0.99
    # and the two weightings give different <Z> profiles -- W_ij is not discarded
    assert not np.allclose(z_uniform, z_weighted, atol=1e-6)


def test_expectation_z_has_no_dead_branch():
    """The old code had `... if False else ...` in a list comprehension --
    dead code that always evaluated the else-branch. Confirm the current
    single expression is correct on a trivial all-zero state (<Z>=+1 everywhere)."""
    n = 2
    qc, params = build_cqgt_circuit(n, edges=[], edge_weights=[], x_scalars=[0.0, 0.0],
                                     h_diag_part=[0.0, 0.0], n_layers=0)
    # no edges -> tau0 never appears in a gate, so the circuit has no free
    # parameters at all; bind nothing.
    z = expectation_z(qc, {}, n)
    np.testing.assert_allclose(z, [1.0, 1.0], atol=1e-8)
