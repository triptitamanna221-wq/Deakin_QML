"""Generic hardware-efficient VQC baseline (BRIEF.md Sec 2.4): same qubit
count as CQGT, same input embedding, but NO exposure-network topology, NO
Hamiltonian (alpha/beta/gamma), NO macro coupling, NO attention -- just a
fixed nearest-neighbor-chain entangling ansatz, independent of W. This is
what isolates "does the graph-informed, Hamiltonian-driven architecture
matter" as opposed to "is a quantum circuit of this size just generically
useful." Built on cqgt/qsim.py, never Qiskit, for the same exact-backprop
reason as CQGTModel.
"""
import torch
import torch.nn as nn

from cqgt import qsim


class GenericVQC(nn.Module):
    def __init__(self, n_qubits, n_features, n_layers=3, hidden_dim=8):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.n_layers = n_layers
        # Fixed linear-chain entangling pattern -- hardware-efficient in the
        # standard sense (nearest-neighbor connectivity), deliberately blind
        # to the real exposure network's actual topology.
        self.chain_edges = [(i, i + 1) for i in range(n_qubits - 1)]

        self.embed = nn.Linear(n_features, 1)
        self.ry_params = nn.Parameter(torch.zeros(n_layers, n_qubits))
        self.rz_params = nn.Parameter(torch.zeros(n_layers, n_qubits))
        self.zz_params = nn.Parameter(torch.zeros(n_layers, max(len(self.chain_edges), 1)))
        self.head = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, W_t=None, macro_t=None):
        """W_t, macro_t accepted but ignored -- interface-compatible with
        cqgt.train.train_stage / CQGTModel.forward, since this baseline is
        defined by NOT using them."""
        B, n = x.shape[0], self.n_qubits
        embed_theta = torch.arctan(self.embed(x).squeeze(-1))  # (B, n)
        state = qsim.zero_state(B, n)
        for i in range(n):
            state = qsim.ry(state, 2 * embed_theta[:, i], i, n)

        for l in range(self.n_layers):
            for i in range(n):
                state = qsim.ry(state, self.ry_params[l, i], i, n)
                state = qsim.rz(state, self.rz_params[l, i], i, n)
            for e, (i, j) in enumerate(self.chain_edges):
                state = qsim.rzz(state, self.zz_params[l, e], i, j, n)

        r = qsim.expect_all_z(state, n).float()  # (B, n)
        p_hat = torch.sigmoid(self.head(r.unsqueeze(-1)).squeeze(-1))  # (B, n)
        return p_hat
