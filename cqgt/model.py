"""CQGT: Causal Quantum Graph Transformer forward pass (BRIEF.md Sec 2.3),
built entirely on cqgt/qsim.py (never Qiskit) so it trains by exact
backprop, per the project's hard constraint.

Forward pass per timestep t (batch = several MC shock-draw scenarios that
share the same network W_t and macro state, differing only in their
per-institution feature vectors -- see cqgt/data/pipeline.py):
  1. Embed:        RY(2 arctan(w^T x_i)) per qubit.
  2. CTQW encoder:  one hopping (RXX.RYY) application per exposure edge,
                    angle = tau0 * Ltilde_ij (tau0 learnable, >=0).
  3. L variational layers, each: RY(phi_i) per qubit -> RZZ(theta_ij) on
     exposure edges only -> e^{-i H_t delta_l} where H_t's entangling part
     is the same hopping term scaled by alpha, and its diagonal part is
     RZ(2 delta_l h_i), h_i = beta f_psi(x_i) + gamma sum_k lambda_k^t beta_ik.
  4. Measure <Z_i> -> r.
  5. Quantum fidelity attention on n_a=2 ancilla qubits (learned per-
     institution query/key parameters, independent of x): x_tilde_i =
     sum_j softmax(lambda F)_ij * (r_j * x_j).
  6. MLP head on [r_i ; x_tilde_i] -> p_hat_i.

The exposure network's edge SET is assumed fixed across a panel (verified
in cqgt/data/pipeline.py's temporal construction: interpolation/AR(1) noise
only rescales magnitudes, never zeroes out an anchor edge) and is supplied
once at construction time -- this is what "topology-matched ansatz" means:
RZZ/hopping parameters exist only on real exposure edges, not all N(N-1)
pairs.
"""
import numpy as np
import torch
import torch.nn as nn

from cqgt import qsim
from cqgt.hamiltonian import normalized_laplacian

N_ANCILLA = 2
QK_DIM = 2 * N_ANCILLA - 1  # RY angles + RZZ angles for build_qk_state-equivalent


class CQGTModel(nn.Module):
    def __init__(self, n_qubits, n_features, edges, n_layers=3, n_macro=3, hidden_dim=8,
                 use_hamiltonian=True, use_quantum_attention=True):
        """use_hamiltonian=False and use_quantum_attention=False are the
        `no_hamiltonian` / `classical_attention` ablations (BRIEF.md Sec
        2.5). The third ablation, `random_edges`, needs no model change --
        it is implemented by constructing this same model with a randomly
        generated (density-matched) `edges` list instead of the real
        exposure topology; see experiments/run_ablation.py."""
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.edges = list(edges)  # [(i, j), ...], i != j, real exposure edges only
        self.n_edges = len(self.edges)
        self.n_layers = n_layers
        self.n_macro = n_macro
        self.use_hamiltonian = use_hamiltonian
        self.use_quantum_attention = use_quantum_attention

        self.embed = nn.Linear(n_features, 1)

        self.tau0_raw = nn.Parameter(torch.tensor(0.5413))    # softplus -> ~1.0
        self.alpha_raw = nn.Parameter(torch.tensor(0.5413))
        self.beta_raw = nn.Parameter(torch.tensor(0.5413))
        self.gamma_raw = nn.Parameter(torch.tensor(0.5413))

        self.f_psi = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        # small-noise, not zero, init: with macro_loadings=0 exactly, gamma's
        # gradient (d h/d gamma = macro_loadings @ macro_t) is structurally
        # zero at step 0, decoupling it from the loss until macro_loadings
        # first moves -- avoid that by breaking symmetry here directly.
        self.macro_loadings = nn.Parameter(torch.randn(n_qubits, n_macro) * 0.01)

        self.phi = nn.Parameter(torch.zeros(n_layers, n_qubits))
        self.zz_params = nn.Parameter(torch.zeros(n_layers, max(self.n_edges, 1)))
        self.delta = nn.Parameter(torch.full((n_layers,), 0.1))

        self.query_params = nn.Parameter(torch.randn(n_qubits, QK_DIM) * 0.1)
        self.key_params = nn.Parameter(torch.randn(n_qubits, QK_DIM) * 0.1)
        self.attn_lambda = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Sequential(
            nn.Linear(1 + n_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _edge_laplacian_weights(self, W_t):
        """W_t: (n,n) numpy array for this snapshot. Returns a (n_edges,)
        torch tensor of the normalized Laplacian's edge entries -- fixed
        data derived from W_t, not a model parameter, so no grad needed."""
        L_tilde = normalized_laplacian(W_t)
        vals = np.array([L_tilde[i, j] for (i, j) in self.edges], dtype=np.float32)
        return torch.from_numpy(vals)

    def _attention_weights(self):
        """Quantum fidelity attention F_ij = |<q_i|k_j>|^2 via a 2-ancilla-
        qubit qsim circuit, batched over the n_qubits institutions (batch
        dimension here = institutions, not scenarios -- this sub-circuit
        does not depend on x at all, only on the learned query/key params).

        classical_attention ablation (use_quantum_attention=False): the
        SAME query_params/key_params tensors are reinterpreted as plain
        real-valued embedding vectors and scored by dot product instead of
        being used as quantum circuit angles -- this isolates the quantum
        vs. classical attention *mechanism* as the only difference, since
        parameter count and shapes are otherwise identical."""
        n = self.n_qubits
        if not self.use_quantum_attention:
            scores = self.query_params @ self.key_params.T  # (n, n) real dot product
            return torch.softmax(self.attn_lambda * scores, dim=1)
        sq = qsim.zero_state(n, N_ANCILLA)
        for a in range(N_ANCILLA):
            sq = qsim.ry(sq, self.query_params[:, a], a, N_ANCILLA)
        for a in range(N_ANCILLA - 1):
            sq = qsim.rzz(sq, self.query_params[:, N_ANCILLA + a], a, a + 1, N_ANCILLA)

        sk = qsim.zero_state(n, N_ANCILLA)
        for a in range(N_ANCILLA):
            sk = qsim.ry(sk, self.key_params[:, a], a, N_ANCILLA)
        for a in range(N_ANCILLA - 1):
            sk = qsim.rzz(sk, self.key_params[:, N_ANCILLA + a], a, a + 1, N_ANCILLA)

        psi_q = sq.reshape(n, -1)  # (n, 2^n_a) complex
        psi_k = sk.reshape(n, -1)
        inner = psi_q.conj() @ psi_k.T  # (n, n) complex
        F = inner.abs() ** 2
        A = torch.softmax(self.attn_lambda * F, dim=1)  # exp(lam F)/sum == softmax
        return A

    def circuit_expectation_z(self, x, W_t, macro_t):
        """The quantum circuit's raw <Z_i> readout (BRIEF.md Sec 2.3 steps
        1-4), before quantum/classical attention and the MLP head. Exposed
        as its own method (not just inlined in forward()) because it is the
        object of the 1-WL separation claim (BRIEF.md Sec 3, III-E) --
        experiments/wl_separation.py compares THIS quantity, not the
        post-attention/post-MLP p_hat, since the claim is about the
        circuit's expressivity specifically, not the classical head that
        follows it. Returns r: (B, n_qubits)."""
        B, n = x.shape[0], self.n_qubits
        edge_L = self._edge_laplacian_weights(W_t)

        embed_theta = torch.arctan(self.embed(x).squeeze(-1))  # (B, n)
        state = qsim.zero_state(B, n)
        for i in range(n):
            state = qsim.ry(state, 2 * embed_theta[:, i], i, n)

        tau0 = torch.nn.functional.softplus(self.tau0_raw)
        for e, (i, j) in enumerate(self.edges):
            state = qsim.hopping(state, tau0 * edge_L[e], i, j, n)

        if self.use_hamiltonian:
            alpha = torch.nn.functional.softplus(self.alpha_raw)
            beta = torch.nn.functional.softplus(self.beta_raw)
            gamma = torch.nn.functional.softplus(self.gamma_raw)
            f_psi_val = self.f_psi(x).squeeze(-1)  # (B, n)
            macro_term = self.macro_loadings @ macro_t  # (n,)
            h = beta * f_psi_val + gamma * macro_term[None, :]  # (B, n)

        for l in range(self.n_layers):
            for i in range(n):
                state = qsim.ry(state, self.phi[l, i], i, n)
            for e, (i, j) in enumerate(self.edges):
                state = qsim.rzz(state, self.zz_params[l, e], i, j, n)
            # no_hamiltonian ablation: skip e^{-i H_t delta_l} entirely (both
            # the alpha-scaled hopping/entangling part and the beta/gamma
            # diagonal RZ part) -- the network-contagion + macro coupling
            # term that BRIEF.md Sec 1 flags as "the paper's central claim",
            # ablated to test whether it's actually earning its keep beyond
            # the CTQW encoder (step 2, still applied) and the per-layer RZZ.
            if self.use_hamiltonian:
                delta_l = self.delta[l]
                for e, (i, j) in enumerate(self.edges):
                    state = qsim.hopping(state, alpha * edge_L[e] * delta_l, i, j, n)
                for i in range(n):
                    state = qsim.rz(state, 2 * delta_l * h[:, i], i, n)

        # qsim runs in complex128 for numerical exactness (Phase 0's atol=1e-6
        # bar against Qiskit); downcast the real-valued outputs to float32
        # here so they interoperate with the classical nn.Linear layers,
        # which default to float32 weights.
        return qsim.expect_all_z(state, n).float()  # (B, n)

    def forward(self, x, W_t, macro_t):
        """x: (B, n_qubits, n_features) float32 -- B scenarios sharing W_t.
        W_t: (n_qubits, n_qubits) numpy array, this snapshot's real network.
        macro_t: (n_macro,) tensor, this snapshot's macro factor values.
        Returns p_hat: (B, n_qubits)."""
        r = self.circuit_expectation_z(x, W_t, macro_t)
        A = self._attention_weights().float()  # (n, n)
        h_attn = r.unsqueeze(-1) * x  # (B, n, F)
        x_tilde = torch.einsum("ij,bjf->bif", A, h_attn)  # (B, n, F)

        head_in = torch.cat([r.unsqueeze(-1), x_tilde], dim=-1)  # (B, n, 1+F)
        p_hat = torch.sigmoid(self.head(head_in).squeeze(-1))  # (B, n)
        return p_hat
