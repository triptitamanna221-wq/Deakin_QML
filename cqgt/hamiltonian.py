import numpy as np

def normalized_laplacian(W):
    d = W.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt

def risk_potential_diagonal(X, weights):
    """D(X_t): linear scalar potential per institution (stand-in for f_psi)."""
    return X @ weights  # shape (N,)

def macro_coupling_diagonal(macro_loadings, macro_factors):
    """C_t as a diagonal contribution; macro_loadings: (K,), macro_factors: (N,K)."""
    return macro_factors @ macro_loadings  # shape (N,)

def build_hamiltonian_diag_and_laplacian(W, X, f_weights, macro_loadings,
                                          macro_factors, alpha, beta, gamma):
    L = normalized_laplacian(W)
    d_term = risk_potential_diagonal(X, f_weights)
    c_term = macro_coupling_diagonal(macro_loadings, macro_factors)
    H_diag_part = beta * d_term + gamma * c_term  # single-qubit Z part
    H = alpha * L  # entangling part (off-diagonal structure)
    return H, H_diag_part
