import numpy as np

def aggregate_risk(p_hat):
    return p_hat.mean()

def perturbative_screen(grad_R_wrt_W, W, alpha):
    """grad_R_wrt_W: dR/dW_ij via autograd/finite-diff, same shape as W."""
    return alpha * grad_R_wrt_W * W

def top_k_edges(screened_scores, edges, k):
    vals = [screened_scores[i, j] for (i, j) in edges]
    order = np.argsort(vals)[::-1][:k]
    return [edges[idx] for idx in order]

def exact_counterfactual(model_predict_fn, W, X, edges_to_test, alpha, beta, gamma,
                          macro_loadings, macro_factors, f_weights):
    from cqgt.hamiltonian import build_hamiltonian_diag_and_laplacian
    R0 = aggregate_risk(model_predict_fn(W, X))
    results = {}
    for (i, j) in edges_to_test:
        W_cf = W.copy()
        W_cf[i, j] = 0.0
        R_cf = aggregate_risk(model_predict_fn(W_cf, X))
        results[(i, j)] = R0 - R_cf
    return dict(sorted(results.items(), key=lambda kv: -kv[1]))
