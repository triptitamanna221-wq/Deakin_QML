import numpy as np

from cqgt.counterfactual import exact_counterfactual, top_k_edges


def test_exact_counterfactual_new_signature_no_unused_args():
    """Old signature took alpha, beta, gamma, macro_loadings, macro_factors,
    f_weights and ignored all of them -- model_predict_fn already closes
    over model state, so those were dead parameters."""
    W = np.array([[0, 1.0, 0], [0, 0, 2.0], [0.5, 0, 0]])
    X = np.zeros((3, 2))

    def predict(W, X):
        # toy risk: proportional to total exposure
        return np.full(3, W.sum())

    edges = [(0, 1), (1, 2), (2, 0)]
    result = exact_counterfactual(predict, W, X, edges)
    assert set(result.keys()) == set(edges)
    # removing the largest edge should yield the largest risk drop
    ranked = top_k_edges(result, edges, k=1)
    assert ranked == [(1, 2)]
