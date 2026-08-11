import numpy as np
from sklearn.pipeline import Pipeline

from baselines.quantum_kernel import dequantized_rff_surrogate, QISKIT_ML_AVAILABLE


def test_rff_surrogate_is_a_single_tied_pipeline():
    """Old version returned an untied (rff, svm) tuple -- callers could fit
    one and predict with the other's state. Now it must be one fitted object."""
    pipe = dequantized_rff_surrogate(n_features=4, n_components=20)
    assert isinstance(pipe, Pipeline)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = (X[:, 0] > 0).astype(int)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (30,)


def test_quantum_kernel_import_is_guarded():
    """Module import must not hard-fail the whole package if
    qiskit-machine-learning is absent or its API has drifted."""
    import baselines.quantum_kernel as qk
    assert isinstance(QISKIT_ML_AVAILABLE, bool)
    if not QISKIT_ML_AVAILABLE:
        import pytest
        with pytest.raises(RuntimeError):
            qk.quantum_kernel_svm(n_features=4)
