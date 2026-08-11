"""Quantum-kernel SVM baseline. Per BRIEF.md Sec 2.4: qiskit-machine-learning's
FidelityQuantumKernel API has drifted across versions, so this baseline is
optional -- import failure here must not break the rest of the package. If
QISKIT_ML_AVAILABLE is False at runtime, skip this baseline and note it in
results, do not fabricate a substitute number.
"""
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    QISKIT_ML_AVAILABLE = True
except ImportError:
    QISKIT_ML_AVAILABLE = False


def quantum_kernel_svm(n_features, reps=2):
    if not QISKIT_ML_AVAILABLE:
        raise RuntimeError(
            "qiskit-machine-learning is not installed or its API is incompatible; "
            "skip the quantum-kernel baseline and note it in results (BRIEF.md Sec 2.4)."
        )
    fmap = zz_feature_map(feature_dimension=n_features, reps=reps)
    qkernel = FidelityQuantumKernel(feature_map=fmap)
    return SVC(kernel=qkernel.evaluate)


def dequantized_rff_surrogate(n_features, n_components=500, gamma=1.0):
    """Classical RFF approximation as a single fitted Pipeline (not a loose
    (rff, svm) tuple, which made it easy to fit one and predict with the
    other's state) -- if this matches qkernel performance, there is no
    genuine quantum advantage; report this comparison explicitly."""
    return Pipeline([
        ("rff", RBFSampler(gamma=gamma, n_components=n_components)),
        ("svm", SVC(kernel="linear", class_weight="balanced")),
    ])
