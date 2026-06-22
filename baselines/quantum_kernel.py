import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler

def quantum_kernel_svm(n_features, reps=2):
    fmap = ZZFeatureMap(feature_dimension=n_features, reps=reps)
    qkernel = FidelityQuantumKernel(feature_map=fmap)
    return SVC(kernel=qkernel.evaluate)

def dequantized_rff_surrogate(n_features, n_components=500, gamma=1.0):
    """Classical RFF approximation -- if this matches qkernel performance,
    no genuine quantum advantage; report this comparison explicitly."""
    rff = RBFSampler(gamma=gamma, n_components=n_components)
    svm = SVC(kernel="linear", class_weight="balanced")
    return rff, svm
