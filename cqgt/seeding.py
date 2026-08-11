import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU) RNGs for one reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Independent numpy Generator for code that should not disturb global state
    (e.g. data generation called alongside a torch training loop)."""
    return np.random.default_rng(seed)
