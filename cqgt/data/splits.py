"""Strict rolling temporal splits. BRIEF.md Sec 2.7: assert
max(train_t) < min(val_t) < min(test_t) in code, not just in prose."""
import numpy as np


def rolling_splits(T, train_frac=0.6, val_frac=0.2):
    train_end = int(round(T * train_frac))
    val_end = int(round(T * (train_frac + val_frac)))
    train_t = np.arange(0, train_end)
    val_t = np.arange(train_end, val_end)
    test_t = np.arange(val_end, T)

    assert len(train_t) > 0 and len(val_t) > 0 and len(test_t) > 0, \
        "rolling_splits produced an empty split; check T and fractions."
    assert train_t.max() < val_t.min() < test_t.min(), \
        "rolling temporal split violated: train/val/test overlap or are out of order."
    return train_t, val_t, test_t
