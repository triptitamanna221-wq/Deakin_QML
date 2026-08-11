import pytest

from cqgt.data.splits import rolling_splits


def test_rolling_splits_are_ordered_and_nonoverlapping():
    train_t, val_t, test_t = rolling_splits(120)
    assert train_t.max() < val_t.min() < test_t.min()
    assert set(train_t) & set(val_t) == set()
    assert set(val_t) & set(test_t) == set()
    assert len(train_t) + len(val_t) + len(test_t) == 120


def test_rolling_splits_rejects_degenerate_T():
    with pytest.raises(AssertionError):
        rolling_splits(2, train_frac=0.6, val_frac=0.2)
