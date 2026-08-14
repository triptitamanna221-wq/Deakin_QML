import numpy as np

from experiments.wl_separation import run


def test_gcn_gat_identical_to_machine_precision():
    results = run(seed=0)
    assert results["gcn"]["abs_diff"].max() == 0.0
    assert results["gat"]["abs_diff"].max() == 0.0


def test_cqgt_distinguishes_c6_from_2xc3():
    results = run(seed=0)
    assert results["cqgt"]["abs_diff"].max() > 1e-3


def test_result_is_not_seed_fragile():
    for seed in (0, 1, 2):
        results = run(seed=seed)
        assert results["gcn"]["abs_diff"].max() == 0.0
        assert results["gat"]["abs_diff"].max() == 0.0
        assert results["cqgt"]["abs_diff"].max() > 1e-4, f"failed at seed={seed}"
