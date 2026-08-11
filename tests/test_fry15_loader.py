from cqgt.data.fry15_loader import ALL_LOADED_YEARS, NETWORK_ANCHOR_YEARS, load_all_years, load_year
from cqgt.data.panel import N_INSTITUTIONS, select_panel


def test_load_year_returns_expected_columns():
    df = load_year(2020)
    for col in ["id_rssd", "name", "m362", "m370", "y832"]:
        assert col in df.columns
    assert len(df) > 0
    assert df["m362"].notna().all()  # 2020 carries M362 for every populated row


def test_2024_has_no_m362_by_design():
    """Regression test for the structural schema gap this project discovered:
    FFIEC's 2024 export omits M362 entirely, not just for some institutions."""
    df = load_year(2024)
    assert df["m362"].isna().all()
    assert df["m370"].notna().any()  # m370 is still real for 2024


def test_network_anchor_years_excludes_2024():
    assert 2024 not in NETWORK_ANCHOR_YEARS
    assert 2024 in ALL_LOADED_YEARS


def test_load_all_years_ids_are_stable_rssd_integers():
    data = load_all_years()
    for y in NETWORK_ANCHOR_YEARS:
        assert data[y]["id_rssd"].notna().all()
        assert (data[y]["id_rssd"] > 0).all()


def test_select_panel_includes_jpmorgan_and_has_n_institutions():
    panel, report = select_panel()
    assert len(panel) == N_INSTITUTIONS
    assert "JPMORGAN CHASE & CO." in panel["name"].values
    assert panel["rank"].iloc[0] == 1  # sorted by size, largest first
    assert panel["y832"].is_monotonic_decreasing


def test_select_panel_excludes_incomplete_institutions_with_reasons():
    panel, report = select_panel()
    excluded_ids = set(report["excluded_incomplete"]["id_rssd"])
    included_ids = set(panel["id_rssd"])
    assert excluded_ids.isdisjoint(included_ids)
    assert report["n_candidates_incomplete"] > 0
