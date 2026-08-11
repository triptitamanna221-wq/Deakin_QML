"""Select the N=12 institution panel from real FR Y-15 data.

Rule (per project constraint, not BRIEF.md's original wording -- adjusted
after discovering FFIEC's 2024 export omits M362 entirely, see NOTES.md):
among the four real network-anchor years (2020-2023), keep institutions
with BOTH m362 (intra-financial-system assets) and m370 (liabilities)
populated in ALL four years, rank by mean Total Exposures (Y832, the
standard G-SIB size measure), and take the largest N. No imputation: an
institution missing any anchor year is excluded outright, not filled in.
"""
import pandas as pd

from cqgt.data.fry15_loader import NETWORK_ANCHOR_YEARS, load_all_years

N_INSTITUTIONS = 12


def _complete_mask(anchor_df):
    def is_complete(g):
        return g["m362"].notna().all() and g["m370"].notna().all() and len(g) == len(NETWORK_ANCHOR_YEARS)
    complete_ids = anchor_df.groupby("id_rssd").filter(is_complete)["id_rssd"].unique()
    return anchor_df["id_rssd"].isin(complete_ids)


def select_panel(n=N_INSTITUTIONS, years_loaded=None):
    """Returns (panel_df, report) where panel_df has one row per selected
    institution (id_rssd, name, mean m362/m370/y832, rank) and report is a
    dict with 'included', 'excluded_too_small', 'excluded_incomplete' for
    full auditability of who made the cut and why."""
    data = load_all_years(years_loaded) if years_loaded else load_all_years()
    anchor = pd.concat([data[y] for y in NETWORK_ANCHOR_YEARS], ignore_index=True)

    complete = anchor[_complete_mask(anchor)]
    incomplete = anchor[~_complete_mask(anchor)]

    sizes = (complete.groupby(["id_rssd", "name"])[["m362", "m370", "y832"]]
             .mean().reset_index().sort_values("y832", ascending=False).reset_index(drop=True))
    sizes["rank"] = sizes.index + 1

    panel = sizes.iloc[:n].copy()
    excluded_too_small = sizes.iloc[n:].copy()

    incomplete_summary = (incomplete.groupby(["id_rssd", "name"])
                           .agg(years_present=("year", "nunique"),
                                m362_nonnull=("m362", "count"),
                                m370_nonnull=("m370", "count"))
                           .reset_index().sort_values("years_present", ascending=False))

    report = {
        "included": panel,
        "excluded_too_small": excluded_too_small,
        "excluded_incomplete": incomplete_summary,
        "n_candidates_complete": len(sizes),
        "n_candidates_incomplete": len(incomplete_summary),
    }
    return panel, report


def format_report(report):
    lines = []
    lines.append(f"N=12 panel selected from {report['n_candidates_complete']} institutions with "
                  f"complete M362+M370 across all {len(NETWORK_ANCHOR_YEARS)} anchor years "
                  f"({NETWORK_ANCHOR_YEARS[0]}-{NETWORK_ANCHOR_YEARS[-1]}), ranked by mean Total "
                  f"Exposures (Y832).")
    lines.append("")
    lines.append("INCLUDED (rank, RSSD, name, mean Y832 $th):")
    for _, r in report["included"].iterrows():
        lines.append(f"  {int(r['rank']):2d}. {int(r['id_rssd'])}  {r['name']:<45s} {r['y832']:>15,.0f}")
    lines.append("")
    lines.append(f"EXCLUDED as too small (rank 13-{report['n_candidates_complete']}, "
                  f"complete data but not top {N_INSTITUTIONS}): "
                  f"{len(report['excluded_too_small'])} institutions, "
                  f"largest excluded = {report['excluded_too_small'].iloc[0]['name']} "
                  f"(rank {int(report['excluded_too_small'].iloc[0]['rank'])})" if len(report['excluded_too_small']) else "  none")
    lines.append("")
    lines.append(f"EXCLUDED for incomplete data ({report['n_candidates_incomplete']} institutions "
                  f"present in some but not all anchor years -- real M&A/failure churn, not a data bug):")
    for _, r in report["excluded_incomplete"].iterrows():
        lines.append(f"  {int(r['id_rssd'])}  {r['name']:<45s} present {int(r['years_present'])}/{len(NETWORK_ANCHOR_YEARS)} years")
    return "\n".join(lines)


if __name__ == "__main__":
    panel, report = select_panel()
    print(format_report(report))
