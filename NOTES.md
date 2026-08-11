# NOTES.md — build log, deviations, and limitations

Running log per BRIEF.md §7. Newest entries at the bottom of each phase's section.

---

## Phase 0 — Foundation (2026-08-11)

### Environment
- System `python3` resolved to Homebrew's 3.14.0, which has no wheel coverage yet
  for `torch_geometric`/`qiskit-aer` on this machine. Used `/usr/local/bin/python3.12`
  instead and built the project venv (`.venv/`) on that. `requirements.txt` is
  pinned to exact versions that installed and passed tests on this interpreter —
  see the header comment in that file for the reinstall command.
- XGBoost on macOS needs the OpenMP runtime, which was not present. Asked the user
  before touching the system; they chose `brew install libomp` over guarding the
  import. Installed via Homebrew (small, reversible, standard dependency for
  XGBoost/LightGBM on macOS).
- `qiskit-machine-learning==0.9.0` installed cleanly and `FidelityQuantumKernel`
  imports and runs against the pinned `qiskit==2.5.1`. Included it in
  requirements.txt, but the import in `baselines/quantum_kernel.py` is still
  guarded (try/except -> `QISKIT_ML_AVAILABLE` flag) per BRIEF.md's own framing
  of that API as version-fragile — if a future `pip install` on another machine
  resolves an incompatible version, the rest of the package still imports and
  only that one baseline is skipped and noted.

### Defects fixed (BRIEF.md §1)
- `experiments/treewidth_entropy.py`: `_, tw = treewidth_min_fill_in(...)` →
  `tw, _ = ...`. Added a regression test against K5 (known treewidth 4) and C6
  (known treewidth 2) so this can't silently regress.
- `baselines/gnn.py`: `forward()` now actually threads `edge_weight` into
  `GCNConv` (native `edge_weight` arg) and into `GATConv`/`TransformerConv`
  (`edge_dim=1` at construction, `edge_attr` at call time). `SAGEConv` has no
  edge-weight mechanism in torch_geometric — left topology-only, documented in
  a code comment, and to be reported as a known limitation of that one baseline
  rather than silently pretended-fixed.
- `cqgt/circuit.py`: `ctqw_layer` replaced the diagonal, amplitude-preserving
  `RZZ` with `RXX(theta_ij)` then `RYY(theta_ij)` per edge (these two gates
  commute, so composition order doesn't matter — verified in
  `tests/test_qsim.py::test_hopping_conserves_total_z`, which also confirms the
  physical sanity check that total <Z> is conserved by a pure hopping layer,
  since X_iX_j+Y_iY_j commutes with total Z). `theta_ij = tau0 * w_ij` now uses
  the per-edge weight, so `build_cqgt_circuit` takes an `edge_weights` argument
  instead of discarding it. `expectation_z`'s dead `if False else` ternary
  removed; now uses `SparsePauliOp` instead of a bare Pauli string.
- `cqgt/counterfactual.py`: `exact_counterfactual` dropped the unused
  `alpha, beta, gamma, macro_loadings, macro_factors, f_weights` parameters —
  `model_predict_fn(W, X)` already has to close over any model state, so those
  were dead and implied a re-derivation that never happened.
- `baselines/quantum_kernel.py`: `dequantized_rff_surrogate` now returns a
  single fitted `sklearn.pipeline.Pipeline([rff, svm])` instead of an untied
  `(rff, svm)` tuple. Also switched `ZZFeatureMap` (deprecated as of Qiskit 2.1)
  to the functional `zz_feature_map`.
- Packaging: `__init__.py` added to `cqgt/`, `baselines/`, `data/`,
  `experiments/`, `scripts/`, `tests/`. `pyproject.toml` added
  (`pip install -e .`), verified every cross-package import (`from cqgt.x
  import y` etc.) resolves.

### Built (not just fixed)
- `cqgt/qsim.py`: the ~100-line batched PyTorch statevector simulator per
  BRIEF.md §2.2 was written now, in Phase 0, because GATE 0 requires its
  Qiskit-equivalence test to pass — even though its consumer (`cqgt/model.py`)
  doesn't exist until Phase 2. State layout: `(batch, 2, 2, ..., 2)`, qubit 0 is
  the fastest-varying trailing axis (matches Qiskit's little-endian
  `Statevector` reshaped in C order — verified directly, not assumed). Gates
  implemented: `ry`, `rz`, `rzz`, `rxx`, `ryy`, and a `hopping = rxx then ryy`
  convenience. All gate angles accept either a shared scalar or a `(batch,)`
  tensor (needed since the embedding angle is per-sample but layer parameters
  are shared across the batch). Uses `complex128` throughout to match Qiskit's
  default precision — needed to clear the `atol=1e-6` bar.
  - `tests/test_qsim.py`: parametrized over qubit counts 2–5 and 4 seeds,
    compares full statevector amplitudes (not just `<Z>`) against
    `qiskit.quantum_info.Statevector` on a mixed RY/hopping/RZZ/RZ circuit.
    Max observed error was ~1e-16 (machine precision), well under the 1e-6
    bar. Also checks `.backward()` produces finite, nonzero gradients — this
    is the thing that lets Phase 2 train by exact backprop instead of
    parameter-shift, per the brief's hard constraint.
- `configs/main.yaml`, `cqgt/seeding.py`, `results/`, `figures/` scaffolding
  per BRIEF.md §6 Phase 0 checklist.

### Explicitly deferred (not fixed now — by design, not oversight)
Per BRIEF.md §6, these defects are architectural rebuilds that belong to later
phases, not isolated bug fixes:
- `cqgt/hamiltonian.py` — `H` is still not consumed by anything. Wiring
  `e^{-iH delta}` into a forward pass is `cqgt/model.py`, which is Phase 2.
- `cqgt/attention.py` — fidelity attention output `A` is still not consumed.
  Wiring it into `x_tilde_i = sum_j A_ij h_j` is also Phase 2 (§2.3 step 5).
- `data/generate_exposures.py`, `scripts/build_panel.py` — star-graph /
  treewidth, missing labels, missing temporal structure, `n_banks=200` default,
  no feature standardization. All of this is superseded by the Phase 1 data
  module (Eisenberg–Noe cascade, core–periphery fallback, real temporal panel)
  per §2.1 / §6 Phase 1. Patching the old generator now would be wasted work.

### Known limitations carried forward
- `SAGEConv` baseline cannot receive edge weights (torch_geometric API
  limitation) — will show up as a capacity/information disadvantage in T1/T2,
  not a bug.
- Quantum-kernel SVM baseline depends on `qiskit-machine-learning`'s API
  remaining stable; guarded so a future drift degrades gracefully to "skipped
  and noted" rather than breaking the whole suite.

### GATE 0 status: PASS
`pytest -q` → 28 passed, 0 failed (includes the mandatory qsim-vs-Qiskit
equivalence test). Reproduce with:
```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
```

---

## Phase 1 — Data (2026-08-11)

### Real data provenance
User supplied 5 real FFIEC FR Y-15 "Snapshot Indicators" CSVs (year-end
2020–2024, `data/raw/fry15/`) and 6 real FRED series CSVs (`data/raw/fred/`).
An earlier automated fetch attempt (not run by this session) had left
Cloudflare CAPTCHA-challenge HTML pages saved with a `.csv` extension in
`data/raw/fry15/`; those were deleted before the real files were added. The
FRED files were legitimate on first inspection (proper `observation_date,
SERIES_ID` FRED export format, real historical values).

### Constraint 1 — select by column name; per-year mnemonic table
FR Y-15 "Snapshot Indicators" reports every G-SIB indicator item three
times, under prefixes `RISK*`/`RISI*`/`RISO*` — one populated block per
reporting population (top-tier US BHC, US intermediate holding company of a
foreign banking organization, or the foreign parent). Exactly one block is
populated per institution-year (one filer, Discover Financial Services in
the 2023 file, explicitly entered `0` rather than leaving the other two
blocks blank — handled as "populated" meaning non-null AND non-zero, not
just non-null, in `cqgt/data/fry15_loader.py::_coalesce_block`).

Column names, order, and even which fields exist drift every single year.
Explicit per-year map (`cqgt/data/fry15_loader.py::YEAR_COLUMN_MAP`), values
are the base column name before the RISK/RISI/RISO prefix:

| Year | id col | name col | date col | m362 (assets) | m370 (liabilities) | y832 (size) | extra fields (m376,m390,m405,m408,m411,m422,m426) |
|---|---|---|---|---|---|---|---|
| 2020 | `ID_RSSD` | `NAME` | `DT` | `M362` | `M370` | `Y832` | present |
| 2021 | `ID_RSSD` | `Name (Legal)` | `DT` | `M362` | `M370` | `Y832` | present |
| 2022 | `ID_RSSD` | `Name (Legal)` | `DT` | `M362` | `M370` | `Y832` | present |
| 2023 | `ID_RSSD` | `NAME` | `AsOfDate` | `M362` | `M370` | `Y832` | present |
| 2024 | `ID_RSSD` | `Name` | `As of Date` | **absent** | `M370` | `Y832` | present |

Field semantics used (identified from BRIEF.md's own naming plus standard
Basel/FSB G-SIB indicator ordering, cross-checked against known bank sizes —
e.g. RISKY832 for JPMorgan ≈ $4.4T, matching its known total leverage
exposure): `M362`=Intra-Financial-System Assets, `M370`=Intra-Financial-
System Liabilities, `Y832`=Total Exposures (G-SIB size indicator, leverage-
ratio denominator). `Y896`/`Y862` (present in 2020 and 2024 only) could not
be confidently identified and are unused. Two files also carry a literal
mojibake BOM (`ï»¿`) baked into the first header cell as text, not a real
UTF-8 BOM byte — stripped explicitly in `_strip_bom_prefix`.

**2024 structurally omits M362** (0 of 45 columns match "M362", verified by
direct grep, vs 3 of N in every other year — not a per-institution gap, the
field does not exist in that year's export). Flagged to the user before
building anything downstream; they chose (over carrying forward 2023's value
or relaxing the completeness rule to M370-only) to **drop 2024 as a network-
reconstruction anchor** and keep only its available real fields (`m370`,
`y832`, and the 7 extra indicator items) as potential future feature
inputs. **Network reconstruction therefore rests on 4 real annual anchors
(2020–2023), not 5.** This must be stated plainly in the paper (§IV-A).

### Constraint 2 — N=12 panel selection: no imputation
Rule: among the 4 real anchor years, keep institutions with BOTH `m362` and
`m370` populated in **all four** years, rank by mean Total Exposures
(`Y832`), take the largest 12. An institution missing any anchor year is
excluded outright, not filled in — see `cqgt/data/panel.py`.

- 48 institutions had complete M362+M370 data across all 4 anchor years.
- **JPMorgan Chase & Co. is rank 1** ($4.44T mean Y832) — the rule does not
  drop it, so no rule change was needed.

**Included (rank, institution, mean Y832 $thousands):**
1. JPMorgan Chase & Co. — 4,442,716,925
2. Bank of America Corporation — 3,583,365,900
3. Citigroup Inc. — 2,952,041,175
4. Wells Fargo & Company — 2,283,341,638
5. Goldman Sachs Group, Inc. — 1,833,568,900
6. Morgan Stanley — 1,432,549,425
7. U.S. Bancorp — 742,806,800
8. Toronto-Dominion Bank, The — 652,600,660
9. PNC Financial Services Group, Inc. — 637,734,666
10. Truist Financial Corporation — 628,018,900
11. TD Group US Holdings LLC — 562,842,522
12. Charles Schwab Corporation, The — 557,232,925

**Excluded as too small:** 36 institutions with complete data but rank
13–48 (largest excluded: Capital One Financial Corporation, rank 13).

**Excluded for incomplete data (12 institutions) — real M&A/failure churn,
not a data bug:** SVB Financial Group and First Citizens Bancshares (SVB
collapsed 2023, First Citizens absorbed its assets and only became a Y-15
filer that year), Credit Suisse Group AG / Credit Suisse AG (acquired by
UBS in 2023), BBVA USA Bancshares / Banco Bilbao Vizcaya Argentaria (BBVA
USA acquired by PNC in 2021), Flagstar Financial (renamed/restructured;
only present 2023+), CIBC Bancorp USA / Canadian Imperial Bank of Commerce,
MUFG Americas Holdings, Synchrony Financial, BPCE — each present in only
1–3 of the 4 anchor years.

**Limitation disclosed for §V:** the panel includes one parent/subsidiary
pair — Toronto-Dominion Bank, The (rank 8, the Canadian parent) and TD
Group US Holdings LLC (rank 11, its US intermediate holding company) — as
two distinct RSSD filers. This is how FR Y-15 legally structures them, but
if intra-TD-group US–Canada exposures are a material share of either
entity's M362/M370, this could inflate the *apparent* interconnectedness
between these two specific nodes relative to two fully independent banking
groups. Disclosed, not hidden; not corrected for.

### Constraint 3 — marginal imbalance: explicit, not silently unconverged
Sum(assets) ≠ sum(liabilities) for the 12-bank panel every year, because
each institution's FR Y-15 figures include claims on/obligations to
counterparties outside the panel. **Decision: proportionally rescale both
vectors to their common average total** (`cqgt/data/reconstruction.py::
rescale_marginals`), not a rest-of-world node — simpler, and IPF then has a
feasible balanced problem by construction. Measured imbalance before
rescaling (assets > liabilities every year — the panel's very largest banks
are net lenders within the reported financial system):

| Year | Σ assets ($k) | Σ liabilities ($k) | Imbalance |
|---|---|---|---|
| 2020 | 1,518,734,288 | 1,386,575,102 | 9.1% |
| 2021 | 1,635,444,078 | 1,260,191,470 | 25.9% |
| 2022 | 1,660,551,737 | 1,281,775,357 | 25.7% |
| 2023 | 1,783,456,252 | 1,342,869,800 | 28.2% |

Both `maxent_reconstruct` (RAS/IPF) and `mindensity_reconstruct` (greedy
fewest-links, in the spirit of Anand/Craig/von Peter 2015 — not their exact
LP formulation, disclosed in the docstring) converge to the rescaled
marginals to floating-point precision (`tests/test_reconstruction.py`). The
greedy mindensity algorithm needed two non-obvious fixes to converge exactly
rather than stranding ~0.03% of the flow: (1) when the largest remaining
lender's only counterparty would be itself, fall through to the
next-largest lender instead of aborting the whole pass; (2) resolve any
genuinely self-referential residual with a 3-edge transportation-simplex
pivot through an existing edge elsewhere in the network, which leaves every
other institution's own marginals exactly untouched.

### Constraint 4 — temporal honesty: T=120 is synthetic structure on real anchors
FR Y-15 is annual. **The panel's 120 "weekly" snapshots are NOT 120 real
observations.** `cqgt/data/temporal.py::build_real_anchor_panel` places the
4 real annual network anchors at evenly-spaced indices (t=0, 40, 79, 119 for
T=120) and linearly interpolates the reconstructed exposure matrix and
marginals between them, then overlays a common AR(1) multiplicative noise
path (ρ=0.95) with variance tripled inside the crisis window (t∈[70,85]).
**Only the 4 anchor weeks are real data; every other week's network is a
disclosed interpolation + noise fabrication.** The macro factors (§ below)
are looked up against a synthetic calendar (linearly spaced dates between
the first and last anchor's real dates) but the FRED *values* returned for
those dates are genuinely real, as-of/forward-filled. **This must be stated
plainly in §IV-A**: real annual cross-section, synthetic weekly interpolation
overlay — not a real weekly panel.

### Feature construction — 4 of 6 features are disclosed proxies
FR Y-15 has no capital/equity or market-price data, so it cannot supply a
literal leverage ratio, Tier-1 ratio, CDS spread, or liquidity ratio.
`cqgt/data/features.py` builds:
- `size_proxy` ("leverage" slot) = Y832 (total exposure) — a size proxy, not
  a real leverage ratio.
- `complexity_proxy` ("Tier-1" slot) = M411 (OTC derivatives notional) / Y832.
- `funding_proxy` ("CDS-proxy" slot) = M376 (securities outstanding,
  wholesale funding) / Y832.
- `liquidity_proxy` = M390 (payments activity) / Y832.
- `out_degree` = real, from the reconstructed network's row sums.
- `macro_breadth_proxy` ("macro sensitivity" slot) = (M422+M426, cross-
  jurisdictional claims+liabilities) / Y832, in place of an empirically
  estimated β against C_t (only 4 annual points exist; a regression beta
  would be unreliably noisy at that sample size).
All z-scored using **training-split statistics only**, then `arctan`, per
BRIEF.md §2.1.

### Eisenberg–Noe cascade — calibration story (important, read before Phase 2)
`external_assets[i] = Y832[i] - M362[i]` (real, FR Y-15-derived). `equity[i]
= capital_ratio × Y832[i]` with `capital_ratio=0.08` — **assumed**, a
literature-typical G-SIB leverage-ratio buffer, not a measured figure for
any institution (FR Y-15 reports no capital data at all).

**Finding during calibration, not a bug:** for these 12 real large banks,
external assets are 7×–180× their interbank liabilities L (they fund mainly
outside the interbank market — realistic for G-SIBs). An Eisenberg-Noe
*payment* shortfall — the only channel through which one bank's shock can
reach another's equity — therefore requires shocking a bank's external
assets almost to zero; partial haircuts (5–50%) wipe out the thin assumed
8% equity buffer on paper (crossing any reasonable loss threshold) without
ever causing an actual missed interbank payment, so they produce **zero
spillover by construction**, not by bug. Verified directly: a 90% haircut
on one bank still causes exactly 0.000 loss for every counterparty; a 98%
haircut starts producing real spillover (0.16–0.52 loss fractions for
directly-exposed creditors). **Final design: `shock_frac=1.0`** (a shocked
bank's external assets are fully wiped — modeling a genuine failure event,
not a partial haircut), each bank independently shocked with probability
`p_shock` each period (Bernoulli, seeded per-(seed,t)), `p_shock` tuned by
bisection search per dataset variant to hit the 8–15% target base rate
(`cqgt/data/cascade.py::tune_p_shock`).

Tuned values and resulting composition (full T=120×N=12=1440-label panel):

| Dataset | p_shock (tuned) | base rate | direct positives | spillover positives |
|---|---|---|---|---|
| real, maxent | 0.1019 | 11.25% | 145 | 17 (10.5% of positives) |
| real, mindensity | 0.0713 | 8.40% | 97 | 24 (19.8% of positives) |
| fallback (core-periphery) | 0.0713 | 11.18% | 97 | 64 (39.8% of positives) |

Consistent with the literature (Upper 2011; Anand/Craig/von Peter 2015):
mindensity's concentrated exposures produce roughly 2× the spillover rate of
maxent's smoothed-out exposures for the same shock parameters, and the
deliberately-concentrated core-periphery fallback shows the most spillover
of all three. This part of the pipeline is working as intended and is
itself a legitimate, reportable empirical finding regardless of the GATE 1
outcome below.

Ground-truth counterfactuals (`cqgt/data/cascade.py::ground_truth_delta_r`):
for each edge with W[i,j]>0, rerun the same shock+clearing with that edge
zeroed, record ΔR = R0 − R_cf where R = mean equity-loss fraction across the
panel. Implemented and unit-tested; not yet run at panel scale (deferred to
Phase 3, where it's actually consumed).

### GATE 1 — CRITICAL FINDING: DOES NOT PASS on any of the three dataset variants

Per the explicit stop-and-report instruction: **the quick GCN does not
clearly beat logistic regression on any of real-maxent, real-mindensity, or
the synthetic fallback**, tested with a lightweight 2-layer GCN
(`scripts/run_gate1.py`) trained on the same rolling 60/20/20 split
(`cqgt/data/splits.py`, asserted in code) used for logistic regression.
Result was re-checked over 5 seeds with more training epochs (300) to rule
out an undertrained-model artifact — the margin stayed small and
inconsistent (fallback: GCN beat logreg in 4/5 seeds by +0.01–0.02 but lost
in 1/5; mindensity: GCN was on average *worse* than logreg, 0.069 vs 0.072,
both barely above the 0.084 prevalence floor).

| Dataset | prevalence | treewidth (min/max/mean) | AUPRC logreg | AUPRC GCN (1 seed) | AUPRC GCN (5-seed mean) | margin |
|---|---|---|---|---|---|---|
| real, maxent | 0.1125 | 11/11/11.00 | 0.1121 | 0.1144 | — | +0.0023 |
| real, mindensity | 0.0840 | 3/6/4.95 | 0.0719 | 0.0778 | 0.0686 | −0.0033 (5-seed mean actually below logreg) |
| fallback (core-periphery) | 0.1118 | 3/3/3.00 | 0.1201 | 0.1305 | 0.1291 | +0.0090 |

(Treewidth note: maxent is fully dense — every anchor year converges to
132/132 possible directed edges since IPF with all-positive marginals never
produces an exact zero — so its treewidth is trivially N−1=11 at every
snapshot; this is expected IPF behavior, not a bug, and is itself evidence
for why the brief frames mindensity as the sparser, more contagion-prone
reconstruction.)

**My read of why:** (1) the shock target is exogenous and i.i.d. across
banks by design (per BRIEF.md's own "apply an exogenous shock to a random
subset" spec) — a large majority of positive labels are therefore
intrinsically unpredictable from pre-shock state, capping achievable AUPRC
for *any* model; (2) `out_degree` is already one of the 6 features fed to
logistic regression, so first-order network-centrality information is not
exclusive to the GCN — only higher-order relational structure would be its
unique advantage, and that signal appears to be smaller than the noise
floor at N=12, T=120, with only 97–162 total positive labels; (3) with only
4 real network anchors, the *topology* barely varies across the 120-week
interpolated panel (mostly smooth weight-magnitude drift between the same 4
endpoints), leaving the GCN little independent structural variation to
learn from across training snapshots.

**This is being reported to the user now, per explicit instruction, rather
than tuned further.** Not proceeding to Phase 2 until they decide how (or
whether) to proceed — options on the table include accepting this as a
disclosed null/negative finding and reframing the paper accordingly,
redesigning the shock mechanism to correlate more with observable network
position (with care to avoid manufacturing leakage), increasing statistical
power (larger T, multiple shock realizations per snapshot), or moving to a
continuous risk-score target instead of a binary threshold label at this
sample size.

### GATE 1 status (original single-draw design): reported as a critical negative finding, project paused for user decision. Superseded below by the refined MC result.
`pytest -q` → 51 passed (Phase 0's 28 + 23 new Phase 1 tests). All Phase 1
infrastructure (loader, panel selection, reconstruction, fallback, temporal
panel, macro factors, features, cascade, splits, GATE 1 script) is built,
tested, and reproducible regardless of how the labeling design evolves.

### User decision on GATE 1 (2026-08-11): increase statistical power, two refinements

User chose Option 1 (increase statistical power), not Option 3 (redesign the
shock mechanism to correlate with network position — explicitly rejected as
manufacturing the signal being tested for). Two specific refinements:

1. **M=20 shock realizations per snapshot**, each a separate training
   example (120 x 20 = 2400 scenario-examples instead of 120), with a
   binary "shocked this draw" node feature appended to the 6-dim feature
   vector (7-dim total). Not leakage: the shock set is the input scenario
   (which banks are hit), not the outcome (who ends up distressed) — this
   is the standard stress-testing framing ("given this scenario and this
   network, who fails?"), analogous to feeding a GNN which nodes are
   seeds in an influence-propagation task.
2. **Primary metric restricted to the non-directly-shocked subset.**
   Directly-shocked banks fail deterministically under `shock_frac=1.0`
   (full wipeout always crosses the 30% loss threshold by a huge margin —
   confirmed in the original GATE 1 run, loss fractions of 500-1100%+ for
   shocked banks), so including them in AUPRC inflates every model
   equally on a trivially-predictable subset and masks the actual
   comparison. The non-shocked subset is pure contagion prediction — the
   only place network structure can causally matter, and exactly the
   claim the paper makes.

**PRE-REGISTERED STOPPING RULE (recorded before running this refined test):**
if, after both refinements, the GCN still does not beat logistic regression
on the non-shocked (spillover) subset, we accept the negative finding,
state the caveat plainly in the paper, and proceed to Phase 2 regardless.
No third redesign will be attempted. This rule is being written down now,
before the refined numbers exist, specifically to prevent iterating on the
label design until a comparison happens to look favorable.

### GATE 1 refined result (M=20 Monte Carlo shock draws, spillover-only metric)

Implementation: `cqgt/data/cascade.py::generate_mc_labels_for_panel` draws
M=20 independent Bernoulli(p_shock) shock realizations per snapshot t
(same tuned `p_shock` as the original single-draw run, same network/features
per t, different random shocked set each draw), producing 120×20=2400
scenario-examples instead of 120. A 7th feature — "shocked this draw"
(binary) — is appended to the 6-dim standardized feature vector for both
models. `scripts/run_gate1_mc.py` trains logistic regression and the same
quick 2-layer GCN architecture on all training-split scenario-examples, and
evaluates AUPRC restricted to nodes where `shocked_this_draw=0` in the test
split (the spillover subset) — confirmed necessary: on the *full* node set,
logreg alone hits AUPRC 0.78–0.92 purely from the shock indicator feature
(directly-shocked banks fail deterministically under `shock_frac=1.0`),
which would have masked any real model comparison.

| Dataset | spillover prevalence | AUPRC logreg (spillover) | AUPRC GCN (spillover) | margin |
|---|---|---|---|---|
| real, maxent | 0.0296 | 0.1232 | 0.1538 | **+0.0306** |
| real, mindensity | 0.0248 | 0.0790 | 0.1266 | **+0.0477** |
| fallback (core-periphery) | 0.0474 | 0.1038 | 0.1113 | **+0.0075** |

3-seed robustness check (GCN weight-init seed varied, same MC-drawn labels):
mindensity margin 0.048–0.058, fallback margin 0.0075–0.0084 — consistently
positive, not a single-seed fluke.

**Pre-registered stopping rule outcome: the GCN beats logistic regression on
the spillover subset for all three dataset variants.** The condition for
accepting the negative finding (GCN does not beat logreg) did not occur, so
per the rule recorded above, this is treated as a pass and the project
proceeds to Phase 2. mindensity shows the clearest, most reliable margin
(consistent with it being the more contagion-prone reconstruction, per the
direct/spillover positive-label composition in the original GATE 1 run);
the fallback network's margin is real but the smallest of the three,
despite having the highest raw count of spillover-positive labels — worth
noting in the paper as a secondary, unexplained observation rather than
over-interpreting.

**Honesty notes for the paper (§IV, §V):** (1) the spillover-only metric is
the scientifically correct primary comparison and should be reported as
such, with the all-node AUPRC reported alongside only as a sanity check,
clearly labeled as inflated by the deterministic direct-shock component;
(2) the "shocked this draw" feature is a scenario input available to every
model equally, not a form of leakage, but it does mean the reported task is
"given a known stress scenario and the network, predict contagion," not
"predict which bank gets hit," which should be stated explicitly in the
methodology section; (3) M=20 Monte Carlo draws per snapshot reuse the same
120 real/interpolated network states 20 times each with different random
shock targets — they increase statistical power for estimating the
spillover relationship but do not add new independent network structure,
consistent with the earlier finding that only 4 real annual anchors exist.

### GATE 1 status: PASS (via the pre-registered MC refinement)
`pytest -q` → 53 passed. GCN beats logistic regression on the spillover
subset for all three dataset variants, holding up over a 3-seed check.
Proceeding to Phase 2 per the user's decision, with the honesty notes above
carried into the paper. Reproduce with:
```
source .venv/bin/activate
python3 scripts/run_gate1.py      # original single-draw diagnostics
python3 scripts/run_gate1_mc.py   # refined M=20 MC / spillover-only result
```
