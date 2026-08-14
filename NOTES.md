# NOTES.md — build log, deviations, and limitations

Running log per BRIEF.md §7. Newest entries at the bottom of each phase's section.

---

## HANDOFF (2026-08-11, end of session) — read this first, written for zero context

**What this project is:** CQGT (Causal Quantum Graph Transformer) for
systemic credit-risk contagion, per `BRIEF.md` in the repo root. Read that
file first if anything below is unclear — it's the full spec this session
has been executing against, phase by phase, with gates that must pass
before advancing.

**Current phase: Phase 2 (Model), NOT complete.** Phases 0 and 1 are done
and committed (GATE 0 and GATE 1 both PASS — see their sections below).
Phase 2's code is built and unit-tested, but **GATE 2 has not been passed**
and **Phase 3 has not started.** Do not start Phase 3 until GATE 2 passes
for real (see below) — that is a standing project rule (BRIEF.md §6), not
new for this handoff.

**Exactly what is blocked on what, right now:**

1. **Immediate next action:** re-run two profiling commands from a clean
   shell (the previous session's background processes will not survive
   into this one; do not try to reattach to them, just re-run):
   ```
   source .venv/bin/activate
   # (a) real per-epoch cost for fallback L3 and all of maxent (mindensity
   #     L1/L2/L3 and fallback L1/L2 are already known — see the "Real
   #     per-epoch measurements" section a few pages below this one)
   python3 -c "
   import time
   import numpy as np
   from cqgt.model import CQGTModel
   from cqgt.data.pipeline import build_paired_dataset, attach_macro_factors
   from cqgt.train import train_stage
   for source in ['fallback', 'maxent']:
       ds = build_paired_dataset(source, n_mc=20, seed=0)
       attach_macro_factors(ds, seed=0)
       n = ds['W_panel'].shape[1]
       edges = list(map(tuple, np.argwhere(ds['W_panel'][0] > 0)))
       print(f'{source}: n_edges={len(edges)}', flush=True)
       for L in [1,2,3]:
           model = CQGTModel(n_qubits=n, n_features=7, edges=edges, n_layers=L)
           start = time.time()
           model, hist = train_stage(model, ds, epochs=2, n_mc_train=None)
           dt = time.time()-start
           print(f'  L={L}: {dt/2:.1f}s/epoch', flush=True)
   "
   # (b) the convergence question: does mindensity L=3's loss actually
   #     plateau, or is it still moving meaningfully, over more epochs?
   #     (a 3-epoch run already showed only ~0.001-0.002/epoch movement —
   #     inconclusive on its own; see "Flat loss curve" section below for
   #     the full context and the LR-normalization hypothesis to test
   #     alongside this)
   python3 -c "
   import numpy as np
   from cqgt.model import CQGTModel
   from cqgt.data.pipeline import build_paired_dataset, attach_macro_factors
   from cqgt.train import train_stage
   ds = build_paired_dataset('mindensity', n_mc=20, seed=0)
   attach_macro_factors(ds, seed=0)
   n = ds['W_panel'].shape[1]
   edges = list(map(tuple, np.argwhere(ds['W_panel'][0] > 0)))
   model = CQGTModel(n_qubits=n, n_features=7, edges=edges, n_layers=3)
   model, hist = train_stage(model, ds, epochs=20, n_mc_train=None)
   for i, h in enumerate(hist):
       print(f'epoch {i}: {h:.5f}')
   "
   ```
   Run both **in the background** (`run_in_background`/`&`) — the second
   one takes on the order of 30-40 minutes at the measured ~102.5s/epoch
   rate for L=3. **Do not run them in the foreground and block on them; do
   other useful work (or just wait) while they complete, and watch memory
   (`sysctl vm.swapusage`) if either seems to be taking far longer than its
   own per-epoch rate would predict** — that was the actual bug found this
   session (see "Swap-thrashing bug" section below), and while it's now
   fixed and verified equivalent (gradient-accumulation refactor,
   `tests/test_train.py::test_gradient_accumulation_matches_single_backward_over_stacked_mean`),
   watch for it recurring under new configurations regardless.

2. **Once those two results exist:** redo the Phase 3 wall-clock projection
   from scratch using the real numbers (the "Wall-clock projection" table
   currently in this file is explicitly marked VOID — do not use it). Apply
   the standing pre-committed rule: if the convergence run shows `E>=3`
   epochs/stage are genuinely needed (not just "still moving a little" —
   read the LR-normalization hypothesis section before concluding this),
   cut maxent's main-table seed count from 5 to 2 (never sparsify maxent —
   the user was explicit and detailed about why, see below). Recompute the
   total wall-clock at whatever `E` and seed counts result, and **report it
   to the user before starting Phase 3** — they set an explicit ~6 hour cap
   and want to be told, not have the decision made for them, if it's over.

3. **Only after that: re-run GATE 2 properly** (full `L=1->2->3` layer
   growth, real `epochs_per_stage` from the convergence evidence, full
   `M=20`, on at minimum mindensity) and confirm loss decreases and CQGT
   beats the validation prevalence floor. **GATE 2 has never actually
   passed** — every attempt so far was either killed early or was a
   short/partial timing run. Do not report GATE 2 as passed without this.

4. **Then, and only then, Phase 3** (baselines, ablations, all metrics,
   the seed/variant budget already agreed: 10 seeds mindensity + 5 maxent +
   5 fallback for the main table, 7 ablations x 5 seeds on mindensity only,
   per the user's explicit instructions this session).

**Two substantive findings from this session that must reach the paper,
regardless of what happens next (do not lose these):**
- **Maxent is a complete directed graph at every snapshot** (132/132
  possible edges, density=1.0000, treewidth=11=N-1 constant) — mechanistically
  guaranteed (IPF/RAS can't zero a positive entry), not a fluke. Explains
  maxent's weak GATE 1 spillover share and margin. **Excludes maxent from
  the entanglement-entropy-vs-treewidth study (F3)** — no treewidth
  variation to correlate against there; mindensity and the fallback carry
  that study instead. Full detail in the "Compute budget negotiation"
  section below.
- **GATE 1 passed via a pre-registered statistical-power refinement**
  (M=20 Monte Carlo shock draws/snapshot, spillover-only AUPRC metric) —
  the original single-draw design showed a genuine negative result first,
  which was reported honestly before the refinement was tried. See "GATE 1"
  sections below for the full pre-registration/stopping-rule precedent,
  which this session's compute-budget negotiation deliberately mirrored.

**Everything else** (per-year FR Y-15 column mapping, the Eisenberg-Noe
cascade calibration story, the N=12 panel selection, all defect fixes) is
unchanged from what's recorded further down and does not need to be
re-verified — only Phase 2 (model training / compute budget / GATE 2) is
in an unfinished, actively-blocked state.

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

### PII redaction (2026-08-14): contact columns stripped from the committed 2024/2025 file

After `data/raw/` was committed (un-ignored per the user's explicit
instruction), the user flagged that the FR Y-15 "Snapshot Indicators"
schema includes a certification/contact block -- and asked to check for
and remove it before it stayed public. Checked: only
`20241231_20250722_FRY15 Snapshot Indicators.csv` (the 2024/2025 file)
carries it; the 2020-2023 files never had these columns under any
RISK/RISI/RISO prefix (grepped all 5 files to confirm before touching
anything). The six columns, and what they actually contained (verified by
reading real values, not assumed from the mnemonic name alone):

| Column | Contents (real values seen) |
|---|---|
| `RISKC490` | CFO name (e.g. "Jeremy Barnum") |
| `RISKJ196` | Submission date |
| `RISK8901` | Preparer name + title (e.g. "Elaine O'Keeffe, Managing Director") |
| `RISK8902` | Preparer direct phone |
| `RISK9116` | A second phone number |
| `RISK4086` | Preparer email address |

None of these are consumed anywhere in `cqgt/data/fry15_loader.py` --
`YEAR_COLUMN_MAP` never references them, so removal is a pure redaction
with zero effect on any pipeline output, verified by re-running the full
test suite (`pytest -q`, 79 passed unchanged) after the edit.

**Mechanics, to keep provenance honest about exactly what changed:** wrote
a one-off Python script (not saved -- run once, ad hoc) that parsed the
file with `csv.reader`, dropped columns by exact header-name match (all
three RISK/RISI/RISO variants of the six mnemonics, though only the
RISK-prefixed ones were actually present), and rewrote with `csv.writer`.
Verified byte-for-byte afterward that nothing else changed: the file's
leading mojibake pseudo-BOM (`ï»¿`, three real UTF-8 characters baked into
the first header cell as text -- see Constraint 1 below, this is NOT a
real BOM byte and `_strip_bom_prefix` depends on it looking exactly like
this) is preserved exactly; row count unchanged (54); line endings
unchanged (bare `\n`, no `\r\n` introduced); every remaining column's data
untouched, only re-ordered by the removal (financial columns keep their
relative order, contact columns spliced out). Grepped the rewritten file
for every name/phone/email pattern that had been present -- none remain.
`data/raw/fry15/` now carries only the FR Y-15 financial indicator fields
`YEAR_COLUMN_MAP` actually uses (plus the small set of "identified from
naming, unused" fields already disclosed in Constraint 1), no personal
contact information.

**Correction to the framing above (2026-08-14, same day):** the user
clarified this is not a privacy remediation -- the FR Y-15 "Snapshot
Indicators" contact fields are published by FFIEC on a public,
unauthenticated government website as the designated public filing
contact for a mandatory regulatory disclosure, so mirroring them here
carried no incremental privacy exposure, and the unmodified originals
remain publicly available from FFIEC regardless of what this repo does.
The user explicitly declined a history rewrite for this reason (a
rebase to scrub the columns from the commit that originally introduced
them was proposed and then cancelled -- `ca06775`, already pushed,
still contains the original six columns in its history, and that is
being left as-is, deliberately, not as an oversight). The redaction
commit (`159e256`) is kept going forward purely as tidiness -- these six
columns are not used anywhere in this project's pipeline, so there is no
reason for the committed copies to carry them -- not because leaving
them was a privacy problem.

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

---

## Phase 2 — Model (2026-08-11)

### User's four Phase 2 requirements (three were listed; flagged the
discrepancy to the user and proceeded on these three, since they don't
block starting the build):
1. Parameter parity: CQGT and every GNN baseline roughly parameter-matched,
   every table reports parameter counts.
2. Paired evaluation: identical splits and identical shock draws across
   all models, so paired bootstrap tests are possible later.
3. 10 seeds, not 5, for the eventual Phase 3 experiment sweep.

Addressed now: (1) `cqgt/data/pipeline.py::build_paired_dataset(source,
seed, n_mc)` is the single entry point every model must use -- same
`(source, seed, n_mc)` triple always yields identical rolling splits
(`rolling_splits`, asserted in code since Phase 1) AND identical M shock
draws (`generate_mc_labels_for_panel`, same `(seed, t, m)`-seeded RNG per
draw). GATE 1's own scripts were refactored onto this shared function so
there is only one paths to a paired dataset, not two slightly-different
ones. (2) `CQGTModel.n_parameters()` reports total trainable scalar count;
printed in every GATE 2 result and will be printed in every Phase 3 table.
(3) is a Phase 3 concern (the actual seed sweep); noted for when baselines
are built.

### Architecture (BRIEF.md Sec 2.3), implemented in `cqgt/model.py`
Built entirely on `cqgt/qsim.py` (never Qiskit, per the hard constraint) so
training is exact backprop:
1. Embed: `RY(2 arctan(w^T x_i))` per qubit, `w` a shared learnable linear
   map from the 7-dim feature vector (6 standardized proxies + the
   "shocked this draw" scenario indicator, see Phase 1's GATE 1 refinement)
   to a scalar.
2. CTQW encoder: one hopping (`RXX.RYY`) application per real exposure
   edge, angle `= tau0 * Ltilde_ij` (`tau0` learnable, `>=0` via softplus).
3. `L` variational layers (`L=1,2,3` via layer-by-layer growth, see
   below), each: `RY(phi_i)` per qubit -> `RZZ(theta_ij)` on exposure edges
   only (topology-matched ansatz -- parameters exist only where a real
   edge exists, not all `N(N-1)` pairs) -> `e^{-i H_t delta_l}`, where
   `H_t`'s entangling part is the same hopping term scaled by `alpha` and
   its diagonal part is `RZ(2 delta_l h_i)`, `h_i = beta f_psi(x_i) +
   gamma sum_k lambda_k^t beta_ik` (`f_psi`: 2-layer MLP; `beta_ik`:
   learnable per-institution macro loadings; `lambda_k^t`: this
   snapshot's real-or-synthetic macro factor value, see below).
4. Measure `<Z_i>` -> `r`.
5. Quantum fidelity attention on `n_a=2` ancilla qubits, reimplemented
   against `qsim` (the original `cqgt/attention.py` is Qiskit-based and
   was kept only as the Phase-0-era reference, not reused here): learned
   per-institution query/key parameter vectors (independent of `x`,
   3 params each for `n_a=2`) build `|q_i>`, `|k_j>`; `F_ij =
   |<q_i|k_j>|^2`; `A = softmax(lambda F)` (`torch.softmax` is exactly the
   original `exp(lam F)/sum` formulation, verified algebraically
   equivalent, cleaner to implement). `x_tilde_i = sum_j A_ij (r_j x_j)`.
6. MLP head on `[r_i ; x_tilde_i]` -> `p_hat_i`.

`alpha`, `beta`, `gamma`, `tau0` are learnable (softplus-constrained `>=0`)
rather than fixed at their `configs/main.yaml` values of 1.0 -- more
principled (the model calibrates how much weight topology vs. local risk
vs. macro gets) and those config values are used only as the initial point.
`cqgt/hamiltonian.py::normalized_laplacian` is reused as-is for the fixed
(non-learnable, data-derived) Laplacian per snapshot; fixed a real
`RuntimeWarning: divide by zero` there (isolated-node edge case, already
correctly handled via `np.where` but still evaluating the division eagerly)
while wiring it in, since this file is now actively in the hot path rather
than deferred.

**Edge set is fixed per dataset, computed once from any snapshot.**
Verified this is safe: real-anchor panels interpolate positive anchor
weights and apply one shared multiplicative noise scale per snapshot (never
per-edge independent noise), so a real edge's weight can shrink but never
flips sign; the fallback panel multiplies by a fixed binary mask every
step. So "topology-matched ansatz" parameters (`zz_params`, and the hopping
terms) are declared once at model construction from `W_panel[0] > 0`.

### Layer-by-layer growth (`cqgt/train.py::grow_model`)
Per BRIEF.md Sec 2.3: train `L=1`, then grow to `L=2`, then `L=3`. Shared
(non-layer) parameters are copied directly across a growth step. Already-
trained layers are copied as-is. Each newly-added layer is initialized as
`N(theta*_prev_layer, (pi/10)^2)` -- centered at the most recently trained
layer's parameters (`growth_std = pi/10`), not the very first layer.
Verified in `tests/test_train.py` with a near-zero `growth_std` that this
converges to an exact copy in the limit.

### Performance finding: qsim is compute-bound, not call-overhead-bound
Before running GATE 2 at the originally-planned settings (M=20 MC draws,
30 epochs/stage), profiled a single forward+backward pass and found it
compute-bound: cost scales close to linearly with `batch_size x n_gates`
(measured directly: B=20 -> 0.27s, B=200 -> 2.85s, B=1440 -> 22.4s for a
fixed toy gate count), not dominated by Python-level per-call overhead as
initially hypothesized. This ruled out the obvious speedup (batching more
scenarios into fewer forward calls) since total compute is conserved
regardless of how work is chunked into calls. A real `L=3`, 26-edge,
`N=12` forward+backward at batch=20 costs ~1.8s; at 72 training timesteps
per epoch that is ~127s/epoch, making the originally-planned 90-epoch
schedule (3 stages x 30 epochs) run for hours per dataset -- not tractable
on CPU-only hardware within this project's timeframe.

**Mitigation for GATE 2 (a sanity check, not final training):** initially
reduced to `n_mc_train=4`/`epochs_per_stage=8` to get a first working
result. The user rejected the idea of a silent scope reduction and asked
for a written, agreed compute budget before Phase 3 -- see the negotiation
below, which supersedes this stopgap.

### Compute budget negotiation (2026-08-11), before Phase 3

User required: profile honestly first, apply optimizations in a specific
order, then report a wall-clock projection for the full Phase 3 sweep
*before* starting it, with an explicit "come back to me" trigger at ~6
hours rather than a unilateral cut. Hard rule: cut seeds/variants, never
epochs below what the GATE 2 curve shows is needed (an undertrained CQGT
losing to a fully-trained GCN would be a false negative, worse than no
result).

**Step 1 -- complex64.** `cqgt/qsim.py`'s `CDTYPE` switched from
`complex128` to `complex64` (paired `RDTYPE = float32` for angle tensors).
Re-ran `tests/test_qsim.py` against Qiskit's `complex128` Statevector
before touching the tolerance: measured max amplitude error **2.2e-7**, max
`<Z>` error **5.4e-7** (n=2..5 qubits, 4 seeds, mixed RY/hopping/RZZ/RZ
circuits) -- matches the ~1e-7 expected from complex64's precision.
`ATOL` loosened `1e-6 -> 1e-5` (an order of magnitude above the measured
worst case) with the measured numbers written into the test docstring, not
just the change. One test (`test_hopping_conserves_total_z`) had a
hardcoded `dtype=torch.float64` comparison that broke under the dtype
switch -- fixed to compare against a tensor of matching dtype instead of a
literal. **63 tests still pass.**

Measured speedup: **~1.2-1.4x** (mindensity L3: 1.77s -> 1.34s), not the
~2x hoped for -- reported as measured, not assumed. Model-level overhead
(Python loops over edges/qubits building small `U` matrices, `movedim`
calls) doesn't shrink with precision, diluting the gain from the actual
tensor-math speedup.

**Step 2 -- vectorization check.** Confirmed by direct code inspection: no
Python loop over the batch dimension exists anywhere in `cqgt/qsim.py` --
every gate (`ry`, `rz`, `rzz`, `rxx`, `ryy`) applies via a single
`torch.einsum` call across the full `(batch, ...)` tensor. No fix needed.
(The Python loops that do exist, in `cqgt/model.py`'s forward pass, are
over edges and qubits -- small fixed counts per architecture, not the
batch/MC-draw dimension, and are not resolvable by vectorization without a
deeper circuit-parallelization rewrite that was out of scope here.)

**Profiled seconds/epoch at full M=20 (72 train timesteps/epoch),
complex64, measured directly:**

| Dataset | edges (of 132 possible) | L1 | L2 | L3 | sum |
|---|---|---|---|---|---|
| mindensity | 24-41 (varies by t) | 41.8s | 67.8s | 96.7s | 206.3s |
| fallback | 26 (constant) | 43.7s | 73.8s | 105.3s | 222.8s |
| maxent | **132 (constant, every t)** | 234.7s | 327.8s | 426.7s | **989.2s** |

**Discovery: maxent is a complete directed graph at every single snapshot**
(density = 1.0000 exactly, `n_edges=132=N(N-1)` with `N=12`, confirmed
programmatically across the whole panel, not just spot-checked) and its
treewidth is a constant 11 = N-1 (the treewidth of the complete graph
`K_12`) at every snapshot -- verified directly, matching the user's
hypothesis exactly. **Mechanistic explanation, not just an empirical
observation:** `maxent_reconstruct` initializes from `outer(assets,
liabilities)` (strictly positive since both marginals are positive after
rescaling) and IPF/RAS is a purely multiplicative row/column scaling --
multiplying a positive number by a positive scale factor can never produce
an exact zero. So maxent is *guaranteed* dense whenever the marginals are
strictly positive, for any N, not a quirk of this particular dataset.
mindensity, by contrast, varies genuinely: 23-41 edges (density 17.4-31.1%),
treewidth 3-6. Fallback's edge *set* is constant by construction (only
weight magnitudes evolve, see Phase 1) at 26 edges (19.7% density),
treewidth constant 3.

**Consequences, to state plainly in the paper (§IV):**
- A complete graph carries no structural information for a model to
  exploit -- every node's topological neighborhood is identical (all other
  N-1 nodes), so message-passing/attention over it is not meaningfully
  different from a fully-connected default. This is the leading explanation
  for maxent's lowest GATE 1 spillover share (10.5% of positives) and
  smallest margin (+0.031) among the three variants -- not a weaker model,
  a structurally uninformative graph.
- Maxent's constant treewidth=11 makes it **degenerate for the entanglement-
  entropy-vs-treewidth study (F3)** -- there is no treewidth variation to
  correlate against. **Maxent is excluded from that specific analysis**;
  mindensity (treewidth 3-6, real variation) and the fallback (guaranteed
  tw>=3 by construction) carry that study.
- "Maximum-entropy reconstruction is structurally degenerate at N=12" is
  reported as a legitimate finding about the method (consistent with
  Anand/Craig/von Peter's framing of maxent as the smooth, contagion-
  under-estimating bracket -- here shown to be smooth to the point of
  carrying no graph structure at all at this N), not a defect in this
  project's pipeline.
- Realized edge count and density for all three variants (table above) will
  be reported in the paper as a matter of course, not just in this log.

**Step 3 -- ablations scope.** Per BRIEF.md Sec 2.5 and standard practice,
confirmed with the user: the 7 ablations run on mindensity only (the
ablation-primary dataset -- clearest GATE 1 margin, genuinely variable
topology, more contagion-prone per Anand/Craig/von Peter's framing). The
3-variant sweep (maxent/mindensity/fallback) applies to the main baseline
table (T1) only.

**Step 4 -- seed budget (revised down from the user's original 10):** main
table gets **10 seeds on mindensity, 5 on maxent, 5 on fallback**; ablations
get **5 seeds on mindensity only**. Every results table states its seed
count explicitly.

**Wall-clock projection, CQGT training only, before starting anything
(FIRST DRAFT):**

| E (epochs/stage) | Main table (10+5+5 seeds) | Ablations (7x5 seeds, mindensity) | Total |
|---|---|---|---|
| 20 | 45.1 hrs | 40.1 hrs | 85.2 hrs |
| 5 | 11.3 hrs | 10.0 hrs | 21.3 hrs |
| 1 (too few to trust) | 2.3 hrs | 2.0 hrs | 4.3 hrs |

Over the ~6 hour cap at any realistic epoch count, driven overwhelmingly by
maxent (5 seeds of maxent cost as much as ~24 seeds of mindensity). This
does not yet include BRIEF.md Sec 2.4's other N=12 quantum-circuit
baselines (generic hardware-efficient VQC, topology-matched VQC without the
Hamiltonian), which are the same simulation-cost class as CQGT and would
add further to this total.

**>>> VOID, superseded below.** These per-epoch numbers (the "Profiled
seconds/epoch" table above) were extrapolated from a single isolated
forward+backward call x72, never actually run through `train_stage`'s real
training loop. When a real run was attempted at these settings, it hit the
swap-thrashing bug described in the next section and never produced a
trustworthy number. The wall-clock table above must be treated as directional
only (it happened to be close to the post-fix real numbers for mindensity,
~8% low, but this was not verified for fallback/maxent before the bug was
found) -- do not cite the 85.2/21.3/4.3 hour figures without the redo below.

**Pre-committed decision (recorded before evidence, mirroring the GATE 1
pre-registration precedent), status: NOT YET APPLIED, still the standing
rule:** if real per-epoch measurements + a real convergence-epoch-count
determination show `E>=3` epochs/stage are needed, **cut maxent to 2 seeds
in the main table** (not sparsify it -- the user was explicit that an
epsilon-thresholded maxent would no longer be the maximum-entropy bracket
that justifies the dual-reconstruction protocol in Anand/Craig/von Peter;
fewer seeds with wider disclosed confidence intervals is the correct, much
cheaper price instead). **The redo (real numbers, real epoch count) has not
completed -- see HANDOFF at the top of this file and the section below.**

### Swap-thrashing bug found, fixed, and verified equivalent (2026-08-11)

Launched the convergence run implied above (mindensity, full M=20, 20
epochs/stage) to get real numbers. It ran for **2+ hours** without
producing a single epoch of output, against a ~70min estimate. Diagnosed
before assuming it was "just slow": `ps` showed the process alive but
`vm_stat`/`sysctl vm.swapusage` showed **17.5GB of 18GB total swap in use**
and **>1.079 billion translation faults** on the process -- genuine memory
thrashing, not merely slow arithmetic.

**Root cause:** the original `train_stage` (`cqgt/train.py`) built one
computation graph spanning all `len(train_t)` (~72) timesteps' forward
passes -- appending each timestep's loss tensor to a Python list -- before
calling `torch.stack(losses).mean().backward()` once per epoch. This holds
~72x one timestep's graph in memory simultaneously before releasing any of
it. Peak memory exceeded physical RAM and the OS started swapping, and
swap I/O is orders of magnitude slower than the arithmetic itself, which is
why wall-clock blew up far more than raw FLOP count would predict.

**Fix:** changed to per-timestep `backward()` with gradient accumulation --
for each `t`, compute `loss_t = BCE(...) / n_t` and call `loss_t.backward()`
immediately (accumulating into `.grad`), instead of stacking all losses
first. This bounds peak memory to one timestep's graph regardless of
`len(train_t)`. Confirmed healthy afterward: RSS ~250-450MB (vs. swap
exhaustion before), no swap growth attributable to the process.

**Verified mathematically equivalent, not assumed** (explicit instruction,
since an unverified refactor at this point would make every downstream
result suspect): added
`tests/test_train.py::test_gradient_accumulation_matches_single_backward_over_stacked_mean`,
comparing per-parameter gradients from (a) one `backward()` over
`torch.stack(losses).mean()` and (b) `backward()` per timestep on
`loss_t/n_t`, accumulated, on a tiny 3-timestep/2-layer toy case with two
`deepcopy`-identical models. **Matched to atol=1e-6, rtol=1e-5 on every
parameter.** This is expected (no BatchNorm or other cross-example shared
state exists in `CQGTModel`, so linearity of differentiation guarantees
sum-of-grads == grad-of-sum exactly, up to floating-point rounding) but was
checked, not assumed. **64 tests pass** (63 + this one).

### Real (post-fix, non-extrapolated) per-epoch measurements

Measured by actually running `train_stage` for 2-3 real epochs and timing
the wall-clock, not extrapolating from one isolated call:

| Dataset | edges | L1 | L2 | L3 |
|---|---|---|---|---|
| mindensity | 24 | **44.4s** (measured, 3 epochs) | **75.7s** (measured, 3 epochs) | **102.5s** (measured, 3 epochs) |
| fallback | 26 | **48.6s** (measured, 2 epochs) | **77.4s** (measured, 2 epochs) | not yet measured -- job in flight, see HANDOFF |
| maxent | 132 | not yet measured -- job in flight, see HANDOFF | not yet measured | not yet measured |

mindensity's real numbers are close to (about 8% above) the original
extrapolated estimate (206.3s vs. real 222.6s summed L1+L2+L3) -- the
compute-cost extrapolation itself was roughly sound; the actual danger was
specifically the accumulation bug causing thrashing when the loop was
really executed, which a single-call extrapolation could never have caught.
**fallback L3 and all of maxent's real numbers are NOT YET MEASURED** --
the background job computing them (`b8oe1frk3`) was still running when this
session's context ran out. Partial output is on disk at
`/private/tmp/claude-501/-Users-tripti-Deakin-QML/5f0bd0b4-db9a-4199-be07-0b00e314cd60/tasks/b8oe1frk3.output`
but that path is unlikely to survive into a new session/machine state --
**treat as informational only; re-run the profiling command in the new
session** (see HANDOFF).

### Flat loss curve observed -- LR-normalization hypothesis, NOT YET INVESTIGATED

The first real 3-epoch-per-stage run (mindensity, M=20, used for the timing
measurement above) produced this loss curve:
```
L=1: 1.27000 -> 1.26830 -> 1.26710
L=2: 1.26340 -> 1.26180 -> 1.26090   (warm-started from L=1's endpoint)
L=3: 1.26330 -> 1.26170 -> 1.26070   (warm-started from L=2's endpoint)
```
Movement per epoch is small (~0.001-0.002), and only 3 epochs/stage were
run, so **it is not yet known whether this is a near-plateau or just slow
movement that would continue meaningfully over more epochs.** A 20-epoch,
L=3-only run (`bg5fo3lm5`, mindensity, no layer growth -- isolating the
convergence question from the growth schedule) was launched to get a real
answer but had produced **zero output** when this session's context ran
out (est. ~34 min needed at the measured 102.5s/epoch rate for L=3; check
`/private/tmp/claude-501/-Users-tripti-Deakin-QML/5f0bd0b4-db9a-4199-be07-0b00e314cd60/tasks/bg5fo3lm5.output`,
though as above this path may not survive into a new session).

**Hypothesis, not yet confirmed either way:** `train_stage` divides each
timestep's loss by `n_t` (~72) before `backward()`, so the accumulated
gradient is a *mean* over timesteps (verified mathematically identical to
the original mean-based approach, see the equivalence test above -- this is
NOT a bug). But combined with a fixed `lr=0.01` and
`grad_clip_norm_=1.0` applied to the *accumulated* (summed-then-implicitly-
averaged) gradient, the effective step size per epoch may be small enough
that many more than 3, or even 20, epochs are needed to see real movement --
or alternatively gradient clipping could be biting harder than intended on
the accumulated norm. **Next session should**: (1) read the 20-epoch curve
once available and look for a plateau vs. continued descent, (2) if still
inconclusive, try training with grad clipping disabled or a higher LR on a
short run to see if the loss moves faster, to distinguish "genuinely
converging slowly, needs a real epoch budget" from "training dynamics are
mis-tuned, needs an LR/clipping fix" -- these have very different
implications for the epochs-per-stage budget decision.

### Flat loss curve diagnosed (2026-08-11, resumed session) -- NOT a bug

Per the user's STEP 1 instruction (overfit test -> gradient norms -> LR
sweep, time-boxed to 3 attempts), ran ATTEMPT 1: overfit 20 examples (a
single fixed timestep, all n_mc=20 shock draws as the batch, no
regularization -- none exists in this codebase anyway -- lr=0.02, 400
epochs), logging per-parameter-group gradient norms every 20 epochs
(quantum_params [phi, zz_params, delta, query_params, key_params,
attn_lambda], f_psi, mlp_head, tau0, alpha_beta_gamma, embed+macro_loadings).

**Result: loss 1.27962 -> 0.00003 over 400 epochs, 100% node-label accuracy
by epoch 60.** Gradient norms across every parameter group stayed within
~1 order of magnitude of each other throughout training (e.g. epoch 0:
quantum_params=4.6e-3, mlp_head=5.3e-2, tau0=5.1e-4, alpha_beta_gamma=2.0e-4
-- smaller, but nowhere near the orders-of-magnitude gap that would signal
a barren plateau). **The architecture is not broken and there is no
empirical barren-plateau signature at N=12.** This resolved STEP 1 in one
attempt (of the 3 allowed); the gradient-norm-only and LR-sweep methods
were not needed as separate attempts.

**Confirmed on the real full panel**, not just the 20-example subset: ran
25 epochs, L=3 fixed (no growth), mindensity, full M=20, all 72 train
timesteps, production lr=0.01. Loss moved 1.26534 -> 1.25112 -- monotonic
every epoch but decelerating, i.e. still looking "flat" by raw BCE
magnitude. **But validation spillover-subset AUPRC = 0.3374 against a
prevalence floor of 0.0170** -- a ~20x lift, already clearing GATE 2's real
bar (beat the prevalence floor) at this budget.

**Root cause of the "flat loss" symptom: not a bug.** Two compounding
factors, both artifacts of how the number was produced, not the model:
(1) class-weighted BCE with extreme spillover-subset imbalance
(pos_weight ~58) keeps the raw loss magnitude high even when the model is
discriminating well -- loss value alone is a poor proxy for AUPRC/ranking
quality under this weighting; (2) the original flat-curve observation
(1.2700->1.2671) came from a 3-epoch profiling run, `train_stage`'s
`CosineAnnealingLR(T_max=max(epochs,1))` therefore used `T_max=3` for
that specific call and collapsed the LR before any real learning could
happen -- **this is not a code bug** (T_max correctly tracks whatever
`epochs` is passed to that call), it was simply the wrong `epochs` value
for a convergence judgment, taken from a call that was only ever meant to
measure per-epoch wall-clock.

### Growth-schedule finding: dropped for Phase 3, not just GATE 2 (2026-08-11)

The Attempt 1 gradient-norm evidence directly bears on BRIEF.md's own
framing of layer-by-layer growth (L=1->2->3) as barren-plateau (challenge
C4) mitigation: if there is no barren-plateau signature at N=12, the
mitigation is empirically unnecessary at this scale. **Decision (user,
2026-08-11): drop layer-wise growth entirely -- CQGT and all 3 Phase 3
ablations (no_hamiltonian, classical_attention, random_edges) now train
directly at fixed L=3, identical protocol, identical epoch budget (25).**
This is reported as a FINDING (not a silent scope cut): the mitigation
BRIEF.md prescribed for C4 was tested and found unneeded here, so it is
listed in Future Work as an untested variant rather than a component of
the reported results. The practical benefit is a fully matched training
protocol across CQGT and every ablation (zero growth-schedule confound in
the comparison), and roughly half the compute of running growth on all 4
model variants.

`configs/main.yaml`: `layer_growth: [1, 2, 3]` -> `[3]` (with the old value
kept in a comment for the record), `epochs_per_stage: 25` added.
`scripts/run_gate2.py`: `layer_schedule=(1,2,3)` -> `(3,)`,
`EPOCHS_PER_STAGE` 8->25, `n_mc_train` 4->None (full M=20 for training, not
just evaluation, matching the run that produced the AUPRC 0.3374 result).

### Environment defect found and fixed: xgboost/torch_geometric import order (2026-08-11)

While smoke-testing the Phase 3 harness (`experiments/run_phase3.py`),
`XGBClassifier.fit()` segfaulted (exit 139) every time, reproducibly,
whenever `torch_geometric.nn` had already been imported in the same
process -- an OpenMP runtime conflict between torch_geometric's bundled
libomp and xgboost's, not a code bug in either library or in this project's
training logic. Verified in isolation with a minimal repro (`import torch;
from torch_geometric.nn import GCNConv; from xgboost import XGBClassifier;
clf.fit(...)` -> segfault every time; swapping to `xgboost` first ->
works). **Fix: `experiments/run_phase3.py` imports `xgboost` before
`torch`/`torch_geometric`, with a comment recording why** -- import order
matters and must stay that way in any file that uses both libraries in the
same process.

### F2 -- 1-WL separation experiment built and run (2026-08-12), while Phase 3 sweep runs

`experiments/wl_separation.py` implements BRIEF.md Sec 3's replacement
Proposition: `C6` (single 6-cycle) vs `2xC3` (two disjoint triangles), both
2-regular, identical node features (all-ones, 4-dim), N=6 qubits/nodes.
GCN and GAT (via `baselines.gnn.GNNBaseline`, same classes used in Phase 3)
get identical weights on both graphs; CQGT gets identical parameters too
(state_dict copied from a C6-topology instance to a 2xC3-topology instance
of the same shape -- both graphs have exactly 6 edges, so `zz_params`/
hopping-term shapes match, and the ONLY difference between the two CQGT
runs is which qubit pairs the gates act on).

Required a small refactor of `cqgt/model.py`: extracted the circuit's raw
`<Z_i>` readout (forward() steps 1-4, before attention/MLP head) into its
own method `circuit_expectation_z()`, called internally by `forward()`. This
is the quantity the 1-WL claim is actually about -- comparing post-MLP-head
`p_hat` instead would let an arbitrary classical head manufacture spurious
differences and undermine the claim's rigor. All 7 `test_model.py` tests
still pass unchanged after the refactor (behavior-preserving, verified, not
assumed). Safe to edit while the Phase 3 sweep's subprocesses are running:
Python doesn't hot-reload already-imported modules, so on-disk edits don't
touch already-running processes.

**Result (`tests/test_wl_separation.py`, verified across 3 seeds):**
- GCN, GAT: `max|output(C6) - output(2xC3)| = 0.000e+00` -- exact machine
  precision, every seed. Matches the 1-WL bound exactly, as it must.
- CQGT: `max|<Z_i>(C6) - <Z_i>(2xC3)| = 4.552e-02` (seed 0), mean 2.57e-02
  -- a real, substantial, non-fragile separation (>1e-4 at every seed
  tested).

**One honest wrinkle, not smoothed over:** naive graph-automorphism
reasoning predicts ALL 6 nodes within C6 (and separately within 2xC3)
should read out identically to each other, since both graphs are
vertex-transitive. The actual per-node `<Z_i>` values within C6 are NOT
uniform (0.347, 0.281, 0.320, 0.332, 0.337, 0.379) -- because the circuit
applies hopping/RZZ gates in a fixed SEQUENTIAL order over the edge list,
and these gates do not all commute (adjacent edges share a qubit), so the
Trotter-like sequential implementation is not exactly symmetric under the
graph's automorphism group even though the underlying Hamiltonian is. This
does not weaken the 1-WL separation claim (GCN/GAT are still exactly
degenerate; CQGT is still clearly distinguishable) but it means the correct
explanation in the paper is "sensitive to graph topology, both spectrally
and through the sequential gate-ordering effects of the Trotterized
implementation" -- not a claim that CQGT recovers per-node automorphism
symmetry when the graph has it. Stated plainly rather than papered over.

Figure `figures/F2_wl_separation.pdf` (300 dpi, Okabe-Ito colorblind-safe
palette): grouped bar chart, `|output(C6) - output(2xC3)|` per node, GCN/GAT
bars at exact zero (annotated "0"), CQGT bars clearly nonzero.

### Phase 4 work started in parallel with the Phase 3 sweep (2026-08-12)

Per explicit instruction: while the 2-seed Phase 3 sweep runs in the
background (subprocesses `run_phase3.py --seed 0` / `--seed 1`, isolated
`_xgb_worker.py` per seed), did the results-independent Phase 4 work that
doesn't need T1/T2/T3 to exist first, since it's near-zero CPU and won't
contend with the sweep.

**`paper/main.tex` did not exist before this session** -- BRIEF.md Sec 5
is phrased as edits to an existing draft ("replace the Proposition",
"rewrite the data statement"), but no `paper/` directory or `.tex` file
was ever created in earlier phases. Built a full IEEEtran skeleton
(`paper/main.tex` + `paper/references.bib`) from scratch, structured
around BRIEF.md's own subsection labels (II-A/II-C, III-A/III-E/III-F,
IV-A/IV-C, V-Limitations), so the section-editing instructions apply
against something real going forward. No LaTeX toolchain is installed in
this environment to compile-check it (`pdflatex` not found) --
brace-balance and `\begin`/`\end` nesting were verified with a small
Python stack-based scan instead (all matched); an actual `pdflatex` /
`bibtex` compile has NOT been run and should be the first thing checked
once a LaTeX installation is available, before trusting this renders
cleanly.

Populated now (results-independent): II-A's symmetrized-Laplacian
limitation; II-C's explicit N-qubit / hopping-Hamiltonian / CTQW
single-excitation-subspace encoding declaration; III-A's concrete
$C_t = \sum_k \lambda_k^t \sum_i \beta_{ik}$; III-E's replacement 1-WL
Proposition (proof sketch + the real F2 numbers, including the honest
per-node-asymmetry wrinkle -- see the F2 section above); III-F's
parameter-shift-as-hardware-path / backprop-is-mathematically-identical
framing; IV-A's real data statement (4 anchors not 5, dual reconstruction,
EN cascade, disclosed interpolation); IV-C's numeric counterfactual
metric definition (Spearman rho + P@10 vs. GCN gradient-saliency); and a
full Limitations subsection covering all 8 items from the user's list
(N<=12, noiseless-only, reconstructed exposures, 4-anchor interpolated
panel, symmetrized Laplacian, 2 seeds, maxent degeneracy, untested
layer-growth) plus a Future Work subsection for everything explicitly cut
this session (full 3-variant sweep, the 4 dropped ablations, layer-growth
test, magnetic Laplacian, real hardware).

Left empty, as instructed: Section IV's three results tables (T1/T2/T3,
currently `\PLACEHOLDER{}` stubs with column headers only) and the
Conclusions subsection. Section IV's narrative prose IS drafted, with
`\PLACEHOLDER{}`-marked claims to fill in once real numbers exist (does
CQGT beat the best classical baseline; which ablation drops AUPRC most;
does CQGT beat GCN gradient-saliency on the counterfactual metric; is
any $\Delta R_{ij}$ non-monotone per Acemoglu et al. -- flagged as a
headline finding if so, not an anomaly). No number anywhere in the file
is invented; every `\PLACEHOLDER{}` names exactly which `results/*.csv`
file will fill it.

### GATE 3 — RESULTS (2026-08-12), honest numbers, reported as they came out

Both seed subprocesses completed cleanly, no crashes: seed 0 in 198.4 min,
seed 1 in 191.9 min (under the conservative 5.6-6.0h/seed projection --
2-way concurrency saw less contention than the 4-way benchmark used for
that projection, consistent with the caveat given at launch).
`results/T1.csv`, `T2.csv`, `T3.csv` (+ `_raw.csv` per-seed detail) exist
and are populated. **Per BRIEF.md's own rule, the numbers below are
reported as they are, not tuned to look better.**

**Scope actually run** (both compute-budget-driven cuts, disclosed):
mindensity dataset only, 2 seeds (not 3 -- see the seed-count decision
above), 5 baselines + CQGT full model (T1), 3 ablations + full_model (T2),
CQGT vs. GCN-gradient-saliency counterfactual comparison pooled over the
first 3 test snapshots (T3).

**T1 — main comparison (mean AUPRC, spillover-only, prevalence floor
≈0.024, n=2 seeds):**

| Model | AUPRC (mean) | seed 0 | seed 1 | AUROC (mean) | params |
|---|---|---|---|---|---|
| gcn | **0.2112** | 0.1462 | 0.2762 | 0.895 | 1345 |
| cqgt_full | 0.1439 | 0.2095 | 0.0782 | 0.819 | 386 |
| gat | 0.1028 | 0.1095 | 0.0962 | 0.863 | 3105 |
| xgboost | 0.0898 | 0.0953 | 0.0843 | 0.862 | 3932 |
| logreg | 0.0777 | 0.0790 | 0.0764 | 0.842 | 8 |
| generic_vqc | 0.0691 | 0.0686 | 0.0696 | 0.786 | 138 |

**GCN has the highest mean AUPRC, not CQGT.** CQGT does not clearly beat
the strongest classical baseline: CQGT's own seed-to-seed swing (0.078 to
0.210, a >2.5x range) is larger than its gap to GCN, and the two models'
95% "CIs" (computed at n=2, see caveat below) overlap heavily. Every model
clears the prevalence floor by a wide margin except generic_vqc, which
barely does (0.069 vs. 0.024) -- consistent with a topology-blind quantum
circuit having little to work with. CQGT does clearly beat generic_vqc,
which is a real, if narrower, finding: encoding the network structure
into the circuit beats not encoding it at all.

**T2 — ablations (mean AUPRC, mindensity, n=2 seeds):**

| Model | AUPRC (mean) | seed 0 | seed 1 |
|---|---|---|---|
| no_hamiltonian | **0.1873** | 0.2595 | 0.1151 |
| classical_attention | 0.1509 | 0.2138 | 0.0881 |
| full_model | 0.1439 | 0.2095 | 0.0782 |
| random_edges | 0.1321 | 0.1901 | 0.0741 |

**This is the uncomfortable finding that must not be buried:** removing
the Hamiltonian term entirely (`no_hamiltonian`) and replacing quantum
attention with classical dot-product attention (`classical_attention`)
BOTH outperform the full model on mean AUPRC, at both seeds
individually, not just on average. Neither of CQGT's two
quantum-mechanism-specific components (the network-contagion Hamiltonian,
which BRIEF.md Sec 1 calls "the paper's central claim," and the quantum
fidelity attention) demonstrates a positive contribution in this run. The
one ablation that moves in the theoretically expected direction is
`random_edges`, which underperforms the full model at both seeds (0.132 <
0.144) -- weak evidence that using the *real* topology matters, but no
evidence that encoding it via a Hamiltonian or attending to it via a
quantum circuit specifically matters, versus the topology-matched RZZ
variational layers alone.

**T3 — counterfactual attribution (Spearman rho vs. ground-truth
Eisenberg-Noe delta-R, n=33 pooled edge-snapshot pairs, n=2 seeds):**

| Model | Spearman rho (mean) | seed 0 | seed 1 | P@10 (mean) |
|---|---|---|---|---|
| cqgt_full | 0.0177 | 0.0185 | 0.0169 | 0.40 |
| gcn_gradient_saliency | -0.5624 | -0.4572 | -0.6677 | 0.45 |

Neither model demonstrates strong causal attribution. CQGT's rho is
approximately zero (no meaningful rank correlation with the true
counterfactual effect); the GCN gradient-saliency baseline's rho is
substantially *negative* at both seeds (actively anti-correlated -- its
screening scores point away from, not toward, the edges that actually
matter). CQGT's P@10 (0.40) is nominally lower than GCN's (0.45, wide
range 0.30-0.60 across seeds) despite CQGT's non-negative rho, which is
not a contradiction so much as a sign both metrics are noisy at n=33
pooled points -- this sample size is thin and the finding should be read
as "inconclusive, weakly favoring neither model being demonstrably
causal," not as a clear win for either.

**Caveat that applies to every number above: n=2 seeds is underpowered.**
Bootstrap CIs computed over 2 points are close to degenerate (they mostly
just reflect the gap between the two observed values, not a real sampling
distribution) and should not be read as rigorous uncertainty
quantification. The compute-budget decision that produced n=2 (not the
originally planned 3, itself already cut from BRIEF's larger seed budget)
is documented above and was disclosed to the user before running, not
discovered after the fact to excuse a disappointing result.

**Honest headline, stated plainly:** this run does not support the claim
that CQGT's quantum-specific mechanisms (Hamiltonian coupling, quantum
attention) outperform simpler alternatives at N=12 on this dataset and
seed budget. It does support two narrower claims: (1) encoding the real
exposure topology beats not encoding any topology (CQGT beats
generic_vqc; real edges beat random edges), and (2) neither CQGT nor a
classical GCN baseline demonstrates reliable causal counterfactual
attribution against the Eisenberg-Noe ground truth at this scale. Per
BRIEF.md's explicit rule, this is reported as the finding, not tuned
further to look better.

**Non-monotone $\Delta R_{ij}$ check (per BRIEF.md's explicit "watch for
and name this if it appears" instruction):** re-derived the ground-truth
Eisenberg-Noe $\Delta R_{ij}$ for the same 33 pooled edge-snapshot pairs
used in T3 (cheap, cascade-only, no model training) and checked signs
directly. **Found: 3/33 (9.1\%, seed 0) and 2/33 (6.1\%, seed 1) are
negative** -- since $\Delta R_{ij}=R_0-R_{cf}$, a negative value means
removing that edge would have INCREASED mean equity-loss fraction (not
decreased -- caught and fixed 2026-08-14 after the user asked to verify
the T3 sign convention; the ground-truth/GCN-saliency comparison itself
had no bug, see the dedicated section below, but this direction was
mis-stated in the first draft of this note and in the paper), i.e. that
edge was net risk-sharing, not net contagion, at that snapshot (range
across both seeds: $-0.139$ to $+0.104$). This matches Acemoglu et al.'s
(2015) theoretical mechanism and is a genuine, positive, headline finding
about the ground-truth labels themselves, independent of how any model
scored against them -- added to the paper's Results (4) rather than left
as the placeholder it started as.

### T3 sign-convention verification (2026-08-14), requested by the user

User flagged that GCN gradient-saliency's Spearman rho=-0.56 against
ground truth is too strong to dismiss as "no attribution" and asked to
verify neither side's ΔR had an accidental sign flip -- if GCN's saliency
were actually rho=+0.56 with the sign corrected, T3's conclusion changes
materially (from "neither model attributes causally" to "GCN saliency
attributes well, CQGT does not").

**Verified computationally, not just re-read algebraically:**
1. Ground-truth convention: hand-built 2-bank toy case (bank 0 shocked/
   wiped out, bank 1 holds a 50-unit claim on bank 0) where the correct
   sign is unambiguous by construction. `ground_truth_delta_r((1,0))`
   returned `+3.125` for this genuinely risk-amplifying edge (R0=9.375,
   R_cf=6.25 after removing it) -- correct, positive-for-risk-amplifying,
   as designed.
2. GCN gradient-saliency internal consistency: compared
   `_gcn_gradient_saliency_delta_r`'s first-order estimate against an
   EXACT re-forward (actually zeroing each of the 11 real edges at one
   test snapshot and re-running the trained GCN, not just trusting the
   gradient). **11/11 edges sign-agreed, Spearman(grad_est, exact) =
   0.945.** The gradient-based screening faithfully represents what the
   GCN itself believes about each edge's counterfactual effect on its own
   output -- no bug in how the estimate is computed relative to the
   model's own behavior.

**Conclusion: no sign bug on either side.** Both R0-R_cf conventions
match, and the GCN-saliency side is independently verified against an
exact re-forward. The measured rho=-0.56 is a real, substantial,
SYSTEMATIC anti-correlation between the trained GCN's own gradient-based
notion of edge importance and the true Eisenberg-Noe causal structure --
a genuine negative finding about gradient saliency as a causal-attribution
method on this architecture (plausibly related to GCNConv's degree-based
renormalization making raw-edge-weight gradients a non-monotone function
of true causal contribution, though that specific mechanism was not
directly tested), not an artifact to fix. T3's original conclusion
("neither model demonstrates reliable causal attribution -- CQGT is
directionless, GCN saliency is actively anti-correlated, which is its own
kind of not working") stands unchanged. The paper's item (4) prose and
this note's earlier direction-of-effect wording were fixed as part of
this verification, since they were caught to be wrong regardless of the
sign-bug question that motivated re-checking.

### GATE 3 status: PASS (deliverable requirement) -- results/T1.csv, T2.csv,
T3.csv exist and are populated, per-seed raw detail preserved, headline
numbers reported above exactly as produced. This is separate from, and
must not be conflated with, whether the results are favorable to CQGT --
they are mixed-to-negative on the model's central architectural claims,
reported as such per BRIEF.md's explicit rule to report null/negative
results rather than keep tuning.

### GATE 2 status: NOT YET PASSED -- blocked on the items above

GATE 2 requires: training loss decreases monotonically-ish over epochs;
CQGT beats the prevalence floor on validation; report the learning curve.
**None of this has been honestly demonstrated yet.** Timeline of attempts:
1. An ad hoc stopgap (`n_mc_train=4`, `epochs_per_stage=8`, all 3 dataset
   variants) was launched, then killed unfinished (produced zero output)
   when the compute-budget conversation started -- never used for anything.
2. A proper attempt (mindensity, full M=20, 20 epochs/stage) hit the
   swap-thrashing bug above and was killed after 2+ hours with zero output.
3. Post-fix, only short (2-3 epoch) timing runs and one in-flight 20-epoch
   convergence run have been done -- none constitute a full GATE 2 pass
   (need the actual layer-growth schedule L=1->2->3 run to completion, loss
   curve inspected, and validation AUPRC compared to prevalence).

**GATE 2 must be re-run properly, on the real epochs-per-stage budget once
determined, before it can be reported as passed.** This is explicitly
required by the user, not an oversight to skip past.
