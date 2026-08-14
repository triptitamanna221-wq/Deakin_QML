# CQGT — Causal Quantum Graph Transformers for Systemic Credit-Risk Contagion

CQGT lifts a bank exposure network's Laplacian directly into a parameterized
quantum circuit's Hamiltonian (a continuous-time-quantum-walk-derived hopping
term, plus a macro-stress transverse field and quantum-fidelity attention),
trained by exact backpropagation through a custom PyTorch statevector
simulator. **This is not a quantum-advantage paper** — it is a reproducible
benchmark and reference implementation for graph-informed quantum models of
interbank contagion, evaluated honestly against classical and generic-quantum
baselines on real FR Y-15 exposure data. Full build log, every deviation from
plan, and every defect found and fixed: [`NOTES.md`](NOTES.md). Full spec:
[`BRIEF.md`](BRIEF.md). Paper draft: [`paper/main.tex`](paper/main.tex).

## Limitations, up front

- **N = 12 institutions.** Hard cap from CPU-only statevector simulation and
  from requiring complete FR Y-15 coverage across all four real anchor years.
- **n = 2 seeds**, not the originally planned 3–10 — an explicit,
  disclosed compute-budget cut on single-machine CPU-only hardware.
  Reported confidence intervals are correspondingly wide and
  under-powered; read every comparison below as suggestive, not resolved.
- **Noiseless simulation only.** No result here ran on real quantum
  hardware. Training uses exact backprop, verified mathematically
  identical to the hardware-executable parameter-shift rule in this
  noiseless setting, but nothing was run on a device.
- **Minimum-density reconstruction only** for the main sweep (of the two
  reconstructions built and validated — see Data below).

## Headline findings — including the negative ones

This project's rule throughout: never invent numbers, never tune until a
null or negative result looks better. What actually came out:

- **CQGT does not clearly beat a plain GCN baseline.** Mean spillover-subset
  AUPRC: GCN 0.211 vs. CQGT 0.144 (prevalence floor ≈ 0.024). CQGT's own
  seed-to-seed swing (0.078–0.210) is larger than its gap to GCN.
- **The ablation study contradicts the model's central architectural
  claim.** Removing the network-contagion Hamiltonian entirely
  (`no_hamiltonian`, AUPRC 0.187) and replacing quantum attention with a
  classical equivalent (`classical_attention`, 0.151) both **outperform**
  the full model (0.144), at both seeds individually. Only `random_edges`
  (0.132) underperforms as theoretically expected — real topology beats a
  random graph, but neither of CQGT's quantum-specific mechanisms
  demonstrates a positive contribution in this run.
- **CQGT does clearly beat a topology-blind generic VQC** of the same qubit
  count (0.144 vs. 0.069) — encoding *some* network structure beats
  encoding none.
- **Neither model attributes causally.** CQGT's predicted counterfactual
  Δ*R* has ~zero rank correlation with Eisenberg–Noe ground truth
  (Spearman ρ ≈ 0.018); a GCN gradient-saliency baseline is *systematically
  anti-correlated* with it (ρ ≈ −0.56). Verified this isn't a sign-convention
  bug on either side (toy-case check on the ground truth, exact-re-forward
  check on the GCN saliency estimate) — the anti-correlation is a real
  finding about gradient saliency on this architecture, not an artifact.
- **A provable 1-Weisfeiler-Lehman separation, replacing an earlier unsound
  entanglement-entropy expressivity claim.** On $C_6$ vs. $2{\times}C_3$
  (both 2-regular, 1-WL-indistinguishable, identical node features): GCN
  and GAT outputs are identical to *exact machine precision*
  (`max|Δ| = 0.000e+00`); CQGT's ⟨*Z*ᵢ⟩ differ substantially
  (`max|Δ| = 4.55e-2`). See `figures/F2_wl_separation.pdf`.
- **Non-monotone Δ*R*, as Acemoglu et al. (2015) predict.** 9.1% / 6.1%
  (seed 0 / seed 1) of ground-truth edge Δ*R* values are negative — removing
  that edge *increases* systemic risk, i.e. the edge was net risk-sharing.
  A real structural feature of the labels, not an anomaly.
- **Maximum-entropy reconstruction is structurally degenerate at N=12.**
  IPF/RAS applied to strictly positive marginals cannot produce an exact
  zero, so the maxent network is a *complete graph* (density 1.0, constant
  treewidth N−1) at every snapshot — carries no exploitable topology, and
  is excluded from the treewidth-vs-entanglement-entropy study for that
  reason.

Full numbers, per-seed breakdowns, and the reasoning behind every one of
these: `NOTES.md` (search "GATE 3"), `results/T{1,2,3}.csv`, and
`paper/main.tex` §IV.

## Data provenance

Bilateral exposures are not publicly observable. They are reconstructed from
**4 real annual FR Y-15 cross-sections** (year-end 2020–2023 intra-financial-
system asset/liability marginals for the 12 largest US G-SIBs with complete
coverage; 2024 excluded because that year's export structurally omits the
required field), reconstructed **two ways** — maximum-entropy (IPF) and
minimum-density (Anand, Craig & von Peter 2015) — which respectively under-
and over-estimate contagion and bracket the true network. Distress labels
and ground-truth causal (Δ*R*) attributions come from an **Eisenberg–Noe
(2001) clearing cascade** over the reconstructed networks: each institution
independently shocked (Bernoulli, tuned to an 8–15% base rate), full
external-asset wipeout, M=20 Monte Carlo shock draws per snapshot.

**Temporal structure is interpolated, not observed.** The panel has T=120
weekly snapshots, but only **4 of them are real** (the annual anchors);
every other week's network and marginals are a disclosed linear
interpolation between anchors with AR(1) multiplicative noise overlaid
(tripled variance in a crisis window). Macro factors are real FRED series
values looked up against a synthetic weekly calendar. This is real annual
cross-sectional data with a synthetic weekly overlay — not a real weekly
panel — and is stated as such everywhere it's used.

Raw source files are committed under `data/raw/` (public-domain US
government / Federal Reserve data, ~950 KB) so a fresh clone reproduces
everything with no manual download step.

## Reproduction

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# GATE 0 — full test suite, including the qsim-vs-Qiskit equivalence test
pytest -q

# GATE 1 — label base rate, treewidth, logreg-vs-GCN diagnostics
python3 scripts/run_gate1.py
python3 scripts/run_gate1_mc.py       # refined M=20 MC / spillover-only version

# GATE 2 — CQGT training loss curve + validation AUPRC vs. prevalence floor
python3 scripts/run_gate2.py

# F2 — 1-WL separation experiment + figure
python3 -m experiments.wl_separation

# Phase 3 — main sweep (mindensity, 5 baselines, CQGT, 3 ablations, 2 seeds).
# xgboost runs in its own subprocess per seed (see NOTES.md — xgboost and
# torch cannot safely share a process on macOS with this OpenMP setup, in
# EITHER import order); OMP_NUM_THREADS=1 lets seeds run concurrently
# without oversubscribing the CPU.
for s in 0 1; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m experiments.run_phase3 \
    --seed $s > results/_seed${s}_log.txt 2>&1 &
done
wait
python3 -m experiments.run_phase3 --aggregate-only --seeds 0 1
```

Expect the Phase 3 sweep to take several hours on CPU-only hardware (see
`NOTES.md`'s wall-clock measurements). Results land in `results/T1.csv`,
`T2.csv`, `T3.csv` (aggregated) with full per-seed detail in the
`*_raw*.csv` files and per-seed console logs in `results/_seed*_log.txt`.

## Repository layout

- `cqgt/` — quantum simulator (`qsim.py`), CQGT model, data pipeline
  (FR Y-15 loading, reconstruction, Eisenberg–Noe cascade, features),
  metrics, training loop.
- `baselines/` — classical (logreg/XGBoost), GNN (GCN/GAT/GraphSAGE/
  GraphTransformer), and generic-VQC baselines.
- `experiments/` — the Phase 3 sweep runner, the 1-WL separation
  experiment, ablation wiring, treewidth/entanglement-entropy study.
- `scripts/` — gate-check entry points (`run_gate1*.py`, `run_gate2*.py`).
- `tests/` — unit tests, including the qsim-vs-Qiskit numerical
  equivalence test that gates every training run in this repo.
- `results/`, `figures/`, `paper/` — generated artefacts; nothing in
  `results/` or `figures/` is hand-edited.
