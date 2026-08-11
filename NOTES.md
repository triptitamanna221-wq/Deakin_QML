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
