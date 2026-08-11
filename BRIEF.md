# BRIEF.md — CQGT Build Specification

> Save this file as `BRIEF.md` in the repo root. It is the complete build spec.

---

## 0. Context and non-negotiables

Repo: `Deakin_QML`. Paper: "Causal Quantum Graph Transformers for Systemic Credit-Risk Contagion Prediction" (IEEE format, `paper/main.tex`).

Current state: ~12 KB of skeleton code across 13 files. **Nothing runs end to end.** No training loop, no labels, no metrics, no entry point.

**Constraints — these are hard:**

- **Laptop, CPU only.** No GPU, no cluster, no quantum hardware.
- **Days, not weeks.** Deliverable is an internship report plus working, reproducible code.
- **N = 12 institutions (12 qubits).** Do not exceed 14.
- **No parameter-shift training.** Use exact backprop through a PyTorch statevector simulator.
- **Never fabricate results.** If an experiment produces a null or negative result, report it as such. A well-executed null result is an acceptable and expected outcome. Fabricated numbers are a catastrophic failure of this task.

**Framing:** this is *not* a quantum-advantage paper. It is a reproducible benchmark and reference implementation for graph-informed quantum models of interbank contagion, evaluated honestly against classical baselines. Write everything in that voice.

---

## 1. Known defects in the existing code

Fix all of these. Several are silent and would corrupt results.

| File | Defect |
|---|---|
| `experiments/treewidth_entropy.py` | `_, tw = nx...treewidth_min_fill_in(...)` — return order is `(width, decomposition)`. Currently assigns the **decomposition tree** to `tw`. Every treewidth number is wrong. Must be `tw, _ = ...`. |
| `baselines/gnn.py` | `forward()` accepts `edge_weight` and **never passes it to any conv**. GNN baselines ignore exposure magnitudes while CQGT uses them — an unfair comparison biased toward CQGT. Pass `edge_weight` to `GCNConv`; use `edge_dim`/`edge_attr` for `GATConv` and `TransformerConv`. |
| `cqgt/hamiltonian.py` | Returns `H = alpha * L` and **nothing ever consumes it**. The network contagion term — the paper's central claim — is not in the model at all. |
| `cqgt/circuit.py` | `ctqw_layer` applies `RZZ`, which is **diagonal in the computational basis and transfers no amplitude**. There is no quantum walk. Also `tau0` is shared across all edges, so `W_ij` is discarded entirely. |
| `cqgt/circuit.py` | `expectation_z` contains dead code (`... if False else ...`). Remove. Use `SparsePauliOp` rather than bare strings if Qiskit is used at all. |
| everywhere | **No feature standardization.** `leverage≈10`, `tier1≈0.12`, `cds≈50`, then `arctan(50)≈1.55≈π/2`. All large-scale features saturate to the same angle; the embedding is degenerate. |
| `data/generate_exposures.py` | Fitness model with `p_edge` clipping produces a **star graph, treewidth ≈ 1–2**. The Proposition is about `tw ≥ 3`. The generator cannot produce the regime the theory is about. |
| `data/generate_exposures.py` | Produces `W` and `X` but **no labels `y`**. |
| `scripts/build_panel.py` | Regenerates the graph i.i.d. each step (`seed+t`); only `fitness` is smoothed, not `W`. **No temporal structure.** Default `n_banks=200` → 200 qubits. |
| `cqgt/counterfactual.py` | `exact_counterfactual` takes `alpha, beta, gamma, macro_*, f_weights` and uses none of them. `perturbative_screen` needs `grad_R_wrt_W`, which nothing computes. |
| `cqgt/attention.py` | Attention output `A` is never consumed. Eq. (14) is unimplemented. |
| `baselines/quantum_kernel.py` | `qiskit_machine_learning` missing from `requirements.txt`; `FidelityQuantumKernel` API is version-fragile; `dequantized_rff_surrogate` returns an untied `(rff, svm)`. |
| packaging | **No `__init__.py` anywhere.** Cross-package imports will fail. |

---

## 2. Target architecture

### 2.1 Data (`cqgt/data/`)

**Exposure networks — two sources, same interface.**

*Primary (if `data/raw/FRY15_*.csv` exist):* extract per-institution **intra-financial system assets** (`M362`) and **intra-financial system liabilities** for the top `N=12` RSSD IDs. These are the row and column marginals of the real interbank network. Reconstruct `W` two ways:

- `maxent` — iterative proportional fitting (RAS) to match marginals, zero diagonal.
- `mindensity` — Anand, Craig & von Peter (2015) minimum-density: greedily load the largest exposures onto the fewest links consistent with the marginals.

Run **both**; report every result under both. They bracket true contagion (min-density over-estimates, max-entropy under-estimates).

*Fallback (if the CSVs are absent):* **core–periphery stochastic block model** — 4 core banks (`p_core=0.9`), 8 periphery (`p_per=0.05`), core→periphery `p=0.3`; Pareto(2.5) weights on top; zero diagonal. This guarantees `tw(G) ≥ 3`. Log loudly which source was used.

**Temporal panel.** `T = 120` weekly snapshots. Persist one latent size/fitness vector, evolve it as AR(1) (`ρ=0.95`); resample only ~5% of edges per step. Inject a **crisis regime** over a defined window (`t ∈ [70, 85]`): raise shock magnitude and core density. This is what the temporal baselines must detect.

**Labels — this is the single most important component.** Labels must depend on the network topology, or the entire paper is void.

Implement an **Eisenberg–Noe clearing cascade**:
1. Assign each bank external assets and equity from its features.
2. Apply an exogenous shock to a random subset (seeded).
3. Solve the clearing payment vector by fixed-point iteration.
4. `y_i = 1` if bank *i*'s equity loss exceeds a threshold. Target base rate ≈ 8–15% — tune the shock magnitude to hit it and record the tuned value.

**Ground-truth counterfactuals — free byproduct.** For each edge, rerun the cascade with `W[i,j] = 0` and record `ΔR_true_ij`. This is the ground truth for the counterfactual metric, which is currently undefined in the paper.

**Features.** `x_i ∈ ℝ⁶`: leverage, Tier-1 ratio, CDS-proxy (or equity-vol proxy), liquidity ratio, out-degree centrality, macro sensitivity. **Z-score using training-split statistics only**, then `arctan`.

**Macro (`C_t`).** If `data/raw/fred_*.csv` exist, PCA them to `K=3` factors → `λ_k^t`. Else use a seeded synthetic AR(1) macro factor and label it as such.

### 2.2 Quantum simulator (`cqgt/qsim.py`)

Write a **PyTorch statevector simulator**, roughly 80 lines. Do not use Qiskit in the training loop.

- State: complex tensor of shape `(batch, 2, 2, ..., 2)`, N axes. `N=12` → 4096 amplitudes. Trivial on CPU.
- Apply gates by reshaping and `torch.einsum` on the target axes.
- Gates needed: `RY`, `RZ`, `RZZ`, and `RXX + RYY` (the hopping pair).
- Fully differentiable — gradients via autograd. Exact, and mathematically identical to parameter-shift.
- **Unit test:** verify against `qiskit.quantum_info.Statevector` on random small circuits, `atol=1e-6`. This test is mandatory.

### 2.3 CQGT model (`cqgt/model.py`)

**Fix the Hamiltonian.** `H_t = α L̃_t + β D(X_t) + γ C_t` must actually be applied:

- **`α L̃_t` — hopping term.** Lift the Laplacian to N qubits as `Σ_ij L̃_ij (X_iX_j + Y_iY_j)/2`. Implement per edge as `RXX(θ_ij) · RYY(θ_ij)` with `θ_ij ∝ α · L̃_ij · δ`. This conserves excitation number and reduces exactly to the CTQW on the single-excitation subspace. **Amplitude now actually moves, and `W_ij` enters per edge.** This replaces the broken `RZZ` "CTQW".
- **`β D(X_t) + γ C_t` — diagonal term.** Per-qubit `RZ(2 δ h_i)` where `h_i = β f_ψ(x_i) + γ Σ_k λ_k β_ik`. `f_ψ` is a small MLP.

**Forward pass:**
1. Embed: `RY(2·arctan(ŵᵀx_i))` per qubit.
2. CTQW encoder: hopping term with learnable `τ₀ ≥ 0`.
3. `L = 3` variational layers, each: `RY(φ_i)` per qubit → `RZZ(θ_ij)` on exposure edges only → `e^{-iH_t δ_ℓ}` (hopping + diagonal as above).
4. Measure `⟨Z_i⟩` → `r ∈ [-1,1]^N`.
5. Quantum fidelity attention on `n_a = 2` ancilla qubits: `F_ij = |⟨q_i|k_j⟩|²`, softmax with learnable `λ`. **Wire it in** — `h_j = ⟨Z_j⟩ · x_j`, `x̃_i = Σ_j A_ij h_j`.
6. MLP head on `[r ; x̃]` → `p̂ ∈ [0,1]^N`.

**Training:** Adam (`lr=0.01`), cosine annealing, grad clip 1.0, class-weighted BCE (`w⁺ = N₋/N₊`). Layer-by-layer growth `L=1→3`, each new layer initialised `N(θ*_prev, (π/10)²)`.

**Counterfactuals:** Phase 1 screening is now `torch.autograd.grad(R, W)` — one line. Phase 2: exact re-forward for top-K (`K=10`).

### 2.4 Baselines (`baselines/`)

Logistic regression, XGBoost, GCN, GAT, GraphTransformer, generic VQC (hardware-efficient, same qubit count), topology-matched VQC **without** the Hamiltonian, and the RFF dequantization surrogate. Quantum-kernel SVM only if `qiskit-machine-learning` installs cleanly — skip and note if not.

**All GNNs must receive edge weights.** **Report parameter counts** for every model in the results table — otherwise any gain is attributable to capacity, not architecture.

### 2.5 Ablations (`experiments/`)

Seven, all on the primary dataset: `no_hamiltonian`, `classical_attention`, `random_quantum_attention`, `random_edges` (matched density), `hardware_efficient_ansatz`, `no_macro_coupling`, `rff_dequantized`.

Ablation `rff_dequantized` is the most important one — it is what makes the paper honest. Do not drop it.

### 2.6 Metrics (`cqgt/metrics.py`)

AUPRC (**primary**), AUROC, F1 at Youden's J, Brier, ECE (10 bins), Kendall-τ, NDCG@5, plus **counterfactual metrics**: Spearman ρ and Precision@10 of predicted `ΔR` against `ΔR_true`.

Also report the **prevalence** (the trivial AUPRC floor) in every table.

### 2.7 Protocol

- **Strict rolling temporal splits.** Assert `max(train_t) < min(val_t) < min(test_t)` **in code**, not just in prose. No feature may use future information.
- **5 seeds** (0–4). Report mean ± 95% bootstrap CI.
- Hyperparameters selected on validation only.
- Config in YAML; everything reproducible from `python -m experiments.run_all --config configs/main.yaml --seed S`.
- Persist raw per-seed results to `results/*.csv`.

---

## 3. Theory task (small, high value)

The Proposition in §III-E is not sound as written: `δ = Ω(k·2^{-O(k)})` **shrinks exponentially in k**, and "entanglement entropy exceeds efficient classical simulation" is a claim about simulation cost, not expressive power or learnability.

Replace with a claim that is provable in three lines and demonstrable numerically:

> **Proposition (1-WL separation).** There exist non-isomorphic graphs `G, G′` with identical node features that any 1-WL-bounded message-passing GNN maps to identical outputs, but for which the topology-matched CQGT circuit yields distinguishable `⟨Z_i⟩`.

Implement `experiments/wl_separation.py`: build `C₆` and `2×C₃` (both 2-regular, 1-WL-indistinguishable), with identical node features. Show GCN/GAT outputs are identical to numerical precision; show CQGT `⟨Z_i⟩` differ. Produce this as a figure.

Keep the entanglement-entropy vs. treewidth study as a **separate empirical measurement**, not as a step in a proof.

---

## 4. Results contract

Produce exactly these artefacts. Nothing else.

**Tables** (`results/`, CSV + LaTeX):
- **T1** — main comparison: all baselines + CQGT × 9 metrics, 5 seeds, mean ± 95% CI, **with parameter counts and prevalence**.
- **T2** — ablations: 7 ablations + full model, same format.
- **T3** — counterfactual attribution: Spearman ρ and P@10 vs. ground truth, CQGT vs. **gradient-saliency baseline on the GCN**. Without that baseline the causal module is unevaluated.

**Figures** (`figures/`, PDF, 300 dpi):
- **F1** — AUPRC bar chart with 95% CI, all models.
- **F2** — the C₆ vs 2×C₃ separation demo.
- **F3** — entanglement entropy vs. treewidth scatter, with Pearson r and p-value.
- **F4** — quantum attention heatmap for one crisis snapshot, transmitters/receivers labelled.
- **F5** — top-K SIEC attribution vs. cascade ground truth.

**Watch for and name this if it appears:** Acemoglu et al. (2015) show that removing an interbank link can *increase* systemic risk by destroying risk-sharing, so `ΔR_ij < 0` is expected for some edges. If `ΔR` is non-monotone, make it a headline finding — it demonstrates the model learned real structure rather than "more edges = more risk."

---

## 5. Paper edits (`paper/main.tex`)

1. **§III-E** — replace the Proposition with the 1-WL version. Delete the entanglement-entropy step from the proof.
2. **§II-C** — declare the encoding explicitly: N qubits, `L̃` lifted as a hopping Hamiltonian, CTQW recovered on the single-excitation subspace. The paper currently conflates an N-dim walk with a 2^N-dim circuit.
3. **§II-A** — symmetrising `L̃` discards direction, and in interbank contagion direction is the mechanism. State this plainly as a limitation (a magnetic Laplacian would fix it — future work).
4. **§III-A** — make `C_t` concrete: `C_t = Σ_k λ_k^t Σ_i β_ik X_i` (macro stress as a transverse field per bank, weighted by factor loading).
5. **§III-F** — keep the parameter-shift equation as the hardware-executable path; state that simulation uses exact backprop/adjoint differentiation, which is mathematically identical.
6. **§IV-A** — rewrite the data statement to match what was actually built. If FR Y-15 was used:
   > *Bilateral exposures are not publicly observable. We reconstruct them from real FR Y-15 intra-financial-system asset and liability marginals for N US bank holding companies using both maximum-entropy and minimum-density reconstruction, which respectively under- and over-estimate contagion and therefore bracket the true network (Anand et al., 2015). Distress labels and ground-truth causal attributions are generated by an Eisenberg–Noe clearing cascade over the reconstructed networks; all results are reported under both reconstructions.*
7. **§II-A** — if CDS data was unavailable, **change the label definition** to whatever was actually implemented. Do not leave an unreproducible CDS-threshold definition in the text.
8. **§IV-C** — define counterfactual intervention accuracy numerically (Spearman ρ + P@K vs. ground truth).
9. **§IV** — populate with real numbers from `results/`.
10. **§V** — write real conclusions, **including negative ones**. State limitations: N≤12, noiseless simulation only, reconstructed (not observed) exposures, symmetrised Laplacian, single-machine compute.
11. Add citations: Eisenberg–Noe (already `[eisenberg2001]`), Anand/Craig/von Peter, Acemoglu (already `[acemoglu2015]`).

---

## 6. Execution phases and gates

Work in order. **Stop at each gate, report, and wait for my confirmation before continuing.**

**Phase 0 — Foundation.** Add `__init__.py` files, `pyproject.toml`, pinned `requirements.txt`, YAML config, seeding utility, `results/` and `figures/` dirs. Fix every defect in §1. Write the qsim unit test against Qiskit.
→ **GATE 0:** `pytest` passes, including the qsim-vs-Qiskit equivalence test.

**Phase 1 — Data.** Loader, reconstruction (both methods), core–periphery fallback, temporal panel with crisis window, Eisenberg–Noe cascade, labels, ground-truth `ΔR`, feature standardisation, treewidth measurement.
→ **GATE 1 (critical):** report label base rate, measured treewidth per snapshot, and AUPRC for logistic-regression-on-features vs. a quick GCN. **The GCN must beat logistic regression by a clear margin.** If it does not, the labels do not depend on the network and everything downstream is meaningless — stop and report, do not proceed.

**Phase 2 — Model.** PyTorch simulator, CQGT with the corrected Hamiltonian, attention wired in, MLP head, training loop with layer-by-layer growth.
→ **GATE 2:** training loss decreases monotonically-ish over epochs; CQGT beats the prevalence floor on validation. Report the learning curve.

**Phase 3 — Experiments.** Baselines, ablations, counterfactual module, all metrics, 5 seeds, rolling splits, results persisted.
→ **GATE 3:** `results/T1.csv`, `T2.csv`, `T3.csv` exist and are populated. Report the headline numbers honestly, whatever they are.

**Phase 4 — Figures and paper.** F1–F5, the WL separation experiment, all §5 paper edits, README with reproduction instructions and a prominent data-provenance disclaimer.
→ **GATE 4:** `make reproduce` runs clean from a fresh checkout.

---

## 7. Rules

- **Never invent numbers.** Every figure in the paper must trace to a file in `results/`.
- If something cannot be done in the time available, **say so and cut it explicitly** rather than stubbing it and implying it ran.
- If a result is negative or null, **report it as the finding.** Do not tune until it looks good.
- Prefer working code over complete code. A running 6-baseline comparison beats a broken 13-baseline one.
- Keep a `NOTES.md` log of every decision, deviation from this brief, and known limitation. This becomes the report's methodology appendix.
- Commit at each gate with a clear message.
