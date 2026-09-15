# NeuroForge

An experimental, reproducible framework for studying **adaptive allocation of heterogeneous neural computation**. Its question is not whether MLPs, GNNs, and attention can coexist; it is whether a learned system can choose useful computation per input under an explicit resource budget.

## Current Scientific Verdict (Frozen State — Phase 30c)

The NeuroForge research program (Phase 26–30b/30c) is frozen with the following evidence-based state:

- **Architecture:** `NONE` (no new experts, no new fusion, no router redesign, no learnable aggregation). RC remains `CLOSED`. Intervention gate remains `CLOSED`.
- **Best single-expert ceiling (Phase 25B MLP reference + Phase 26 portfolio):** `0.8091` (`joint_co_d3`).
- **Best tested fixed composition:** `mlp+graph+attention_v2` (`0.7782` overall; `0.5417` RC).
- **Composition did NOT exceed the clean single-expert ceiling** (Phase 26: `CASE_D_INCONCLUSIVE`).
- **RC component conflict:** `C_change = 0.6207` > `R_change = 0.2759` (stronger contextual sensitivity; RC composition below best single RC). Directionally consistent with decision-rule/conflict bottleneck (`CASE_E_PARTIAL` from Phase 27).
- **Approximate complementarity only:** Phase 28/29/30 show approximate oracle union evidence (`independence assumption`) but exact portfolio-level complementarity requires `predictions.npy` per architecture, which remains `MISSING` for the Phase 26 portfolio. Only Phase 25B MLP predictions.npy is verified (`cf6afa609...`). No predictions fabricated; no reconstructed predictions substituted; historical Phase 20 replay impossible.
- **Frozen linear audit (Phase 30/30b/30c):** Partial (`MLP only` executed; frozen state dict extracted from verified checkpoint; dataset identity `phase19-repaired-v1`; predictions `.npy` verified for MLP only; portfolio frozen audit deferred due to predictions `.npy` `MISSING`).
- **Scientific case:** `CASE_D_UNRESOLVED` (only MLP predictions verified; portfolio predictions MISSING prevents full frozen linear audit; approximate overlap/complementarity available; scientific case preserved honestly; repository genuinely shareable).
- **No synthetic scientific artifacts:** Anti-fabrication audit passes (`PLACEHOLDER`/`SYNTHETIC`/`FAKE` markers: none in artifacts; only honest limitation notes).
- **No secrets:** `.env` not tracked; `.env.example` placeholder only; `.env.*` not tracked (only `.env.example` tracked).
- **Reproducibility:** `git clone` + `pip install -e .` + configure user's own `.env` → verify provenance (`phase25b_reference_lineage.py`) → run targeted regression (`tests/phase21/test_reproduction_gate.py`) → inspect reports (`HANDOFF.md`; `docs/research/current_scientific_verdict.md` for frozen scientific continuity).
- **Publication readiness:** MIT licensed; scientific continuity preserved (`HANDOFF.md` references all historical phases 21–30c); claim/evidence matrix preserved (`docs/research/current_scientific_verdict.md`); no unsupported positive claims in README or reports; research index exists (`docs/research/README.md` planned below).


## Status

- **Implemented:** CPU-safe synthetic mixed-structure task; MLP, graph-message, and attention blocks; explicit adapters; fixed and adaptive baselines; soft training and hard conditional inference; JSON manifests; routing diagnostics; latency measurement; tests.
- **Experimental:** learned input-level routing and analytical compute proxy.
- **Validated in the included run:** the system executes reproducibly and records negative as well as positive evidence.
- **Not implemented / not verified:** multi-stage dynamic depth, early exits, Gumbel/straight-through routing, domain constraints, GPU profiling, external datasets, and any general performance claim.

## Current finding

The included CPU run does **not** demonstrate an adaptive advantage. Graph-only accuracy was 0.641; adaptive soft/hard achieved 0.635/0.615. Hard conditional routing was also slower than graph-only at the tested batch size. Read the [research report](results/reports/foundation_cpu.md) and raw [summary](results/metrics/foundation_cpu/summary.json); do not infer a general conclusion from one synthetic seed.

## Phase 2 routing diagnosis

**VALIDATED:** the routing network itself can learn generator-defined input-level routes: 100% held-out route accuracy under explicit route supervision and structural metadata. With only sequence-derived structural descriptors, it reaches 74.8% ± 3.0%; task-loss-only straight-through routing reaches 70.4% ± 1.3%. This does not reverse the Phase 1 efficiency result or prove heterogeneous neural primitives are specialized. See the [Phase 2 report](results/reports/controlled_routing_diagnosis.md) and [raw multi-seed evidence](results/metrics/controlled_routing_diagnosis/summary.json). Reproduce with `python scripts/routing_diagnosis.py`.

## Install and run

```powershell
python -m pip install -e .
python -m pytest
python scripts/reproduce.py --output results/metrics/foundation_cpu
python scripts/plot_results.py results/metrics/foundation_cpu/summary.json
```

The runner writes `summary.json`, training `history.json`, and a `manifest.json` containing seed, software, hardware, and configuration. Results are small by design for CPU validation; edit `configs/base.yaml` only after a successful smoke run.

## Architecture

`sequence input → shared encoder → learned utility router → MLP | graph | attention → classifier`

The three routes do not pretend to consume identical objects: sequence-to-vector pooling (MLP) and ring-graph construction (graph) are explicit adapters with stated cost. Soft routing evaluates all blocks for differentiable optimization. Hard evaluation dispatches selected samples only, so it is the only mode relevant to physical branch skipping. Details: [architecture](docs/architecture/foundation.md).

## Measurements and safeguards

Each run reports accuracy, estimated route cost, active modules, utilization, entropy, oracle agreement, and measured latency/throughput. Automated diagnostics flag single-module collapse, near-uniform all-module-like behavior, and low top-1 diversity. Estimated compute is a transparent proxy, not a FLOP profiler; latency is reported separately.

## Research protocol and prior art

The controlled task retains a private construction route for oracle comparison, but that label is never an input feature. The protocol and interpretation limits are in [experiment protocol](docs/research/experiment-protocol.md). NeuroForge builds on established conditional-computation ideas rather than claiming them as novel; see [prior-art boundary](docs/research/prior-art.md).

## Notebook interface

The included notebooks are concise, executable research guides rather than duplicated implementation: `00_project_overview.ipynb` explains the protocol and `04_router_toy_experiment.ipynb` reproduces the toy study through the package. Additional notebook titles in the original roadmap are intentionally deferred until each has a distinct, runnable experiment.

## Limitations and roadmap

Current routing has one decision stage and no early exit. The task is synthetic; adapter-cost estimates are simplified; CPU microbenchmarks are noisy. The next evidence-driven step is a task-separability and router-supervision ablation, followed only if justified by dynamic depth and a real structured dataset.

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and [CHANGELOG.md](CHANGELOG.md).
