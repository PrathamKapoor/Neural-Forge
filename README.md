# NeuroForge

An experimental, reproducible framework for studying **adaptive allocation of heterogeneous neural computation**. Its question is not whether MLPs, GNNs, and attention can coexist; it is whether a learned system can choose useful computation per input under an explicit resource budget.

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
