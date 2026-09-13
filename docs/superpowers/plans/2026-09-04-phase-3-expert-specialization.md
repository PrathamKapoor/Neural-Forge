# Phase 3 Expert Specialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible capacity-matched cross-evaluation of existing NeuroForge primitives.

**Architecture:** Add deterministic datasets and standalone wrappers around the existing blocks, then use a common runner for capacity selection, training, controls, measurement, serialization, plots, report, and notebook. Existing Phase 1/2 modules and artifacts are untouched.

**Tech Stack:** Python 3.13, PyTorch, pytest, matplotlib, PyYAML.

**Spec:** `docs/superpowers/specs/2026-09-04-phase-3-expert-specialization-design.md`

## Global Constraints

- Use existing `MLPBlock`, `GraphBlock`, and `AttentionBlock` without semantic changes.
- Select capacity solely by deterministic parameter-count search; never use test performance.
- Evaluate all nine architecture/family cells across three seeds with one shared protocol.
- Preserve all Phase 1/2 artifacts and record negative or invalidating controls honestly.

---

### Task 1: Benchmark datasets and structural controls

**Files:** Create `src/neuroforge/datasets/expert_specialization.py`; modify `src/neuroforge/datasets/__init__.py`; create `tests/unit/test_expert_specialization_dataset.py`.

**Interfaces:** Produce `ExpertSpecializationDataset(family, split, samples, seed)`, returning feature/target dictionaries; `apply_structural_control(features, family)` returns controlled tensors.

- [ ] Write deterministic/balance/control tests and run them to observe missing-import failures.
- [ ] Implement generators and controls without explicit family metadata.
- [ ] Re-run the dataset tests and confirm they pass.

### Task 2: Standalone experts and capacity search

**Files:** Create `src/neuroforge/models/specialists.py`; modify `src/neuroforge/models/__init__.py`; create `src/neuroforge/evaluation/specialization.py`; create `tests/unit/test_specialists.py`.

**Interfaces:** `StandaloneSpecialist(architecture, input_dim, hidden_dim, depth, num_classes)` returns logits; `select_capacity(...)` returns deterministic parameter-matched depths and counts; metric helpers return ranks/margins/stability.

- [ ] Write failing construction, shape, parameter-count, deterministic-search, and metric tests.
- [ ] Implement wrappers using existing primitive blocks and the deterministic depth search.
- [ ] Re-run these tests and confirm they pass.

### Task 3: Common experiment, instrumentation, and serialization

**Files:** Create `src/neuroforge/training/expert_specialization.py`; modify `src/neuroforge/training/__init__.py`; create `configs/phase3.yaml`; create `scripts/expert_specialization.py`; create `tests/integration/test_expert_specialization.py`.

**Interfaces:** `run_expert_specialization(config, output_dir)` writes `summary.json`, `per_seed.json`, `history.json`, `manifest.json`, and `source_data.csv`.

- [ ] Write a smoke test that expects all nine cells and serialized evidence, then observe its failure.
- [ ] Implement shared training, FLOP/activation estimates, warmed latency, memory, shortcuts, controls, and conditional oracle gate.
- [ ] Re-run smoke and unit tests.

### Task 4: Analysis artifacts and documentation

**Files:** Create `src/neuroforge/visualization/expert_specialization.py`, `notebooks/17_capacity_matched_expert_specialization.ipynb`, `results/reports/phase3_expert_specialization.md`; modify `docs/research/log.md` and visualization exports.

**Interfaces:** plotting function reads serialized Phase 3 summary and produces the required research figures.

- [ ] Add a failing plot-generation test.
- [ ] Implement plots, execute experiment, generate report/notebook/log from actual serialized results.
- [ ] Run `python -m pytest -v`; inspect generated report and figures.
