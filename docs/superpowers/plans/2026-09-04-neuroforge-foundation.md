# NeuroForge Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a reproducible adaptive heterogeneous-computation experiment foundation.

**Architecture:** A deterministic synthetic mixed-structure dataset drives fixed and adaptive models built from an MLP, attention, and graph-message block. Routing is soft during training and hard during measured inference; results include explicit routing diagnostics and real latency.

**Tech Stack:** Python 3.13, PyTorch 2.13, PyYAML, pytest, matplotlib (optional for plots).

**Spec:** `docs/superpowers/specs/2026-09-04-neuroforge-foundation-design.md`

## Global Constraints

- Support CPU-only PyTorch execution.
- Keep all experiments deterministic when a seed is supplied.
- Do not make unverified performance claims.
- Include adapter overhead in estimated computation.

---

### Task 1: Reproducible package and configuration

**Files:** `pyproject.toml`, `src/neuroforge/config.py`, `tests/unit/test_config.py`

- [ ] Write failing tests for defaults and invalid configurations.
- [ ] Implement typed configuration loading and validation.
- [ ] Run focused tests.

### Task 2: Dataset and representation contracts

**Files:** `src/neuroforge/datasets/mixed.py`, `src/neuroforge/core/representations.py`, `tests/unit/test_dataset.py`

- [ ] Write failing deterministic-data and shape-contract tests.
- [ ] Implement the controlled mixed task and explicit adapters.
- [ ] Run focused tests.

### Task 3: Blocks, router, and diagnostics

**Files:** `src/neuroforge/blocks/*`, `src/neuroforge/routing/*`, `tests/unit/test_router.py`

- [ ] Write failing routing-normalization and collapse-detection tests.
- [ ] Implement differentiable router and heterogeneous blocks.
- [ ] Run focused tests.

### Task 4: Models, training, evaluation, CLI

**Files:** `src/neuroforge/models/*`, `src/neuroforge/training/*`, `src/neuroforge/evaluation/*`, `scripts/*`, `tests/integration/test_training.py`

- [ ] Write failing end-to-end update test.
- [ ] Implement baselines, adaptive model, metrics, latency benchmark, and reproducible artifact output.
- [ ] Run the full suite and CPU smoke experiment.

### Task 5: Documentation and research artifacts

**Files:** `README.md`, `docs/research/*`, `notebooks/*`, `results/reports/*`

- [ ] Document hypotheses, prior-art boundaries, experiment protocol, results, and limits.
- [ ] Create executable notebook guides that invoke reusable modules.
- [ ] Run reproduction command and record the actual outcome.
