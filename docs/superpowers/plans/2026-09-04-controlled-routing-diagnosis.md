# Controlled Routing Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish whether NeuroForge routing can learn observable generator-defined routes, independently from the flawed Phase 1 oracle.

**Architecture:** Add an explicit controlled dataset and descriptor-conditioned routing probe without modifying Phase 1 models or evidence. Run reproducible multi-seed router baselines, supervised and indirect routing conditions, metrics, figures, and report.

**Tech Stack:** Python, PyTorch CPU, PyYAML, pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-04-controlled-routing-diagnosis-design.md`

## Global Constraints

- Preserve Phase 1 outputs and conclusions.
- Oracle labels come exclusively from generation.
- Separate routing quality from prediction, proxy compute, and latency.
- Use three configured CPU seeds; report mean and standard deviation.

---

### Task 1: Controlled data and routing metrics

**Files:** `src/neuroforge/datasets/controlled_routing.py`, `src/neuroforge/evaluation/routing_metrics.py`, `tests/unit/test_controlled_routing.py`

- [ ] Write failing tests for deterministic observable family descriptors, valid oracle routes, confusion matrix, and per-route accuracy.
- [ ] Implement generator, descriptors, metrics, and pathological-route checks.
- [ ] Run focused tests.

### Task 2: Descriptor-conditioned router probes

**Files:** `src/neuroforge/routing/probes.py`, `tests/unit/test_routing_probes.py`

- [ ] Write failing tests for supervised loss and soft/hard route shapes.
- [ ] Implement direct router and differentiable indirect routing probe.
- [ ] Run focused tests.

### Task 3: Multi-seed experiment and visualization

**Files:** `src/neuroforge/training/routing_diagnosis.py`, `src/neuroforge/visualization/routing_diagnosis.py`, `scripts/routing_diagnosis.py`, `tests/integration/test_routing_diagnosis.py`

- [ ] Write a failing artifact/reproducibility smoke test.
- [ ] Implement configured runner, latency breakdown, aggregation, JSON artifacts, and figures.
- [ ] Run the complete three-seed experiment.

### Task 4: Notebook and evidence report

**Files:** `notebooks/16_controlled_routing_diagnosis.ipynb`, `docs/research/log.md`, `results/reports/controlled_routing_diagnosis.md`

- [ ] Add a top-to-bottom notebook invoking the reusable experiment.
- [ ] Document findings and limitations based only on persisted output.
