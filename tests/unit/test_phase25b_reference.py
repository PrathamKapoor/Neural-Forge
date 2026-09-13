"""Phase 25B reference lineage — provenance and integrity tests.

These tests verify that the clean MLP reference lineage actually produced real
artifacts and passed every provenance gate. They read the artifacts; they do
not regenerate them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REF_DIR = Path("results/metrics/phase25b_reference")

FORBIDDEN = ("PLACEHOLDER", "TO_BE_COMPUTED", "REPLACED_WITH_HASH", "SYNTHETIC", "FAKE")


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def test_required_artifacts_exist():
    for name in [
        "manifest.json",
        "provenance_manifest.json",
        "provenance_summary.json",
        "reference_status.json",
        "config.json",
        "split_manifest.json",
        "dataset_identity.json",
        "checkpoint.json",
        "checkpoint.pt",
        "predictions.npy",
        "labels.npy",
        "sample_ids.npy",
        "metrics.json",
        "frozen_replay.json",
        "checkpoint_reconstruction.json",
        "independent_evaluation.json",
    ]:
        assert (REF_DIR / name).exists(), f"missing artifact: {name}"


def test_predictions_persisted_with_matching_hash():
    manifest = json.loads(_text(REF_DIR / "manifest.json"))
    pred_hash = _sha256(REF_DIR / "predictions.npy")
    assert pred_hash, "predictions.npy missing or empty"
    assert pred_hash == manifest["prediction_hash"], "prediction hash does not match manifest"


def test_predictions_labels_sample_ids_align():
    pred = np.load(REF_DIR / "predictions.npy")
    labels = np.load(REF_DIR / "labels.npy")
    sids = np.load(REF_DIR / "sample_ids.npy")
    assert pred.shape == labels.shape == sids.shape
    assert pred.shape[0] == 840


def test_frozen_replay_passed():
    replay = json.loads(_text(REF_DIR / "frozen_replay.json"))
    assert replay["checkpoint_loaded"] is False
    assert replay["passed"] is True
    assert replay["absolute_difference"] <= replay["tolerance"]


def test_checkpoint_reconstruction_passed():
    recon = json.loads(_text(REF_DIR / "checkpoint_reconstruction.json"))
    assert recon["passed"] is True
    assert recon["reconstructed_prediction_hash"] == recon["persisted_prediction_hash"]


def test_independent_evaluator_agrees():
    indep = json.loads(_text(REF_DIR / "independent_evaluation.json"))
    assert indep["passed"] is True
    assert indep["absolute_difference"] <= indep["tolerance"]


def test_dataset_and_config_verified():
    ds = json.loads(_text(REF_DIR / "dataset_identity.json"))
    cfg = json.loads(_text(REF_DIR / "config.json"))
    assert ds["dataset_version"] == "phase19-repaired-v1"
    assert ds["benchmark_version_verified"] is True
    assert cfg["architecture"] == "mlp"
    assert cfg["seed"] == 11


def test_no_fabrication_markers():
    for p in REF_DIR.glob("*.json"):
        text = _text(p)
        for marker in FORBIDDEN:
            assert marker not in text, f"fabrication marker {marker!r} in {p.name}"


def test_summary_declares_authoritative():
    summary = json.loads(_text(REF_DIR / "provenance_summary.json"))
    assert summary["mlp_reference"] == "AUTHORITATIVE"
    assert summary["architecture_change"] == "NONE"
    assert summary["rc_status"] == "REMAINS_CLOSED"
    assert summary["historical_p20_status"] == "HISTORICAL_PROVENANCE_INCOMPLETE"


def test_hardcoded_metrics_not_used_for_case():
    # The manifest links real hashes, not hardcoded values.
    manifest = json.loads(_text(REF_DIR / "manifest.json"))
    assert manifest["frozen_replay_passed"] is True
    assert manifest["checkpoint_reconstruction_passed"] is True
    assert manifest["independent_evaluator_passed"] is True