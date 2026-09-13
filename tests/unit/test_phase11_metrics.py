"""Unit tests for Phase 11 diagnostic metrics functions."""
from __future__ import annotations

import pytest
import torch

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase11_metrics import (
    aggregation_diagnosis,
    build_causal_diagnosis,
    composition_order_diagnosis,
    interface_compatibility,
    minimal_validated_composition,
    mixed_task_component_validation,
    oracle_composition_feasibility,
    representation_transfer_probe,
    routing_representation_decodability,
    routing_selection_diagnosis,
)
from neuroforge.models.specialists import StandaloneSpecialist


def _make_specialists() -> dict[str, StandaloneSpecialist]:
    return {
        "mlp": StandaloneSpecialist(architecture="mlp", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
        "graph": StandaloneSpecialist(architecture="graph", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
        "attention": StandaloneSpecialist(architecture="attention", input_dim=8, hidden_dim=24, depth=1, num_classes=2),
        "attention_v2": StandaloneSpecialist(architecture="attention_v2", input_dim=8, hidden_dim=24, depth=2, num_classes=2),
    }


@pytest.fixture()
def trained_experts_and_dataset():
    """Train all 4 experts for 1 epoch so the metrics have non-trivial logits."""
    from neuroforge.evaluation.specialization import select_capacity_phase6
    from neuroforge.training.phase8b_compute_aware import _train_expert
    selected = select_capacity_phase6()
    experts: dict[str, StandaloneSpecialist] = {}
    for arch, fam in [("mlp", "feature"), ("graph", "relational"), ("attention", "contextual"), ("attention_v2", "contextual")]:
        exp, _ = _train_expert(
            architecture=arch, target_family=fam, depth=selected.depths[arch],
            seed=11, train_samples=120, val_samples=40, epochs=1, batch_size=20,
        )
        experts[arch] = exp
    eval_ds = Phase9MixedStructureDataset(samples_per_type=10, seed=11)
    return experts, eval_ds


def test_oracle_composition_feasibility_returns_all_combos(trained_experts_and_dataset):
    experts, eval_ds = trained_experts_and_dataset
    raw_flops = {"mlp": 8928.0, "graph": 8112.0, "attention": 39168.0, "attention_v2": 46296.0}
    res = oracle_composition_feasibility(
        experts, ("mlp", "graph", "attention", "attention_v2"), eval_ds, raw_flops,
        combinations=[("mlp",), ("graph",), ("mlp", "graph"), ("mlp", "graph", "attention_v2")],
    )
    assert "per_combo" in res
    for combo in ("mlp", "graph", "mlp+graph", "mlp+graph+attention_v2"):
        assert combo in res["per_combo"]
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
        assert f in res["ceiling_per_family"]
    assert 0.0 <= res["ceiling_overall"] <= 1.0


def test_representation_transfer_probe_returns_all_experts_and_stages(trained_experts_and_dataset):
    experts, eval_ds = trained_experts_and_dataset
    component_targets = {
        "sf": torch.tensor([int(eval_ds.items[i]["sf"] == 1) for i in range(len(eval_ds))], dtype=torch.long),
        "sr": torch.tensor([int(eval_ds.items[i]["sr"] == 1) for i in range(len(eval_ds))], dtype=torch.long),
        "sc": torch.tensor([int(eval_ds.items[i]["sc"] == 1) for i in range(len(eval_ds))], dtype=torch.long),
        "final_target": eval_ds.targets,
    }
    res = representation_transfer_probe(experts, eval_ds, component_targets)
    for exp_name in ("mlp", "graph", "attention", "attention_v2"):
        assert exp_name in res["per_expert"]
        for stage in ("input", "after_encoder", "after_blocks"):
            assert stage in res["per_expert"][exp_name]
            for task in ("sf", "sr", "sc", "final_target"):
                entry = res["per_expert"][exp_name][stage][task]
                assert "overall" in entry
                assert 0.0 <= entry["overall"] <= 1.0


def test_interface_compatibility_returns_per_family(trained_experts_and_dataset):
    experts, eval_ds = trained_experts_and_dataset
    res = interface_compatibility(experts, eval_ds)
    for seq in ("mlp", "graph", "mlp+graph", "mlp+graph+attention_v2"):
        assert seq in res["per_sequence"]
        for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
            assert f in res["per_sequence"][seq]["per_family"]


def test_aggregation_diagnosis_methods_present(trained_experts_and_dataset):
    experts, eval_ds = trained_experts_and_dataset
    res = aggregation_diagnosis(experts, eval_ds, methods=("raw", "weighted_router", "concat_only_pooled"))
    for m in ("raw", "weighted_router", "concat_only_pooled"):
        assert m in res["methods"]
        assert "per_family" in res["methods"][m]
    assert "counterfactual" in res

    res = mixed_task_component_validation(experts, eval_ds, families=("FR", "RC", "FC", "FRC"))
    # Each family contains only the components in its name
    family_components = {"FR": ("F", "R"), "RC": ("R", "C"), "FC": ("F", "C"), "FRC": ("F", "R", "C")}
    for fam, expected_comps in family_components.items():
        assert fam in res["component_load_bearing"]
        for c in expected_comps:
            assert c in res["component_load_bearing"][fam]
    assert "per_expert_pred_components" in res
    assert "destroy_results" in res
    best_combos = {"F": "mlp", "R": "graph", "C": "attention_v2", "FR": "mlp+graph", "RC": "graph+attention_v2", "FC": "mlp+attention_v2", "FRC": "mlp+graph+attention_v2"}

    def sel_fn(features):
        return [["mlp"] for _ in range(len(features))]

    res = routing_selection_diagnosis(sel_fn, experts, eval_ds, best_combos)
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
        assert 0.0 <= res["agreement_rate_per_family"].get(f, 0.0) <= 1.0
        assert 0.0 <= res["useful_recovery_rate_per_family"].get(f, 0.0) <= 1.0
    assert "mean_k" in res


def test_mixed_task_component_validation_returns_four_families(trained_experts_and_dataset):
    experts, eval_ds = trained_experts_and_dataset
    res = mixed_task_component_validation(experts, eval_ds, families=("FR", "RC", "FC", "FRC"))
    family_components = {"FR": ("F", "R"), "RC": ("R", "C"), "FC": ("F", "C"), "FRC": ("F", "R", "C")}
    for fam, expected_comps in family_components.items():
        assert fam in res["component_load_bearing"]
        for c in expected_comps:
            assert c in res["component_load_bearing"][fam]
    assert "per_expert_pred_components" in res
    assert "destroy_results" in res


def test_minimal_validated_composition_with_and_without_adapter(trained_experts_and_dataset):
    experts, eval_ds = trained_experts_and_dataset
    best_combos = {"F": "mlp", "R": "graph", "C": "attention_v2", "FR": "mlp+graph", "RC": "graph+attention_v2", "FC": "mlp+attention_v2", "FRC": "mlp+graph+attention_v2"}
    train_ds = Phase9MixedStructureDataset(samples_per_type=10, seed=11 + 50_000)
    no_ad = minimal_validated_composition(experts, eval_ds, best_combos, use_adapter=False, train_ds_for_adapter=train_ds)
    with_ad = minimal_validated_composition(experts, eval_ds, best_combos, use_adapter=True, train_ds_for_adapter=train_ds)
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
        assert f in no_ad["per_family"]
        assert f in with_ad["per_family"]
    assert with_ad["total_extra_params"] >= 0
    assert with_ad["use_adapter"] is True
    assert no_ad["use_adapter"] is False


def test_routing_representation_decodability_returns_7_families(trained_experts_and_dataset):
    _, eval_ds = trained_experts_and_dataset
    # router representation is mean+std pooled
    feats = eval_ds.features
    mean = feats.mean(dim=1)
    std = feats.std(dim=1)
    router_repr = torch.cat([mean, std], dim=-1)
    res = routing_representation_decodability(router_repr, eval_ds)
    for f in ("F", "R", "C", "FR", "RC", "FC", "FRC"):
        assert f in res["per_family"]
    assert 0.0 <= res["overall"] <= 1.0


def test_build_causal_diagnosis_has_all_categories(trained_experts_and_dataset):
    experts, eval_ds = trained_experts_and_dataset
    oracle = oracle_composition_feasibility(
        experts, ("mlp", "graph", "attention", "attention_v2"), eval_ds,
        raw_flops={"mlp": 8928.0, "graph": 8112.0, "attention": 39168.0, "attention_v2": 46296.0},
        combinations=[("mlp",), ("graph",), ("mlp", "graph")],
    )
    component_targets = {
        "sf": torch.tensor([int(eval_ds.items[i]["sf"] == 1) for i in range(len(eval_ds))], dtype=torch.long),
        "sr": torch.tensor([int(eval_ds.items[i]["sr"] == 1) for i in range(len(eval_ds))], dtype=torch.long),
        "sc": torch.tensor([int(eval_ds.items[i]["sc"] == 1) for i in range(len(eval_ds))], dtype=torch.long),
        "final_target": eval_ds.targets,
    }
    rep = representation_transfer_probe(experts, eval_ds, component_targets)
    iface = interface_compatibility(experts, eval_ds)
    agg = aggregation_diagnosis(experts, eval_ds, methods=("raw", "weighted_router"))
    order = composition_order_diagnosis(experts, eval_ds)
    valid = mixed_task_component_validation(experts, eval_ds)
    feats = eval_ds.features
    router_repr = torch.cat([feats.mean(dim=1), feats.std(dim=1)], dim=-1)
    repr_dec = routing_representation_decodability(router_repr, eval_ds)

    def sel_fn(f):
        return [["mlp"] for _ in range(len(f))]

    routing = routing_selection_diagnosis(sel_fn, experts, eval_ds, {"F": "mlp", "R": "graph", "C": "attention_v2"})

    diag = build_causal_diagnosis(
        oracle, rep, iface, agg, order, routing, valid, repr_dec,
        phase10_overall_mixed=0.5, phase10_ceiling=0.6,
    )
    for cat in [
        "EXPERT_CAPABILITY",
        "REPRESENTATION_TRANSFER",
        "AGGREGATION",
        "COMPOSITION_ORDER",
        "ROUTER_SELECTION",
        "BENCHMARK_SEMANTICS",
        "ROUTING_REPRESENTATION",
        "COMPUTE_ECONOMICS",
    ]:
        assert cat in diag
        assert diag[cat]["status"] in {
            "SUPPORTED",
            "PARTIALLY SUPPORTED",
            "NOT SUPPORTED",
            "INCONCLUSIVE",
            "NOT TESTED",
        }
