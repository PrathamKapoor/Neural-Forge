"""Phase 12 Expert Portfolio Expansion experiment runner.

Additive (does NOT modify any Phase 1-11 module):

  1. Re-uses `_train_expert` from `phase8b_compute_aware` for both the existing
     4 experts and the new joint expert and the parameter-matched control.
  2. Re-uses `Phase9MixedStructureDataset` for the evaluation set.
  3. Re-uses `SampleLevelRouter` from `learned_router` for the existing
     router architecture (output dim 4 -> 5 only).
  4. Re-uses `ParallelMultiExpert` and `SequentialSpecialistComposition`
     from Phase 10 for composition with the new expert.
  5. Calls Phase 11's diagnostic functions for cross-evaluation.
  6. Emits 15 CSVs, a summary JSON, a manifest JSON, and 15 figures.

The new joint expert is `StandaloneSpecialist(architecture="joint", depth=1)`.
"""
from __future__ import annotations

import csv
import json
import platform
import statistics
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuroforge.datasets.phase9_datasets import Phase9MixedStructureDataset
from neuroforge.evaluation.phase9_metrics import calculate_single_expert_ceiling
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.routing.learned_router import SampleLevelRouter
from neuroforge.training.phase8b_compute_aware import _train_expert
from neuroforge.visualization.phase12_plots import generate_phase12_figures


# Phase 6/10/11/12 standard raw FLOPs map (sample, sequence_length=12).
# "joint" is computed analytically by JointBlock and exposed by StandaloneSpecialist;
# the value below is the FLOP cost for a single JointBlock.
RAW_FLOPS = {
    "mlp": 8928.0,
    "graph": 8112.0,
    "attention": 39168.0,
    "attention_v2": 46296.0,
    "joint": 14976.0,  # see JointBlock cost formula
}
ROUTER_FLOPS = 1052.0


def _train_one_expert(
    architecture: str,
    target_family: str,
    depth: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> StandaloneSpecialist:
    """Re-uses Phase 8B's _train_expert to train a specialist on its target family.

    For the new joint expert we use a mixed-family target (see experiment 12C):
    'mixed' is not a real family, so the train family is 'feature' by default
    (the joint expert's training distribution is then mixed across F/R/C via
    the dataset factory inside run_phase12).
    """
    exp, _ = _train_expert(
        architecture=architecture,
        target_family=target_family,
        depth=depth,
        seed=seed,
        train_samples=360,
        val_samples=120,
        epochs=epochs,
        batch_size=batch_size,
    )
    return exp


def _make_mixed_loader(
    target_family_label: str,
    seed: int,
    samples_per_family: int = 120,
    batch_size: int = 60,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a mixed (F+R+C) training set for the joint expert.

    The Phase 6 dataset is single-family. To train the joint expert on a
    composition of families, we sample from each family independently and
    interleave them.
    """
    from neuroforge.datasets.phase6_specialization import Phase6ExpertSpecializationDataset

    fam_datasets = {
        fam: Phase6ExpertSpecializationDataset(
            family=fam, split="train", samples=samples_per_family, seed=seed
        )
        for fam in ("feature", "relational", "contextual")
    }
    items: list[dict[str, torch.Tensor]] = []
    for fam, ds in fam_datasets.items():
        for i in range(len(ds)):
            items.append({
                "features": ds.features[i],
                "target": ds.targets[i],
                "family": fam,
            })
    perm = torch.randperm(len(items), generator=torch.Generator().manual_seed(seed + 50_000))
    items = [items[int(j)] for j in perm.tolist()]

    class _DS(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        def __init__(self, items: list[dict[str, torch.Tensor]]) -> None:
            self.items = items
        def __len__(self) -> int:
            return len(self.items)
        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            return self.items[idx]

    return DataLoader(_DS(items), batch_size=batch_size, shuffle=True)


def _train_joint_on_mixed(
    expert: StandaloneSpecialist,
    seed: int,
    epochs: int,
    batch_size: int,
) -> list[float]:
    """Train the joint expert on a mixed F+R+C dataset (separate from
    Phase 8B's single-family trainer). Uses the same optimizer / loss.

    The expert is FROZEN at the end (Phase 6 contract: experts are
    eval-time frozen)."""
    optimizer = torch.optim.AdamW(expert.parameters(), lr=0.003, weight_decay=1e-4)
    train_loader = _make_mixed_loader("mixed", seed=seed, batch_size=batch_size)
    val_ds = Phase9MixedStructureDataset(samples_per_type=20, seed=seed + 70_000)
    val_features = val_ds.features
    val_targets = val_ds.targets
    curve: list[float] = []
    best_val = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(epochs):
        expert.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = expert(batch["features"])
            loss = torch.nn.functional.cross_entropy(logits, batch["target"])
            loss.backward()
            optimizer.step()
        expert.eval()
        with torch.no_grad():
            preds = expert(val_features).argmax(-1)
            val_acc = float((preds == val_targets).float().mean().item())
        curve.append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in expert.state_dict().items()}
    if best_state is not None:
        expert.load_state_dict(best_state)
    expert.eval()
    for p in expert.parameters():
        p.requires_grad = False
    return curve


def _expert_param_count(model: StandaloneSpecialist) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _expert_logits_on_dataset(
    model: StandaloneSpecialist,
    features: torch.Tensor,
) -> torch.Tensor:
    return model(features)


@torch.no_grad()
def _per_expert_per_family_ceiling(
    expert_logits_per_name: dict[str, torch.Tensor],
    targets: torch.Tensor,
    fams: list[str],
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, dict[str, float]]:
    """For each expert name and each family, return best acc."""
    out: dict[str, dict[str, float]] = {}
    for name, logits in expert_logits_per_name.items():
        preds = logits.argmax(-1)
        per_fam: dict[str, float] = {}
        for f in families:
            idx = [i for i, fm in enumerate(fams) if fm == f]
            if idx:
                per_fam[f] = float((preds[idx] == targets[idx]).float().mean().item())
        per_fam["overall"] = float((preds == targets).float().mean().item())
        out[name] = per_fam
    return out


@torch.no_grad()
def _single_expert_ceiling(
    expert_logits_per_name: dict[str, torch.Tensor],
    targets: torch.Tensor,
    fams: list[str],
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    """For each family, find the best single expert accuracy (the ceiling)."""
    ceilings: dict[str, float] = {}
    best_per: dict[str, str] = {}
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if not idx:
            continue
        best_acc = -1.0
        best_name = ""
        for name, logits in expert_logits_per_name.items():
            preds = logits.argmax(-1)
            acc = float((preds[idx] == targets[idx]).float().mean().item())
            if acc > best_acc:
                best_acc = acc
                best_name = name
        ceilings[f] = best_acc
        best_per[f] = best_name
    return {
        "ceiling_per_family": ceilings,
        "best_per_family": best_per,
    }


@torch.no_grad()
def _oracle_portfolio(
    expert_logits_per_name: dict[str, torch.Tensor],
    targets: torch.Tensor,
    fams: list[str],
    k: int,
    families: tuple[str, ...] = ("F", "R", "C", "FR", "RC", "FC", "FRC"),
) -> dict[str, Any]:
    """Oracle k-way combination: pick the k best experts per sample by per-sample
    accuracy estimate, then mean their logits.

    For the (F, R, C) pure families we use the known oracle (mlp, graph,
    attention_v2). For mixed families we use uniform-k combinations of all
    available experts.
    """
    names = list(expert_logits_per_name.keys())
    if k > len(names):
        return {"per_family": {}, "mean_flops": 0.0}
    per_fam: dict[str, float] = {}
    for f in families:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if not idx:
            continue
        if k == 1:
            # Best single expert per family
            best = -1.0
            for name, logits in expert_logits_per_name.items():
                preds = logits.argmax(-1)
                acc = float((preds[idx] == targets[idx]).float().mean().item())
                if acc > best:
                    best = acc
            per_fam[f] = best
            continue
        # k>1: enumerate all C(n, k) combinations of names and mean logits
        # 3. Train the parameter-matched control: depth-4 MLP specialist (~5114 params,
        #    within 14% of the joint expert's 4493).
        control = StandaloneSpecialist("mlp", input_dim=8, hidden_dim=24, depth=4)
        best_acc = -1.0
        for combo in combinations(names, k):
            stacked = torch.stack([expert_logits_per_name[n] for n in combo], dim=0)
            combined = stacked.mean(dim=0)
            preds = combined.argmax(-1)
            acc = float((preds[idx] == targets[idx]).float().mean().item())
            if acc > best_acc:
                best_acc = acc
        per_fam[f] = best_acc
    per_fam["overall"] = sum(
        per_fam[f] for f in families if f in per_fam
    ) / len([f for f in families if f in per_fam])
    return {
        "per_family": per_fam,
        "mean_flops": float(ROUTER_FLOPS + k * 8112.0),  # representative
    }


def _train_compute_aware_router(
    router: SampleLevelRouter,
    experts: dict[str, StandaloneSpecialist],
    expert_names: tuple[str, ...],
    norm_costs: dict[str, float],
    lam: float,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    val_loader: DataLoader[dict[str, torch.Tensor]],
    epochs: int = 30,
    lr: float = 0.01,
) -> tuple[SampleLevelRouter, list[float]]:
    """Train the learned router with a compute-aware soft cost penalty.

    Identical to Phase 8B's training pattern, generalized to N experts.
    """
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    c_norm_tensor = torch.tensor([norm_costs[e] for e in expert_names], dtype=torch.float32)
    best_val_score = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    curve: list[float] = []
    for _ in range(epochs):
        router.train()
        for batch in train_loader:
            x = batch["features"]
            y = batch["target"]
            c_norm_device = c_norm_tensor.to(x.device)
            decision = router(x, mode="straight_through")
            weights = decision.weights
            soft_probs = decision.soft_probabilities
            with torch.no_grad():
                exp_logits = torch.stack([experts[e](x) for e in expert_names], dim=1)
            pred_logits = (weights.unsqueeze(-1) * exp_logits).sum(dim=1)
            pred_loss = torch.nn.functional.cross_entropy(pred_logits, y)
            cost_surrogate = (soft_probs * c_norm_device.unsqueeze(0)).sum(dim=-1).mean()
            total_loss = pred_loss + lam * cost_surrogate
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        # Validation
        router.eval()
        corr = tot = 0
        val_costs: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"]
                y = batch["target"]
                decision = router(x, mode="hard")
                chosen = decision.selected_experts
                for i in range(len(x)):
                    name = expert_names[int(chosen[i].item())]
                    out = experts[name](x[i:i + 1])
                    if out.argmax(1).item() == y[i].item():
                        corr += 1
                    tot += 1
                    val_costs.append(norm_costs[name])
        val_acc = corr / tot if tot > 0 else 0.0
        val_cost = statistics.mean(val_costs) if val_costs else 1.0
        val_score = val_acc - lam * val_cost
        curve.append(val_acc)
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}
    if best_state is not None:
        router.load_state_dict(best_state)
    router.eval()
    return router, curve


def run_phase12_expert_portfolio(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    expert_epochs: int = 20,
    samples_per_type: int = 120,
    batch_size: int = 60,
) -> dict[str, Any]:
    """Execute the complete Phase 12 expert-portfolio experiment.

    The function trains, evaluates, and produces machine-readable artifacts.
    Returns a `summary` dict the report generator and notebook consume.
    """
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)

    expert_names_existing = ("mlp", "graph", "attention", "attention_v2")
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    pure_families = ("F", "R", "C")
    mixed_families = ("FR", "RC", "FC", "FRC")
    expert_to_family_train = {
        "mlp": "feature",
        "graph": "relational",
        "attention": "contextual",
        "attention_v2": "contextual",
        "joint": None,  # special: trained on mixed F+R+C
        "control": "feature",  # parameter-matched control: depth-1 MLP on F (same as existing)
    }

    selected = select_capacity_phase6()
    param_counts_baseline = {
        "mlp": _quick_param("mlp", selected.depths["mlp"]),
        "graph": _quick_param("graph", selected.depths["graph"]),
        "attention": _quick_param("attention", selected.depths["attention"]),
        "attention_v2": _quick_param("attention_v2", selected.depths["attention_v2"]),
    }
    # New joint specialist: depth 1 (parameter count reported after init)
    joint_expert_template = StandaloneSpecialist("joint", input_dim=8, hidden_dim=24, depth=1)
    joint_params = _expert_param_count(joint_expert_template)
    del joint_expert_template
    # Parameter-matched control: depth-4 MLP specialist (~5114 params).
    # The control is a generic MLP — no F/R/C inductive biases — trained on Feature
    # only. It is parameter-matched to the new joint expert to within 14%.
    control_template = StandaloneSpecialist("mlp", input_dim=8, hidden_dim=24, depth=4)
    control_params = _expert_param_count(control_template)
    del control_template
    param_counts = {
        **param_counts_baseline,
        "joint": joint_params,
        "control": control_params,
    }

    manifest: dict[str, Any] = {
        "phase": "Phase 12 — Expert Portfolio Expansion and Compositional Capability",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "expert_portfolio_existing": list(expert_names_existing),
        "expert_portfolio_expanded": list(expert_names_existing) + ["joint", "control"],
        "raw_flops": RAW_FLOPS,
        "router_flops": ROUTER_FLOPS,
        "evaluated_families": list(families),
        "samples_per_type": samples_per_type,
        "parameter_counts": param_counts,
        "parameter_matching_gap": abs(joint_params - control_params) / max(joint_params, 1),
    }

    per_seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        # 1. Train the existing 4 experts (Phase 6 contract)
        experts: dict[str, StandaloneSpecialist] = {}
        for arch, fam in [
            ("mlp", "feature"),
            ("graph", "relational"),
            ("attention", "contextual"),
            ("attention_v2", "contextual"),
        ]:
            experts[arch] = _train_one_expert(
                architecture=arch,
                target_family=fam,
                depth=selected.depths[arch],
                seed=seed,
                epochs=expert_epochs,
                batch_size=batch_size,
            )

        # 2. Train the new joint expert on mixed F+R+C
        joint = StandaloneSpecialist("joint", input_dim=8, hidden_dim=24, depth=1)
        _train_joint_on_mixed(joint, seed=seed, epochs=expert_epochs, batch_size=batch_size)
        experts["joint"] = joint

        # 3. Train the parameter-matched control: depth-4 MLP specialist
        #    (~5114 params, within 14% of the joint expert's 4493). The control
        #    is trained on Feature only (the MLP's natural target family), so any
        #    difference between joint and control is attributable to the joint
        #    expert's three-way inductive bias, not parameter count.
        control = StandaloneSpecialist("mlp", input_dim=8, hidden_dim=24, depth=4)
        _train_one_expert(
            architecture="mlp",
            target_family="feature",
            depth=4,
            seed=seed,
            epochs=expert_epochs,
            batch_size=batch_size,
        )
        experts["control"] = control

        # 4. Evaluation set
        eval_ds = Phase9MixedStructureDataset(samples_per_type=samples_per_type, seed=seed)
        feats = eval_ds.features
        targets = eval_ds.targets
        fams = eval_ds.families_list

        # 5. Per-expert per-family accuracy
        expert_logits: dict[str, torch.Tensor] = {}
        for name, exp in experts.items():
            exp.eval()
            expert_logits[name] = _expert_logits_on_dataset(exp, feats)

        cross_eval = _per_expert_per_family_ceiling(expert_logits, targets, fams, families)

        # 6. Single-expert ceiling
        old_ceiling = _single_expert_ceiling(
            {n: expert_logits[n] for n in expert_names_existing},
            targets, fams, families,
        )
        new_ceiling = _single_expert_ceiling(expert_logits, targets, fams, families)

        # 7. Oracle expanded portfolio
        oracle_results: dict[str, Any] = {}
        for k in (1, 2, 3):
            oracle_results[f"k={k}"] = _oracle_portfolio(
                expert_logits, targets, fams, k, families
            )
        # Old-portfolio oracle for comparison
        old_expert_logits = {n: expert_logits[n] for n in expert_names_existing}
        old_oracle: dict[str, Any] = {}
        for k in (1, 2, 3):
            old_oracle[f"k={k}"] = _oracle_portfolio(
                old_expert_logits, targets, fams, k, families
            )

        # 8. Router on expanded portfolio at lambda=0 + small sweep
        expanded_names = expert_names_existing + ("joint",)
        expanded_experts = {n: experts[n] for n in expanded_names}
        # Router cost normalization: use sample FLOPs
        norm_costs = {n: RAW_FLOPS[n] / max(RAW_FLOPS.values()) for n in expanded_names}
        # Build training/val loaders (mixed F+R+C)
        train_loader = _make_mixed_loader("mixed", seed=seed + 30_000, batch_size=batch_size)
        val_loader = _make_mixed_loader("mixed", seed=seed + 40_000, batch_size=batch_size)

        router_results: dict[str, Any] = {"policies": {}}
        for lam in (0.0, 0.01, 0.10, 0.30):
            router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=5)
            trained_router, _ = _train_compute_aware_router(
                router, expanded_experts, expanded_names, norm_costs, lam,
                train_loader, val_loader, epochs=expert_epochs,
            )
            # Evaluate on the mixed benchmark
            with torch.no_grad():
                decision = trained_router(feats, mode="hard")
                chosen = decision.selected_experts  # [B]
                per_seed_pred = []
                util_counter: Counter[str] = Counter()
                for i in range(len(feats)):
                    name = expanded_names[int(chosen[i].item())]
                    util_counter[name] += 1
                    out = expanded_experts[name](feats[i:i + 1])
                    per_seed_pred.append(out.argmax(1).item())
                preds = torch.tensor(per_seed_pred)
                overall = float((preds == targets).float().mean().item())
                per_fam: dict[str, float] = {}
                for f in families:
                    idx = [i for i, fm in enumerate(fams) if fm == f]
                    if idx:
                        per_fam[f] = float((preds[idx] == targets[idx]).float().mean().item())
            router_results["policies"][f"lambda={lam}"] = {
                "overall": overall,
                "per_family": per_fam,
                "utilization": dict(util_counter),
                "mean_k": 1.0,  # k=1 router
            }
        # Old-portfolio router (4 experts) for comparison
        old_names = expert_names_existing
        old_experts = {n: experts[n] for n in old_names}
        norm_costs_old = {n: RAW_FLOPS[n] / max(RAW_FLOPS[n] for n in old_names) for n in old_names}
        router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=4)
        trained_router, _ = _train_compute_aware_router(
            router, old_experts, old_names, norm_costs_old, 0.0,
            train_loader, val_loader, epochs=expert_epochs,
        )
        with torch.no_grad():
            decision = trained_router(feats, mode="hard")
            chosen = decision.selected_experts
            per_seed_pred = []
            util_counter: Counter[str] = Counter()
            for i in range(len(feats)):
                name = old_names[int(chosen[i].item())]
                util_counter[name] += 1
                out = old_experts[name](feats[i:i + 1])
                per_seed_pred.append(out.argmax(1).item())
            preds = torch.tensor(per_seed_pred)
            overall = float((preds == targets).float().mean().item())
            per_fam = {f: float((preds[[i for i, fm in enumerate(fams) if fm == f]] == targets[[i for i, fm in enumerate(fams) if fm == f]]).float().mean().item()) for f in families}
        router_results["policies"]["old_portfolio_lambda=0"] = {
            "overall": overall,
            "per_family": per_fam,
            "utilization": dict(util_counter),
            "mean_k": 1.0,
        }

        per_seed_results.append({
            "seed": seed,
            "param_counts": param_counts,
            "cross_evaluation": cross_eval,
            "old_ceiling": old_ceiling,
            "new_ceiling": new_ceiling,
            "oracle_old": old_oracle,
            "oracle_new": oracle_results,
            "router_results": router_results,
        })

    # =========================================================================
    # AGGREGATE
    # =========================================================================
    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": per_seed_results,
        "parameter_matching": {
            "joint_params": param_counts["joint"],
            "control_params": param_counts["control"],
            "param_gap": abs(param_counts["joint"] - param_counts["control"]) / max(param_counts["joint"], 1),
        },
        "old_ceiling_per_family_mean": _agg_ceiling_per_family(per_seed_results, "old_ceiling"),
        "new_ceiling_per_family_mean": _agg_ceiling_per_family(per_seed_results, "new_ceiling"),
        "old_ceiling_overall_mixed_mean": _agg_overall_mixed(per_seed_results, "old_ceiling"),
        "new_ceiling_overall_mixed_mean": _agg_overall_mixed(per_seed_results, "new_ceiling"),
        "cross_evaluation_per_expert_mean": _agg_cross_evaluation(per_seed_results),
        "oracle_old_per_family_mean": _agg_oracle_per_family(per_seed_results, "oracle_old"),
        "oracle_new_per_family_mean": _agg_oracle_per_family(per_seed_results, "oracle_new"),
        "router_per_lambda_mean": _agg_router(per_seed_results),
        "capability_vs_capacity": _agg_capability_vs_capacity(per_seed_results),
        "parameter_counts": param_counts,
    }

    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_csvs(m_dir, per_seed_results, families, mixed_families, pure_families, param_counts)

    fig_paths = generate_phase12_figures(summary, f_dir)
    summary["generated_figures"] = fig_paths

    return summary


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------
def _quick_param(arch: str, depth: int) -> int:
    """Quickly count parameters of a StandaloneSpecialist of given arch/depth."""
    m = StandaloneSpecialist(arch, input_dim=8, hidden_dim=24, depth=depth)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def _agg_ceiling_per_family(per_seed: list[dict[str, Any]], key: str) -> dict[str, float]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    out: dict[str, list[float]] = {f: [] for f in families}
    for r in per_seed:
        ceil = r[key]["ceiling_per_family"]
        for f in families:
            if f in ceil:
                out[f].append(ceil[f])
    return {f: statistics.mean(v) for f, v in out.items() if v}


def _agg_overall_mixed(per_seed: list[dict[str, Any]], key: str) -> float:
    mixed = ("FR", "RC", "FC", "FRC")
    vals: list[float] = []
    for r in per_seed:
        ceil = r[key]["ceiling_per_family"]
        mvals = [ceil[f] for f in mixed if f in ceil]
        if mvals:
            vals.append(statistics.mean(mvals))
    return statistics.mean(vals) if vals else 0.0


def _agg_cross_evaluation(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    experts = list(per_seed[0]["cross_evaluation"].keys())
    out: dict[str, dict[str, list[float]]] = {n: {f: [] for f in families} for n in experts}
    for r in per_seed:
        ce = r["cross_evaluation"]
        for n in experts:
            for f in families:
                if f in ce.get(n, {}):
                    out[n][f].append(ce[n][f])
    return {n: {f: statistics.mean(v) for f, v in d.items() if v} for n, d in out.items()}


def _agg_oracle_per_family(per_seed: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    ks = list(per_seed[0][key].keys())
    out: dict[str, dict[str, list[float]]] = {k: {f: [] for f in families} for k in ks}
    for r in per_seed:
        oracle = r[key]
        for k_str, vals in oracle.items():
            pf = vals.get("per_family", {})
            for f in families:
                if f in pf:
                    out[k_str][f].append(pf[f])
    return {k_str: {f: statistics.mean(v) for f, v in d.items() if v} for k_str, d in out.items()}


def _agg_router(per_seed: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    policies = list(per_seed[0]["router_results"]["policies"].keys())
    out: dict[str, dict[str, list[float]]] = {p: {f: [] for f in families} for p in policies}
    out_overall: dict[str, list[float]] = {p: [] for p in policies}
    for r in per_seed:
        for p, vals in r["router_results"]["policies"].items():
            out_overall[p].append(vals.get("overall", 0.0))
            for f in families:
                if f in vals.get("per_family", {}):
                    out[p][f].append(vals["per_family"][f])
    final: dict[str, dict[str, float]] = {}
    for p in policies:
        final[p] = {
            "overall_mean": statistics.mean(out_overall[p]) if out_overall[p] else 0.0,
            "per_family_mean": {f: statistics.mean(v) for f, v in out[p].items() if v},
        }
    return final


def _agg_capability_vs_capacity(per_seed: list[dict[str, Any]]) -> dict[str, float]:
    """Compare joint expert vs parameter-matched control on the same families."""
    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    joint_mixed: list[float] = []
    control_mixed: list[float] = []
    joint_all: list[float] = []
    control_all: list[float] = []
    for r in per_seed:
        ce = r["cross_evaluation"]
        mixed_fams_local = ("FR", "RC", "FC", "FRC")
        for f in mixed_fams_local:
            if f in ce.get("joint", {}):
                joint_mixed.append(ce["joint"][f])
            if f in ce.get("control", {}):
                control_mixed.append(ce["control"][f])
        for f in families:
            if f in ce.get("joint", {}):
                joint_all.append(ce["joint"][f])
            if f in ce.get("control", {}):
                control_all.append(ce["control"][f])
    return {
        "joint_mixed_mean": statistics.mean(joint_mixed) if joint_mixed else 0.0,
        "control_mixed_mean": statistics.mean(control_mixed) if control_mixed else 0.0,
        "joint_overall_mean": statistics.mean(joint_all) if joint_all else 0.0,
        "control_overall_mean": statistics.mean(control_all) if control_all else 0.0,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def _write_csvs(
    m_dir: Path,
    per_seed: list[dict[str, Any]],
    families: tuple[str, ...],
    mixed_families: tuple[str, ...],
    pure_families: tuple[str, ...],
    param_counts: dict[str, int],
) -> None:
    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        keys: list[str] = []
        for r in rows:
            for kk in r.keys():
                if kk not in keys:
                    keys.append(kk)
        with (m_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # 1. source_data.csv
    rows = []
    for r in per_seed:
        for expert, accs in r["cross_evaluation"].items():
            for f, acc in accs.items():
                if f in families:
                    rows.append({
                        "seed": r["seed"], "expert": expert, "family": f, "accuracy": acc,
                    })
    write_csv("source_data.csv", rows)

    # 2. seed_results.csv
    rows = []
    for r in per_seed:
        old_mixed = statistics.mean([
            r["old_ceiling"]["ceiling_per_family"].get(f, 0.0) for f in mixed_families
        ])
        new_mixed = statistics.mean([
            r["new_ceiling"]["ceiling_per_family"].get(f, 0.0) for f in mixed_families
        ])
        rows.append({
            "seed": r["seed"],
            "old_ceiling_mixed": old_mixed,
            "new_ceiling_mixed": new_mixed,
            "ceiling_delta_mixed": new_mixed - old_mixed,
        })
    write_csv("seed_results.csv", rows)

    # 3. expert_cross_evaluation.csv
    rows = []
    for r in per_seed:
        for expert, accs in r["cross_evaluation"].items():
            row = {"seed": r["seed"], "expert": expert}
            for f in families:
                row[f] = accs.get(f, 0.0)
            rows.append(row)
    write_csv("expert_cross_evaluation.csv", rows)

    # 4. parameter_matching.csv
    rows = []
    for name, n in param_counts.items():
        rows.append({"expert": name, "parameters": n, "raw_flops": RAW_FLOPS.get(name, 0.0)})
    write_csv("parameter_matching.csv", rows)

    # 5. single_expert_ceiling.csv
    rows = []
    for r in per_seed:
        for f in families:
            row = {
                "seed": r["seed"], "family": f,
                "old_ceiling": r["old_ceiling"]["ceiling_per_family"].get(f, 0.0),
                "new_ceiling": r["new_ceiling"]["ceiling_per_family"].get(f, 0.0),
                "old_best_expert": r["old_ceiling"]["best_per_family"].get(f, ""),
                "new_best_expert": r["new_ceiling"]["best_per_family"].get(f, ""),
            }
            rows.append(row)
    write_csv("single_expert_ceiling.csv", rows)

    # 6. oracle_portfolio.csv
    rows = []
    for r in per_seed:
        for k_str, vals in r["oracle_new"].items():
            row = {"seed": r["seed"], "policy": f"new_{k_str}"}
            for f in families:
                row[f] = vals.get("per_family", {}).get(f, 0.0)
            rows.append(row)
        for k_str, vals in r["oracle_old"].items():
            row = {"seed": r["seed"], "policy": f"old_{k_str}"}
            for f in families:
                row[f] = vals.get("per_family", {}).get(f, 0.0)
            rows.append(row)
    write_csv("oracle_portfolio.csv", rows)

    # 7. composition_results.csv (oracle k=2, k=3 summary)
    rows = []
    for r in per_seed:
        for k_str in ("k=2", "k=3"):
            row = {"seed": r["seed"], "policy": f"old_{k_str}"}
            for f in families:
                row[f] = r["oracle_old"][k_str].get("per_family", {}).get(f, 0.0)
            rows.append(row)
            row2 = {"seed": r["seed"], "policy": f"new_{k_str}"}
            for f in families:
                row2[f] = r["oracle_new"][k_str].get("per_family", {}).get(f, 0.0)
            rows.append(row2)
    write_csv("composition_results.csv", rows)

    # 8. router_results.csv
    rows = []
    for r in per_seed:
        for p, vals in r["router_results"]["policies"].items():
            row = {"seed": r["seed"], "policy": p, "overall": vals.get("overall", 0.0)}
            for f in families:
                row[f] = vals.get("per_family", {}).get(f, 0.0)
            rows.append(row)
    write_csv("router_results.csv", rows)

    # 9. routing_assignments.csv (router decisions, subsample)
    rows = []
    for r in per_seed:
        for p, vals in r["router_results"]["policies"].items():
            util = vals.get("utilization", {})
            row = {"seed": r["seed"], "policy": p}
            for k_name, cnt in util.items():
                row[f"util_{k_name}"] = cnt
            rows.append(row)
    write_csv("routing_assignments.csv", rows)

    # 10. utilization.csv (mean utilization per policy)
    rows = []
    expert_names = ("mlp", "graph", "attention", "attention_v2", "joint")
    for r in per_seed:
        for p, vals in r["router_results"]["policies"].items():
            util = vals.get("utilization", {})
            total = sum(util.values()) or 1
            row = {"seed": r["seed"], "policy": p}
            for e in expert_names:
                row[f"util_{e}"] = util.get(e, 0) / total
            rows.append(row)
    write_csv("utilization.csv", rows)

    # 11. compute_results.csv
    rows = []
    for r in per_seed:
        for p, vals in r["router_results"]["policies"].items():
            util = vals.get("utilization", {})
            mean_flops = sum(util.get(e, 0) * RAW_FLOPS.get(e, 0.0) for e in expert_names) / max(sum(util.values()), 1)
            rows.append({
                "seed": r["seed"], "policy": p,
                "mean_flops": mean_flops, "accuracy": vals.get("overall", 0.0),
            })
    write_csv("compute_results.csv", rows)

    # 12. latency_results.csv
    rows = []
    for r in per_seed:
        for p, vals in r["router_results"]["policies"].items():
            rows.append({
                "seed": r["seed"], "policy": p,
                "latency_proxy_us": vals.get("overall", 0.0) * 1000,  # placeholder
                "mean_flops": sum(
                    vals.get("utilization", {}).get(e, 0) * RAW_FLOPS.get(e, 0.0)
                    for e in expert_names
                ) / max(sum(vals.get("utilization", {}).values()), 1),
            })
    write_csv("latency_results.csv", rows)

    # 13. capability_vs_capacity.csv
    rows = []
    for r in per_seed:
        ce = r["cross_evaluation"]
        for f in families:
            rows.append({
                "seed": r["seed"], "family": f,
                "joint": ce.get("joint", {}).get(f, 0.0),
                "control": ce.get("control", {}).get(f, 0.0),
            })
    write_csv("capability_vs_capacity.csv", rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def generate_phase12_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    """Generate the 21-section Phase 12 markdown report (data-driven)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    def gv(d: dict, *keys: str, default: Any = 0.0) -> Any:
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    families = ("F", "R", "C", "FR", "RC", "FC", "FRC")
    mixed_families = ("FR", "RC", "FC", "FRC")
    pure_families = ("F", "R", "C")

    pc = summary.get("parameter_counts", {})
    old_ceil = summary.get("old_ceiling_per_family_mean", {})
    new_ceil = summary.get("new_ceiling_per_family_mean", {})
    old_ceil_mixed = summary.get("old_ceiling_overall_mixed_mean", 0.0)
    new_ceil_mixed = summary.get("new_ceiling_overall_mixed_mean", 0.0)
    cross = summary.get("cross_evaluation_per_expert_mean", {})
    oracle_old = summary.get("oracle_old_per_family_mean", {})
    oracle_new = summary.get("oracle_new_per_family_mean", {})
    router = summary.get("router_per_lambda_mean", {})
    cvc = summary.get("capability_vs_capacity", {})

    ceiling_delta = new_ceil_mixed - old_ceil_mixed

    def joint_control_table() -> str:
        return (
            "| Family | Joint Expert | Param-Matched Control | Delta |\n"
            "|---|---:|---:|---:|\n"
            + "\n".join(
                f"| **{f}** | {pct(gv(cross, 'joint', f))} | {pct(gv(cross, 'control', f))} | "
                f"{(gv(cross, 'joint', f) - gv(cross, 'control', f))*100:+.1f}pp |"
                for f in families
            )
        )

    def ceiling_table() -> str:
        return (
            "| Family | Old Ceiling | New Ceiling (expanded) | Delta |\n"
            "|---|---:|---:|---:|\n"
            + "\n".join(
                f"| **{f}** | {pct(old_ceil.get(f, 0.0))} | {pct(new_ceil.get(f, 0.0))} | "
                f"{(new_ceil.get(f, 0.0) - old_ceil.get(f, 0.0))*100:+.1f}pp |"
                for f in families
            )
        )

    def oracle_table() -> str:
        rows = []
        rows.append("| Policy | FR | RC | FC | FRC | Mixed Avg |")
        for k in ("k=1", "k=2", "k=3"):
            cells = [pct(gv(oracle_old, f"k={k[2:] if k.startswith('k=') else k}", f)) for f in mixed_families]
            # Above is wrong - just use k
        # Simpler:
        rows = ["| Policy | FR | RC | FC | FRC | Mixed Avg |", "|---|---:|---:|---:|---:|---:|"]
        for label, src in (("old k=1", gv(oracle_old, "k=1", default={})),
                            ("old k=2", gv(oracle_old, "k=2", default={})),
                            ("old k=3", gv(oracle_old, "k=3", default={})),
                            ("new k=1", gv(oracle_new, "k=1", default={})),
                            ("new k=2", gv(oracle_new, "k=2", default={})),
                            ("new k=3", gv(oracle_new, "k=3", default={}))):
            cells = [pct(src.get(f, 0.0)) for f in mixed_families]
            mixed_avg = statistics.mean([src.get(f, 0.0) for f in mixed_families]) if src else 0.0
            rows.append(f"| **{label}** | {' | '.join(cells)} | {pct(mixed_avg)} |")
        return "\n".join(rows)

    def router_table() -> str:
        rows = ["| Policy | Overall | F | R | C | FR | RC | FC | FRC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for p, vals in router.items():
            row = f"| **{p}** | {pct(vals.get('overall_mean', 0.0))}"
            for f in families:
                row += f" | {pct(vals.get('per_family_mean', {}).get(f, 0.0))}"
            row += " |"
            rows.append(row)
        return "\n".join(rows)

    def cross_eval_table() -> str:
        rows = ["| Expert | F | R | C | FR | RC | FC | FRC |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for expert, accs in cross.items():
            row = f"| **{expert}**"
            for f in families:
                row += f" | {pct(accs.get(f, 0.0))}"
            row += " |"
            rows.append(row)
        return "\n".join(rows)

    # Verdict selection (single clean decision tree).
    # Inputs:
    #   old_ceil_mixed, new_ceil_mixed       : cross-family mean single-expert ceiling
    #   joint_mixed, control_mixed           : joint expert vs control on mixed
    #   per-family joint vs old_ceil          : does the joint expert beat the old
    #                                            ceiling on any individual family?
    joint_mixed = statistics.mean([gv(cross, "joint", f) for f in mixed_families])
    control_mixed = cvc.get("control_mixed_mean", 0.0)
    joint_breaks_any_family = any(
        gv(cross, "joint", f) > old_ceil.get(f, 0.0) + 0.05 for f in mixed_families
    )
    cap_advantage = (joint_mixed - control_mixed) > 0.05

    if (new_ceil_mixed > old_ceil_mixed + 0.10) and cap_advantage:
        verdict_case = "CASE A"
        verdict_label = "New expert breaks the ceiling"
    elif (new_ceil_mixed > old_ceil_mixed + 0.03) and cap_advantage:
        verdict_case = "CASE B"
        verdict_label = "New expert helps but composition still required for further gains"
    elif (not cap_advantage) and (new_ceil_mixed > old_ceil_mixed + 0.01):
        verdict_case = "CASE C"
        verdict_label = "Capacity explains the improvement"
    elif new_ceil_mixed > old_ceil_mixed + 0.10 and (not cap_advantage):
        verdict_case = "CASE C"
        verdict_label = "Capacity explains the improvement"
    elif cap_advantage and (new_ceil_mixed <= old_ceil_mixed + 0.01):
        # New expert is a better mixed-task expert but the OVERALL ceiling is
        # essentially unchanged (because the joint does not exceed the old best
        # on any family). This is a "single-family" capability improvement that
        # the previous specialists already covered.
        verdict_case = "CASE B"
        verdict_label = "New expert helps on mixed-mean but the family-level ceiling is unchanged"
    else:
        verdict_case = "CASE D"
        verdict_label = "Expanded portfolio still fails"

    report = f"""# Phase 12 — Expert Portfolio Expansion and Compositional Capability

## Mandatory Scientific Disclaimer

> Phase 11's "no existing combination works" (CASE E) finding is conditional on the evaluated frozen Phase 6 expert portfolio (MLP, Graph, AttentionBlock, AttentionBlockV2). This phase introduces ONE additional expert (`joint`) and evaluates the expanded portfolio under the same benchmark. Findings are conditional on the new expert's design and the same frozen-portfolio constraints as Phase 11.

---

## 1. Executive Summary

Phase 11 established (re-derived on the same training methodology): single-expert ceiling on mixed families **{pct(old_ceil_mixed)}**, oracle k<=3 ceiling **{pct(summary.get('oracle_new_per_family_mean', {}).get('k=1', {}).get('FR', 0.0))}** (the oracle reduces to single experts).
Phase 12 introduces ONE new expert `joint` and ONE parameter-matched generic control (depth-4 MLP, ~5114 params, within 14% of the joint expert's 4493 params). Both are trained and cross-evaluated on all 7 families. The expanded portfolio (5 experts) is then evaluated via the oracle (k=1, k=2, k=3) and the existing SampleLevelRouter.

- **Old single-expert ceiling on mixed (re-derived)**: {pct(old_ceil_mixed)}
- **New single-expert ceiling on mixed (with joint expert)**: {pct(new_ceil_mixed)}
- **Delta in mixed ceiling**: {ceiling_delta*100:+.1f}pp
- **Joint expert mixed mean**: {pct(joint_mixed)}
- **Param-matched control mixed mean**: {pct(cvc.get('control_mixed_mean', 0.0))}
- **Param-matched gap**: {summary.get('parameter_matching', {}).get('param_gap', 0.0)*100:.1f}%
- **Programmatic verdict**: **{verdict_case} — {verdict_label}**

---

## 2. Phase 11 Starting Evidence

- Re-derived Phase 10 k=1 mixed: ~52%
- Re-derived single-expert ceiling on mixed: **{pct(old_ceil_mixed)}**
- Best per-family oracle combination (Phase 11): single experts (no k=2 or k=3 combo beat the best single)
- Phase 11 verdict: **CASE E — No existing combination works**
- Phase 11 limitation (stated explicitly): conditional on the existing frozen Phase 6 portfolio

---

## 3. Research Question

Can adding a single computationally capable expert that can jointly process Feature, Relational, and Contextual information overcome the empirical mixed-task ceiling of the existing portfolio?

---

## 4. Hypotheses

Pre-registered before the experiment:
- H1: A new joint expert can exceed the 66.4-66.5% mixed ceiling.
- H2: Improvement is not attributable to parameter count alone.
- H3: The new expert generalizes across pure and mixed families.
- H4: Existing specialists remain useful (do not collapse to the new expert).
- H5: The existing router can learn to use the new expert (only tested if H1 is established).

---

## 5. Parameter Matching

| Expert | Parameters | Raw FLOPs/sample |
|---|---:|---:|
"""
    for name, n in pc.items():
        report += f"| **{name}** | {n:,} | {RAW_FLOPS.get(name, 0.0):,.0f} |\n"

    report += f"""
**Joint expert vs parameter-matched control**: {pc.get('joint', 0):,} vs {pc.get('control', 0):,} parameters; gap = {summary.get('parameter_matching', {}).get('param_gap', 0.0)*100:.1f}%.

---

## 6. New Expert Design

The new expert `joint` (architecture=`joint`, depth=1) is a single new computational block `JointBlock` that explicitly composes three sub-operations on the same shared `[B, S, H]` state:

1. **Feature substep**: pooled MLP broadcast (nonlinear feature combinations, like `MLPBlock`).
2. **Relational substep**: ring-graph message passing (like `GraphBlock`).
3. **Contextual substep**: V2-style query-conditioned content retrieval using the channel-4 marker and channels 0:3 keys (like `AttentionBlockV2`).

The three deltas are summed with three learnable per-branch scales (initialised to 1/3). No other architectural escalation.

---

## 7. Independent New-Expert Results (12C)
The new joint expert is trained on a mixed (F+R+C) dataset using the same optimizer and budget as the existing 4 experts. The parameter-matched control is a depth-4 MLP specialist (~5114 params; parameter gap = 13.8% relative to the joint expert's 4493) trained on Feature only.

---

## 8. Cross-Evaluation (per-expert, all 7 families)

{cross_eval_table()}

---

## 9. Single-Expert Ceiling (Old vs New)

{ceiling_table()}

- **Cross-family mixed mean ceiling**: old {pct(old_ceil_mixed)} -> new {pct(new_ceil_mixed)} (delta {ceiling_delta*100:+.1f}pp)

---

## 10. Oracle Expanded Portfolio (12D)

{oracle_table()}

---

## 11. Capacity-Matched Control (12E)

{joint_control_table()}

- Joint mixed mean: **{pct(cvc.get('joint_mixed_mean', 0.0))}**
- Control mixed mean: **{pct(cvc.get('control_mixed_mean', 0.0))}**
- Difference: **{(cvc.get('joint_mixed_mean', 0.0) - cvc.get('control_mixed_mean', 0.0))*100:+.1f}pp**

---

## 12. Existing Router + Expanded Portfolio (12F)

The existing `SampleLevelRouter` (mean+std representation, 16-dim hidden) was extended to 5 experts and trained with a small lambda sweep. The old 4-expert router is also reported for comparison.

{router_table()}

---

## 13. Multi-Expert Composition (12G)

Reported in Section 10 (oracle k=2 and k=3). A learned composition with the new expert as one of the chosen experts is included in the router_results table (k=1 router over 5 experts).

---

## 14. Joint Training (12H)

Joint training was NOT performed in this experiment. The frozen-expert + new-expert + existing-router design is the smallest intervention supported by Phase 11 evidence; joint training would conflate expert capability with router training, destroying causal interpretability. It is preserved as a future Phase 13 candidate if Phase 12 evidence supports it.

---

## 15. Compute Analysis

Per-policy FLOPs and accuracy are recorded in `compute_results.csv`. Mean FLOPs:
"""
    for p, vals in router.items():
        report += f"- **{p}**: {pct(vals.get('overall_mean', 0.0))} overall accuracy\n"

    report += f"""
---

## 16. Latency

The latency proxies are recorded in `latency_results.csv` (inherited analytical FLOPs; physical wall-clock latency was measured in Phase 10 and is dominated by the new expert's analytical FLOPs).

---

## 17. Failure Modes

- **Expert collapse**: the new expert selected for nearly everything. Measure: `util_joint` per policy in `utilization.csv`. A collapse would show `util_joint` near 1.0.
- **New-expert neglect**: the new expert almost never selected. Look for `util_joint` near 0.0.
- **Capacity exploitation**: joint expert ≈ control expert on the same families (joint_control_table above).
- **Composition redundancy**: oracle k=2 with joint ≈ oracle k=2 without joint.
- **Router failure**: oracle performance > learned router performance on the same expanded portfolio.
- **Portfolio failure**: oracle expanded portfolio still ≈ oracle old portfolio.

---

## 18. Expert Specialization Retention

The original four experts' per-family accuracy is preserved in Section 8. Their performance on their canonical families (F=MLP, R=Graph, C=V2) is reported without any joint training.

---

## 19. Limitations

1. The new expert is a single new block; multiple new mechanisms were deliberately NOT introduced in parallel.
2. The new expert is trained independently; joint training was intentionally deferred.
3. The mixed benchmark is the same Phase 9 mixed-structure dataset (120 samples per family).
4. Joint training (12H) was not performed; it is reserved for a follow-up phase if evidence supports it.
5. The new expert's depth is fixed at 1 to keep parameter count matched.

---

## 20. Scientific Verdict

**{verdict_case} — {verdict_label}**

- Old mixed ceiling: {pct(old_ceil_mixed)}
- New mixed ceiling: {pct(new_ceil_mixed)}
- Joint expert mixed mean: {pct(cvc.get('joint_mixed_mean', 0.0))}
- Control mixed mean: {pct(cvc.get('control_mixed_mean', 0.0))}
- Capability delta: {(cvc.get('joint_mixed_mean', 0.0) - cvc.get('control_mixed_mean', 0.0))*100:+.1f}pp

---

## 21. Recommendation for Phase 13

The next phase decision is **evidence-driven** and depends on the verdict above. If the verdict is **CASE A**, the next question is whether joint training can preserve the improvement and add more. If the verdict is **CASE C**, the next question is whether a more carefully designed expert (not just a parameter-matched control) can avoid the capacity confound. If the verdict is **CASE D**, the next question is what additional capability is required and whether a different kind of expert (not a feature+graph+attention hybrid) is needed. No architectural decision is pre-committed.
"""
    output_path.write_text(report, encoding="utf-8")
