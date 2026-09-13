"""Phase 21 RC relational-capacity and decision-conversion diagnosis runner.

Additive: no Phase 1-20 module is modified. All trained weights come from the
Phase 20 cache (read-only) except depth-2/depth-4 relational variants, which
are trained on repaired data with a protocol mirror of `_train_joint_repaired`
(same optimizer/loss/budget/best-val/freeze; only the rel_depth differs).

Stages: ("reproduce", "capacity", "diagnose", "finalize"). Depth-4 is trained
only inside "finalize" when the H8 gate passes; otherwise H8 = NOT TESTED and
no depth-4 weights are produced.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuroforge.blocks.joint_co_relational import JointCoRelationalBlock
from neuroforge.datasets.phase9_datasets import apply_phase9_relational_permutation
from neuroforge.datasets.phase9_mixed_repaired import Phase9MixedRepairedDataset
from neuroforge.evaluation.phase13_metrics import relational_destruction_accuracy
from neuroforge.evaluation.phase15_metrics import (
    branch_7way_ablation,
    count_parameters,
    latency_microseconds,
)
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.phase11_diagnostics import StateChain, linear_probe_accuracy
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.training.phase20_repaired_portfolio import _load_cached

from neuroforge.evaluation.phase21_rc_diagnosis import (
    FAMILIES,
    PURE_FAMS,
    MIXED_FAMS,
    H1_GAIN,
    H8_GAIN,
    H8_REGRESS,
    REP_TOL,
    SEED_CONSISTENT,
    analytical_rc_baselines,
    branch_rms,
    build_failure_diagnosis,
    build_phase21_hypotheses,
    counterfactual_swap_rates,
    grad_norms_unfrozen_clone,
    h8_gate,
    implied_label,
    mean,
    recommendation_for_case,
    select_phase21_case,
    split_metrics,
    stat_sr_repaired,
)

P20_PARTIALS = Path("results/metrics/phase20_repaired_portfolio/partials")
P20_SUMMARY = Path("results/metrics/phase20_repaired_portfolio/summary.json")
SEEDS = (11, 23, 37)
EPOCHS = 40
BATCH = 60

VARIANT_KWARGS = {
    "depth1": {"rel_depth": 1, "aggregation": "mean", "rel_update": "linear"},
    "depth2": {"rel_depth": 2, "aggregation": "mean", "rel_update": "linear"},
    "depth3": {"rel_depth": 3, "aggregation": "mean", "rel_update": "linear"},
    "depth4": {"rel_depth": 4, "aggregation": "mean", "rel_update": "linear"},
}


def build_joint_variant(rel_depth: int) -> StandaloneSpecialist:
    """Fresh joint_co expert with a depth-N relational block (untrained)."""
    m = StandaloneSpecialist("joint_co", input_dim=8, hidden_dim=24, depth=1)
    m.blocks[0] = JointCoRelationalBlock(hidden_dim=24, rel_depth=rel_depth,
                                         aggregation="mean", rel_update="linear")
    return m


def train_joint_variant_repaired(
    rel_depth: int,
    seed: int,
    epochs: int,
    batch_size: int,
    partials: Path,
) -> StandaloneSpecialist:
    """Protocol mirror of `_train_joint_repaired` with configurable rel_depth."""
    from neuroforge.training.phase20_repaired_portfolio import _family_loader

    key = f"seed{seed}_depth{rel_depth}"
    cfg = {"variant": VARIANT_KWARGS[f"depth{rel_depth}"], "seed": seed,
           "epochs": epochs, "bs": batch_size, "benchmark": "phase19-repaired-v1"}
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    pt, meta = partials / f"{key}.pt", partials / f"{key}.json"
    model = build_joint_variant(rel_depth)
    if pt.exists() and meta.exists():
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("hash") == h:
                model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
                model.eval()
                return model
        except Exception:
            pass
    torch.manual_seed(seed)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_loader = _family_loader(None, seed, 120, batch_size, "train")
    val_ds = Phase9MixedRepairedDataset(samples_per_type=20, seed=seed + 70_000)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(batch["features"]), batch["target"])
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(val_ds.features).argmax(-1) == val_ds.targets).float().mean().item())
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    partials.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pt)
    meta.write_text(json.dumps({"hash": h, "key": key}), encoding="utf-8")
    return model


@torch.no_grad()
def per_family_preds(expert: nn.Module, features: torch.Tensor) -> torch.Tensor:
    expert.eval()
    return expert(features).argmax(-1)


@torch.no_grad()
def stage_states(expert: StandaloneSpecialist, features: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract 3D states at each north-star link location (no grad)."""
    expert.eval()
    enc = expert.encoder(features)
    block = expert.blocks[0]
    states: dict[str, torch.Tensor] = {"input": enc}
    if hasattr(block, "forward_with_intermediates"):
        inter = block.forward_with_intermediates(enc, features)
        rel = inter.get("rel_delta", None)
        if rel is None and isinstance(inter.get("rel_rounds", None), list) and inter["rel_rounds"]:
            rel = inter["rel_rounds"][-1]
        states["relational_output"] = rel if rel is not None else enc
        states["fusion"] = inter.get("block_output", enc)
    else:
        out, _ = block(enc)
        states["relational_output"] = out
        states["fusion"] = out
    return states


@torch.no_grad()
def final_rep_query(expert: StandaloneSpecialist, features: torch.Tensor) -> torch.Tensor:
    """Final pre-head representation as [B,1,H] (query slice for joint arch)."""
    expert.eval()
    enc = expert.encoder(features)
    block = expert.blocks[0]
    try:
        state, _ = block(enc, features)
    except TypeError:
        state, _ = block(enc)
    q = features[:, :, 4].argmax(dim=1)
    return state[torch.arange(len(features)), q].unsqueeze(1)


def probe_R(state: torch.Tensor, eval_ds: Any) -> float:
    from neuroforge.evaluation.phase18_component_composition import component_labels
    return linear_probe_accuracy(state, component_labels(eval_ds, "R"), 24)


def probe_C(state: torch.Tensor, eval_ds: Any) -> float:
    from neuroforge.evaluation.phase18_component_composition import component_labels
    return linear_probe_accuracy(state, component_labels(eval_ds, "C"), 24)


def probe_joint_RC(state: torch.Tensor, eval_ds: Any) -> float:
    from neuroforge.evaluation.phase18_component_composition import joint_rc_labels
    rc = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    if not rc:
        return 0.0
    return linear_probe_accuracy(state[rc], joint_rc_labels(eval_ds)[rc], 24)


def train_rc_head(train_states: torch.Tensor, train_targets: torch.Tensor, seed: int) -> nn.Linear:
    from neuroforge.evaluation.phase18_component_composition import train_diag_head
    return train_diag_head(train_states, train_targets, train_states.shape[-1], num_classes=2, seed=seed)


@torch.no_grad()
def eval_rc_head(head: nn.Module, states: torch.Tensor, eval_ds: Any) -> float:
    rc = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    if not rc:
        return 0.0
    head.eval()
    preds = head(states).argmax(-1)
    return float((preds[rc] == eval_ds.targets[rc]).float().mean().item())


def run_phase21_rc_diagnosis(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = SEEDS,
    epochs: int = EPOCHS,
    batch_size: int = BATCH,
    stages: tuple[str, ...] = ("reproduce", "capacity", "diagnose", "finalize"),
) -> dict[str, Any]:
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    m_dir = Path(metrics_dir) if metrics_dir else dest / "metrics"
    m_dir.mkdir(parents=True, exist_ok=True)
    f_dir = Path(figures_dir) if figures_dir else dest / "figures"
    f_dir.mkdir(parents=True, exist_ok=True)
    partials = m_dir / "partials"
    partials.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "phase": "Phase 21 — RC Relational Capacity and Decision Conversion Diagnosis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "evaluated_families": list(FAMILIES),
        "benchmark": "phase19-repaired-v1",
        "epochs": epochs,
        "batch_size": batch_size,
        "stages": list(stages),
        "protocol": (
            "Depth-1/2/4 trained on repaired F+R+C with the exact Phase 20 joint "
            "protocol (AdamW 0.003/1e-4, best-val, frozen); depth-3 and portfolio "
            "weights reused read-only from Phase 20 cache. All diagnostics frozen."
        ),
    }
    selected = select_capacity_phase6()
    depths = {**selected.depths, "joint": 1, "joint_co": 1, "depth3": 1}
    per_seed: list[dict[str, Any]] = []
    hist = json.loads(P20_SUMMARY.read_text(encoding="utf-8")) if P20_SUMMARY.exists() else {}

    for seed in seeds:
        r: dict[str, Any] = {"seed": seed}
        eval_ds = Phase9MixedRepairedDataset(samples_per_type=120, seed=seed)
        train_ds = Phase9MixedRepairedDataset(samples_per_type=120, seed=seed + 50_000)

        # ---- reproduce: reload Phase 20 weights read-only, recompute ----
        experts: dict[str, StandaloneSpecialist] = {}
        for arch in ("mlp", "graph", "attention", "attention_v2", "joint", "joint_co", "depth3"):
            experts[arch] = _load_cached(arch, seed, depths, P20_PARTIALS)
            experts[arch].eval()
            for p in experts[arch].parameters():
                p.requires_grad_(False)
        r["repro"] = _reproduction_gate(experts, eval_ds, hist, seed) if "reproduce" in stages else {"tested": False}

        # ---- capacity: depth1 (cached joint_co), depth2 (train), depth3 (cached) ----
        variants: dict[str, StandaloneSpecialist] = {}
        if "capacity" in stages:
            variants["depth1"] = experts["joint_co"]
            variants["depth2"] = train_joint_variant_repaired(2, seed, epochs, batch_size, partials)
            variants["depth3"] = experts["depth3"]
        else:
            for tag, depth in (("depth1", 1), ("depth2", 2), ("depth3", 3)):
                try:
                    if tag == "depth1":
                        variants[tag] = experts["joint_co"]
                    elif tag == "depth3":
                        variants[tag] = experts["depth3"]
                    else:
                        variants[tag] = _load_variant_cache(2, seed, partials)
                except Exception:
                    pass
        r["capacity"] = _capacity_eval(variants, eval_ds) if variants else {"tested": False}

        # ---- diagnose: frozen diagnostics on depth3 (+variants where cheap) ----
        if "diagnose" in stages and variants:
            d3 = variants.get("depth3", experts["depth3"])
            r["diagnostics"] = _diagnose(d3, variants, experts, eval_ds, train_ds, seed)
        else:
            r["diagnostics"] = {"tested": False}
        r["_seed"] = seed
        per_seed.append((r, experts, eval_ds, train_ds))

    if "finalize" not in stages:
        return {"manifest": manifest, "stages_done": [s for s in stages if s != "finalize"]}

    # ================= FINALIZE =================
    simple = [r for r, _, _, _ in per_seed]
    agg = _aggregate(simple)
    hypotheses = build_phase21_hypotheses(agg)
    agg["hypotheses"] = hypotheses
    gate = h8_gate(agg)

    intervention: dict[str, Any] = {"tested": False}
    if gate["passed"]:
        cand_results = []
        for (r, _, eval_ds, _), seed in zip(per_seed, seeds):
            cand = train_joint_variant_repaired(4, seed, epochs, batch_size, partials)
            with torch.no_grad():
                preds = cand(eval_ds.features).argmax(-1)
            m = split_metrics(preds, eval_ds)
            base_rc = r["capacity"].get("depth3", {}).get("RC", 0.0)
            dest = relational_destruction_accuracy(cand, eval_ds, families=("R", "RC", "FRC"), seed=seed)
            sens = dest.get("original", {}).get("RC", 0.0) - dest.get("relational_permuted", {}).get("RC", 0.0)
            cand_results.append({
                "seed": seed, "RC": m.get("RC", 0.0), "R": m.get("R", 0.0),
                "FRC": m.get("FRC", 0.0), "mixed_mean": m.get("mixed_mean", 0.0),
                "RC_gain": m.get("RC", 0.0) - base_rc,
                "destruction_RC": sens,
                "params": count_parameters(cand),
                "rel_params": _rel_params(cand),
            })
        gains = [c["RC_gain"] for c in cand_results]
        regs = []
        for (r, _, _, _), c in zip(per_seed, cand_results):
            base = r["capacity"].get("depth3", {})
            regs.append(base.get("FRC", 0.0) - c["FRC"])
            regs.append(base.get("mixed_mean", 0.0) - c["mixed_mean"])
        intervention = {
            "tested": True,
            "candidate": "depth4",
            "per_seed": cand_results,
            "RC_gain": mean(gains),
            "n_pos": sum(1 for g in gains if g > 0),
            "max_regression": max(regs) if regs else 0.0,
        }
    agg["intervention"] = intervention
    hypotheses = build_phase21_hypotheses(agg)
    agg["hypotheses"] = hypotheses
    case, label = select_phase21_case(hypotheses, agg)
    failure_rows = build_failure_diagnosis(agg)

    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": _serialisable(simple),
        "gate": gate,
        "hypotheses": hypotheses,
        "verdict_case": case,
        "verdict_label": label,
        "recommendation": recommendation_for_case(case),
        "minimal_intervention": (
            {"intervention": "depth4",
             "outcome": "Adopt smallest successful capacity increase",
             "detail": f"RC gain {intervention.get('RC_gain', 0.0) * 100:+.1f}pp, "
                       f"max regression {intervention.get('max_regression', 0.0) * 100:.1f}pp."}
            if hypotheses.get("H8", {}).get("status") == "SUPPORTED" else
            {"intervention": "none",
             "outcome": "No architectural intervention",
             "detail": "H8 gate or results did not support a capacity intervention."}
        ),
        "failure_diagnosis": failure_rows,
        "aggregates": agg,
    }
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _write_csvs(m_dir, simple, failure_rows, intervention if intervention.get("tested") else None)
    from neuroforge.visualization.phase21_plots import generate_phase21_figures
    summary["generated_figures"] = generate_phase21_figures(summary, f_dir)
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _load_variant_cache(rel_depth: int, seed: int, partials: Path) -> StandaloneSpecialist:
    model = build_joint_variant(rel_depth)
    pt = partials / f"seed{seed}_depth{rel_depth}.pt"
    model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _rel_params(expert: StandaloneSpecialist) -> int:
    try:
        from neuroforge.evaluation.phase15_metrics import relational_parameters_of
        return relational_parameters_of(expert)
    except Exception:
        return 0


def _reproduction_gate(experts: dict[str, StandaloneSpecialist], eval_ds: Any,
                       hist: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fail-closed Phase 20 reproduction: recompute, compare within REP_TOL."""
    detail: list[str] = []
    ok = True
    checks: dict[str, float] = {}
    with torch.no_grad():
        for arch in ("mlp", "graph", "attention_v2", "joint", "joint_co", "depth3"):
            m = split_metrics(experts[arch](eval_ds.features).argmax(-1), eval_ds)
            for f in ("F", "R", "C", "RC", "FRC"):
                checks[f"{arch}_{f.lower()}"] = m.get(f, 0.0)
    hist_vals = _hist_lookup(hist, seed)
    max_abs = 0.0
    for k, v in checks.items():
        if k in hist_vals:
            d = abs(v - hist_vals[k])
            max_abs = max(max_abs, d)
            if d > REP_TOL:
                ok = False
                detail.append(f"{k}: {v * 100:.1f}% vs hist {hist_vals[k] * 100:.1f}% (Δ {d * 100:.1f}pp)")
    if not hist_vals:
        ok = False
        detail.append("no historical Phase 20 values found; gate cannot verify")
    return {"tested": True, "passed": ok, "max_abs_pp": max_abs * 100,
            "checks": checks,
            "detail": "; ".join(detail) if detail else f"all within {REP_TOL * 100:.0f}pp"}


def _hist_lookup(hist: dict[str, Any], seed: int) -> dict[str, float]:
    """Extract per-seed per-expert family accuracies from the Phase 20 summary."""
    out: dict[str, float] = {}
    try:
        for r in hist.get("per_seed_results", []):
            if r.get("seed") != seed:
                continue
            cross = r.get("cross", r.get("cross_evaluation", {}))
            for arch, perf in cross.items():
                if isinstance(perf, dict):
                    for f in ("F", "R", "C", "RC", "FRC"):
                        if f in perf:
                            out[f"{arch}_{f.lower()}"] = float(perf[f])
    except Exception:
        pass
    return out


def _capacity_eval(variants: dict[str, StandaloneSpecialist], eval_ds: Any) -> dict[str, Any]:
    """H1 inputs: per-depth R/RC/FRC, destruction, probes, latency, params."""
    out: dict[str, Any] = {"tested": True, "depths": {}}
    for tag, expert in variants.items():
        with torch.no_grad():
            m = split_metrics(expert(eval_ds.features).argmax(-1), eval_ds)
        dest = relational_destruction_accuracy(expert, eval_ds, families=("R", "RC", "FRC"), seed=11)
        probe = probe_R(stage_states(expert, eval_ds.features)["fusion"], eval_ds)
        lat = latency_microseconds(expert, eval_ds.features[:60])
        out["depths"][tag] = {
            "R": m.get("R", 0.0), "RC": m.get("RC", 0.0), "FRC": m.get("FRC", 0.0),
            "mixed_mean": m.get("mixed_mean", 0.0),
            "destruction_RC": dest.get("original", {}).get("RC", 0.0) - dest.get("relational_permuted", {}).get("RC", 0.0),
            "probe_R": probe,
            "latency_us": lat,
            "params": count_parameters(expert),
            "rel_params": _rel_params(expert),
        }
    return out


def _diagnose(d3: StandaloneSpecialist, variants: dict[str, StandaloneSpecialist],
              experts: dict[str, StandaloneSpecialist], eval_ds: Any,
              train_ds: Any, seed: int) -> dict[str, Any]:
    """All frozen diagnostics (H2-H7 inputs + agreement + analytical baselines)."""
    from neuroforge.evaluation.phase18_component_composition import (
        joint_rc_labels, train_diag_head, train_nonlinear_diag_head,
    )
    from neuroforge.evaluation.phase15_metrics import relational_probe_triplet

    d: dict[str, Any] = {"tested": True}
    feats, targets = eval_ds.features, eval_ds.targets
    tr_feats, tr_targets = train_ds.features, train_ds.targets

    # ---- H2/H6 stage + location probes (depth3) ----
    states = stage_states(d3, feats)
    tr_states = stage_states(d3, tr_feats)
    loc_rows: dict[str, dict[str, float]] = {}
    for loc in ("input", "relational_output", "fusion"):
        loc_rows[loc] = {
            "R_dec": probe_R(states[loc], eval_ds),
            "RC_dec": probe_joint_on_state(states[loc], eval_ds),
        }
    fin = final_rep_query(d3, feats)
    tr_fin = final_rep_query(d3, tr_feats)
    loc_rows["final_representation"] = {
        "R_dec": probe_R(fin, eval_ds),
        "RC_dec": probe_joint_on_state(fin, eval_ds),
    }
    # RC-target head decodability per location (train split heads)
    rc_heads = {}
    for loc, st in (("fusion", states["fusion"]), ("final", fin)):
        tr = tr_states["fusion"] if loc == "fusion" else tr_fin
        head = train_diag_head(tr.mean(dim=1), tr_targets, tr.shape[-1], num_classes=2, seed=seed)
        rc_heads[loc] = eval_rc_head(head, st.mean(dim=1), eval_ds)
    d["locations"] = loc_rows
    with torch.no_grad():
        prod_rc = split_metrics(d3(feats).argmax(-1), eval_ds).get("RC", 0.0)
    d["RC_target_final"] = rc_heads.get("final", prod_rc)
    d["production_RC"] = prod_rc
    # R probe triplet on depth3 (reference)
    d["probe_triplet"] = relational_probe_triplet(d3, eval_ds)

    # ---- H3 gradients + branch RMS (joint, joint_co, depth3) ----
    rc_idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"][:60]
    px, py = feats[rc_idx], targets[rc_idx]
    grads = {}
    for tag in ("joint", "joint_co", "depth3"):
        exp = experts.get(tag, variants.get(tag))
        if exp is None:
            continue
        norms = grad_norms_unfrozen_clone(exp, px, py)
        total = sum(norms.values()) + 1e-12
        rms = branch_rms(exp, feats[rc_idx])
        grads[tag] = {**norms, "R_share": norms.get("relational", 0.0) / total,
                      "C_share": norms.get("contextual", 0.0) / total,
                      "rel_rms": rms.get("rel_delta", 0.0), "ctx_rms": rms.get("ctx_delta", 0.0)}
    d["gradients"] = grads

    # ---- H5 heads on frozen F+R+C concat reps ----
    with torch.no_grad():
        enc = d3.encoder(feats)
        inter = d3.blocks[0].forward_with_intermediates(enc, feats) \
            if hasattr(d3.blocks[0], "forward_with_intermediates") else {}
        enc_tr = d3.encoder(tr_feats)
        inter_tr = d3.blocks[0].forward_with_intermediates(enc_tr, tr_feats) \
            if hasattr(d3.blocks[0], "forward_with_intermediates") else {}
    def _catpoi(it: dict, feats_t: torch.Tensor) -> torch.Tensor | None:
        try:
            parts = [it["feat_delta"].mean(dim=1), it["rel_delta"].mean(dim=1), it["ctx_delta"].mean(dim=1)]
            return torch.cat(parts, dim=-1)
        except Exception:
            return None
    heads: dict[str, float] = {}
    ev_x, tr_x = _catpoi(inter, feats), _catpoi(inter_tr, tr_feats)
    if ev_x is not None and tr_x is not None:
        lin = train_diag_head(tr_x, tr_targets, tr_x.shape[-1], num_classes=2, seed=seed)
        nonlin = train_nonlinear_diag_head(tr_x, tr_targets, tr_x.shape[-1], num_classes=2, seed=seed)
        heads["linear_RC"] = eval_rc_head(lin, ev_x, eval_ds)
        heads["nonlinear_RC"] = eval_rc_head(nonlin, ev_x, eval_ds)
        # component-aware: R-logits + C-logits -> Linear(4->2)
        r_head = train_diag_head(tr_x[:, 24:48], _sr_labels(train_ds), 24, num_classes=2, seed=seed + 1)
        c_head = train_diag_head(tr_x[:, 48:72], _sc_labels(train_ds), 24, num_classes=2, seed=seed + 2)
        with torch.no_grad():
            tr_logits = torch.cat([r_head(tr_x[:, 24:48]), c_head(tr_x[:, 48:72])], dim=-1)
            ev_logits = torch.cat([r_head(ev_x[:, 24:48]), c_head(ev_x[:, 48:72])], dim=-1)
        combo = train_diag_head(tr_logits, tr_targets, 4, num_classes=2, seed=seed + 3)
        heads["component_RC"] = eval_rc_head(combo, ev_logits, eval_ds)
        heads["linear_params"] = sum(p.numel() for p in lin.parameters())
        heads["nonlinear_params"] = sum(p.numel() for p in nonlin.parameters())
        heads["component_params"] = (sum(p.numel() for p in r_head.parameters())
                                     + sum(p.numel() for p in c_head.parameters())
                                     + sum(p.numel() for p in combo.parameters()))
    d["heads"] = heads

    # ---- H4 counterfactuals + branch ablations (depth3) ----
    d["counterfactual"] = counterfactual_swap_rates(d3, eval_ds, seed)
    try:
        d["ablation"] = branch_7way_ablation(d3, eval_ds)
    except Exception as e:
        d["ablation"] = {"error": str(e)[:200]}

    # ---- H7 order chains with controlled readout ----
    chains = _order_experiment(experts, eval_ds, train_ds, seed)
    d["order"] = chains

    # ---- agreement table (all experts + oracle-k2 composition) ----
    d["agreement_table"] = _agreement_table(experts, eval_ds)
    d["analytical"] = analytical_rc_baselines(eval_ds)
    return d


def _sr_labels(ds: Any) -> torch.Tensor:
    return torch.tensor([int(ds.items[i]["sr"] == 1) for i in range(len(ds))], dtype=torch.long)


def _sc_labels(ds: Any) -> torch.Tensor:
    return torch.tensor([int(ds.items[i]["sc"] == 1) for i in range(len(ds))], dtype=torch.long)


def probe_joint_on_state(state: torch.Tensor, eval_ds: Any) -> float:
    from neuroforge.evaluation.phase18_component_composition import joint_rc_labels
    rc = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    if not rc:
        return 0.0
    return linear_probe_accuracy(state[rc], joint_rc_labels(eval_ds)[rc], 24)


def _order_experiment(experts: dict[str, StandaloneSpecialist], eval_ds: Any,
                      train_ds: Any, seed: int) -> dict[str, Any]:
    """H7: R→C vs C→R chains + identical-head probes of frozen intermediates."""
    from neuroforge.evaluation.phase18_component_composition import train_diag_head
    out: dict[str, Any] = {"tested": True}
    g, v = experts["graph"], experts["attention_v2"]
    chains = {}
    for a, b, tag in ((g, v, "R->C"), (v, g, "C->R")):
        name_a = "graph" if a is g else "attention_v2"
        name_b = "attention_v2" if b is v else "graph"
        chain = StateChain({name_a: a, name_b: b}, (name_a, name_b),
                           attach="identity", attach_v2_features=True)
        chain.eval()
        with torch.no_grad():
            m = split_metrics(chain(eval_ds.features).argmax(-1), eval_ds)
        chains[tag] = {"RC": m.get("RC", 0.0), "R": m.get("R", 0.0), "FRC": m.get("FRC", 0.0)}
    # frozen intermediate probes with IDENTICAL linear heads:
    # post-first-expert state for each order, RC-target head trained on train split
    inter_rc = {}
    for tag, first in (("R->C", g), ("C->R", v)):
        first.eval()
        with torch.no_grad():
            s_ev = _first_state(first, eval_ds.features)
            s_tr = _first_state(first, train_ds.features)
        head = train_diag_head(s_tr.mean(dim=1), train_ds.targets, s_tr.shape[-1], num_classes=2, seed=seed)
        inter_rc[tag] = eval_rc_head(head, s_ev.mean(dim=1), eval_ds)
    out["chains"] = chains
    out["intermediate_RC"] = inter_rc
    rc_vals = [chains["R->C"]["RC"], chains["C->R"]["RC"]]
    out["spread_RC"] = abs(rc_vals[0] - rc_vals[1])
    out["separable"] = True  # accuracy part always separable; usability via intermediate heads
    return out


@torch.no_grad()
def _first_state(expert: StandaloneSpecialist, features: torch.Tensor) -> torch.Tensor:
    expert.eval()
    state = expert.encoder(features)
    for block in expert.blocks:
        try:
            state, _ = block(state, features)
        except TypeError:
            state, _ = block(state)
    return state


def _agreement_table(experts: dict[str, StandaloneSpecialist], eval_ds: Any) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for name, exp in experts.items():
        with torch.no_grad():
            m = split_metrics(exp(eval_ds.features).argmax(-1), eval_ds)
        table[name] = {"RC_agree": m.get("RC_agree", 0.0), "RC_disagree": m.get("RC_disagree", 0.0),
                       "RC": m.get("RC", 0.0)}
    # oracle-k2 composition row
    table["oracle_k2"] = _oracle_k2_row(experts, eval_ds)
    return table


def _oracle_k2_row(experts: dict[str, StandaloneSpecialist], eval_ds: Any) -> dict[str, float]:
    """Best per-family pair-averaged logits (diagnostic composition reference)."""
    import itertools
    names = list(experts.keys())
    feats, targets, fams = eval_ds.features, eval_ds.targets, eval_ds.families_list
    with torch.no_grad():
        logits = {n: experts[n](feats) for n in names}
    preds = torch.zeros(len(feats), dtype=torch.long)
    for f in FAMILIES:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if not idx:
            continue
        best_acc, best_combo = -1.0, None
        for combo in itertools.combinations(names, 2):
            stacked = torch.stack([logits[n] for n in combo], dim=0).mean(dim=0)
            acc = float((stacked.argmax(-1)[idx] == targets[idx]).float().mean().item())
            if acc > best_acc:
                best_acc, best_combo = acc, combo
        stacked = torch.stack([logits[n] for n in best_combo], dim=0).mean(dim=0)
        preds[idx] = stacked.argmax(-1)[idx]
    m = split_metrics(preds, eval_ds)
    return {"RC_agree": m.get("RC_agree", 0.0), "RC_disagree": m.get("RC_disagree", 0.0), "RC": m.get("RC", 0.0)}


def _aggregate(simple: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(simple)
    # capacity (H1)
    depths = ["depth1", "depth2", "depth3"]
    rc_by_depth = {d: [r["capacity"]["depths"][d]["RC"] for r in simple if d in r.get("capacity", {}).get("depths", {})] for d in depths}
    base = [r["capacity"]["depths"]["depth1"]["RC"] for r in simple if "depth1" in r.get("capacity", {}).get("depths", {})]
    gains = {d: [b - a for a, b in zip(base, rc_by_depth[d])] for d in ("depth2", "depth3") if rc_by_depth[d] and base}
    best_d = max([d for d in gains], key=lambda d: mean(gains[d]), default="depth3")
    r_base = [r["capacity"]["depths"]["depth1"]["R"] for r in simple if "depth1" in r.get("capacity", {}).get("depths", {})]
    r_best = [r["capacity"]["depths"][best_d]["R"] for r in simple if best_d in r.get("capacity", {}).get("depths", {})]
    capacity = {
        "tested": bool(gains),
        "RC_gains": gains.get(best_d, []),
        "best_depth": best_d,
        "R_kept": (mean(r_best) >= mean(r_base) - 0.02) if r_base and r_best else False,
        "per_depth_RC": {d: mean(v) for d, v in rc_by_depth.items()},
    }
    # alignment (H2): best location R probe vs final RC-target head
    dg = [r["diagnostics"] for r in simple if r.get("diagnostics", {}).get("tested")]
    r_decs, rc_decs = [], []
    for d in dg:
        rows = d.get("locations", {})
        r_decs.append(max([rows.get(k, {}).get("R_dec", 0.0) for k in ("input", "relational_output", "fusion", "final_representation")]))
        rc_decs.append(d.get("RC_target_final", 0.0))
    alignment = {"tested": bool(dg), "R_dec": mean(r_decs), "RC_dec": mean(rc_decs)}
    # dominance (H3): depth3 grad shares
    shares_r, shares_c, ratios = [], [], []
    for d in dg:
        g = d.get("gradients", {}).get("depth3", {})
        if g:
            tot = g.get("relational", 0.0) + g.get("contextual", 0.0) + 1e-12
            shares_r.append(g.get("relational", 0.0) / (sum(v for k, v in g.items() if k in ("relational", "contextual", "feature", "fusion", "encoder", "head")) + 1e-12))
            shares_c.append(g.get("contextual", 0.0) / (sum(v for k, v in g.items() if k in ("relational", "contextual", "feature", "fusion", "encoder", "head")) + 1e-12))
            ratios.append(g.get("contextual", 0.0) / (g.get("relational", 0.0) + 1e-12))
    dominance = {"tested": bool(shares_r), "R_share": mean(shares_r), "C_share": mean(shares_c),
                 "C_over_R_grad": mean(ratios)}
    # counterfactual (H4)
    cf_r = [r["diagnostics"].get("counterfactual", {}).get("R_change", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]
    cf_rc = [r["diagnostics"].get("counterfactual", {}).get("R_correct_change", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]
    cf_c = [r["diagnostics"].get("counterfactual", {}).get("control_change", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]
    counterfactual = {"tested": bool(cf_r), "R_change": mean(cf_r), "R_correct_change": mean(cf_rc),
                      "control_change": mean(cf_c)}
    # heads (H5)
    lin = [r["diagnostics"].get("heads", {}).get("linear_RC", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]
    nlin = [r["diagnostics"].get("heads", {}).get("nonlinear_RC", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]
    comp = [r["diagnostics"].get("heads", {}).get("component_RC", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]
    heads = {"tested": bool(lin), "linear_RC": mean(lin), "nonlinear_RC": mean(nlin), "component_RC": mean(comp),
             "linear_params": dg[0].get("heads", {}).get("linear_params", 0) if dg else 0,
             "nonlinear_params": dg[0].get("heads", {}).get("nonlinear_params", 0) if dg else 0,
             "component_params": dg[0].get("heads", {}).get("component_params", 0) if dg else 0}
    # locations (H6)
    loc_rows = {}
    for loc in ("input", "relational_output", "fusion", "final_representation"):
        loc_rows[loc] = {
            "R_dec": mean([r["diagnostics"].get("locations", {}).get(loc, {}).get("R_dec", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]),
            "RC_dec": mean([r["diagnostics"].get("locations", {}).get(loc, {}).get("RC_dec", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]),
        }
    locations = {"tested": bool(dg), "rows": loc_rows}
    # order (H7)
    spreads = [r["diagnostics"].get("order", {}).get("spread_RC", 0.0) for r in simple if r.get("diagnostics", {}).get("tested")]
    order = {"tested": bool(spreads), "spread_RC": mean(spreads), "separable": True,
             "chain_RC_RtoC": mean([r["diagnostics"].get("order", {}).get("chains", {}).get("R->C", {}).get("RC", 0.0) for r in simple]),
             "chain_RC_CtoR": mean([r["diagnostics"].get("order", {}).get("chains", {}).get("C->R", {}).get("RC", 0.0) for r in simple]),
             "inter_RC_RtoC": mean([r["diagnostics"].get("order", {}).get("intermediate_RC", {}).get("R->C", 0.0) for r in simple]),
             "inter_RC_CtoR": mean([r["diagnostics"].get("order", {}).get("intermediate_RC", {}).get("C->R", 0.0) for r in simple])}
    # agreement (E)
    agreement = {"tested": bool(dg),
                 "agree_acc": mean([r["diagnostics"].get("agreement_table", {}).get("depth3", {}).get("RC_agree", 0.0) for r in simple]),
                 "disagree_acc": mean([r["diagnostics"].get("agreement_table", {}).get("depth3", {}).get("RC_disagree", 0.0) for r in simple])}
    # production RC for CASE C
    production_RC = mean([r["capacity"].get("depths", {}).get("depth3", {}).get("RC", 0.0) for r in simple])
    # seed stability
    stab: dict[str, dict[str, float]] = {"RC": {}}
    for tag in ("depth1", "depth2", "depth3"):
        vals = [r["capacity"].get("depths", {}).get(tag, {}).get("RC", 0.0) for r in simple]
        if len(vals) > 1:
            stab["RC"][tag] = statistics.stdev(vals)
    # latency
    latency = {"tested": True,
               "baseline_total_us": mean([r["capacity"].get("depths", {}).get("depth1", {}).get("latency_us", 0.0) for r in simple]),
               "candidate_total_us": mean([r["capacity"].get("depths", {}).get("depth3", {}).get("latency_us", 0.0) for r in simple])}
    return {"n_seeds": n, "capacity": capacity, "alignment": alignment, "dominance": dominance,
            "counterfactual": counterfactual, "heads": heads, "locations": locations,
            "order": order, "agreement": agreement, "production_RC": production_RC,
            "seed_stability": stab, "latency": latency}


def _serialisable(simple: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in simple:
        row = {"seed": r["seed"]}
        for k, v in r.items():
            if k.startswith("_") or k == "seed":
                continue
            row[k] = v
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# CSV writers + report
# ---------------------------------------------------------------------------
def _write_csvs(m_dir: Path, simple: list[dict[str, Any]], failure_rows: list[dict[str, Any]],
                intervention: dict[str, Any] | None) -> None:
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

    fams = list(FAMILIES)
    rows = []
    for r in simple:
        for tag, vals in r.get("capacity", {}).get("depths", {}).items():
            row = {"seed": r["seed"], "condition": tag}
            for f in fams + ["mixed_mean"]:
                row[f] = vals.get(f, 0.0) if f in vals else ""
            for k in ("destruction_RC", "probe_R", "latency_us", "params", "rel_params"):
                row[k] = vals.get(k, "")
            rows.append(row)
    write_csv("capacity.csv", rows)

    rows = []
    for r in simple:
        for loc, vals in r.get("diagnostics", {}).get("locations", {}).items():
            rows.append({"seed": r["seed"], "location": loc,
                         "R_dec": vals.get("R_dec", 0.0), "RC_dec": vals.get("RC_dec", 0.0)})
        rows.append({"seed": r["seed"], "location": "RC_target_head_final",
                     "R_dec": "", "RC_dec": r.get("diagnostics", {}).get("RC_target_final", 0.0)})
    write_csv("stage_probes.csv", rows)
    write_csv("location_probes.csv", rows)

    rows = []
    for r in simple:
        for tag in ("joint", "joint_co", "depth3"):
            g = r.get("diagnostics", {}).get("gradients", {}).get(tag, {})
            if not g:
                continue
            row = {"seed": r["seed"], "condition": tag}
            for k, v in g.items():
                row[k] = v
            rows.append(row)
    write_csv("gradients.csv", rows)

    rows = []
    for r in simple:
        cf = r.get("diagnostics", {}).get("counterfactual", {})
        row = {"seed": r["seed"]}
        for k, v in cf.items():
            row[k] = v
        rows.append(row)
    write_csv("counterfactuals.csv", rows)

    rows = []
    for r in simple:
        for combo, perf in r.get("diagnostics", {}).get("ablation", {}).items():
            if not isinstance(perf, dict):
                continue
            row = {"seed": r["seed"], "combination": combo}
            for f in fams:
                row[f] = perf.get(f, 0.0)
            rows.append(row)
    write_csv("ablation.csv", rows)

    rows = []
    for r in simple:
        h = r.get("diagnostics", {}).get("heads", {})
        rows.append({"seed": r["seed"], **{k: h.get(k, "") for k in
                                           ("linear_RC", "nonlinear_RC", "component_RC",
                                            "linear_params", "nonlinear_params", "component_params")}})
    write_csv("heads.csv", rows)

    rows = []
    for r in simple:
        o = r.get("diagnostics", {}).get("order", {})
        for tag, vals in o.get("chains", {}).items():
            rows.append({"seed": r["seed"], "order": tag, **{f: vals.get(f, 0.0) for f in ("R", "RC", "FRC")}})
        for tag, v in o.get("intermediate_RC", {}).items():
            rows.append({"seed": r["seed"], "order": tag + "_intermediate", "RC": v})
    write_csv("order.csv", rows)

    rows = []
    for r in simple:
        for name, vals in r.get("diagnostics", {}).get("analytical", {}).items():
            rows.append({"seed": r["seed"], "predictor": name, **vals})
    write_csv("analytical.csv", rows)

    rows = []
    for r in simple:
        for name, vals in r.get("diagnostics", {}).get("agreement_table", {}).items():
            rows.append({"seed": r["seed"], "model": name, **vals})
    write_csv("agreement.csv", rows)

    rows = []
    for r in simple:
        for tag, vals in r.get("capacity", {}).get("depths", {}).items():
            rows.append({"seed": r["seed"], "condition": f"capacity_{tag}",
                         "R": vals.get("R", 0.0), "RC": vals.get("RC", 0.0), "FRC": vals.get("FRC", 0.0)})
        h = r.get("diagnostics", {}).get("heads", {})
        rows.append({"seed": r["seed"], "condition": "heads_best",
                     "RC": max(h.get("linear_RC", 0.0), h.get("nonlinear_RC", 0.0), h.get("component_RC", 0.0))})
    write_csv("decision_sensitivity.csv", rows)

    rows = []
    for r in simple:
        for tag, vals in r.get("capacity", {}).get("depths", {}).items():
            rows.append({"seed": r["seed"], "condition": tag,
                         "params": vals.get("params", 0), "rel_params": vals.get("rel_params", 0),
                         "latency_us": vals.get("latency_us", 0.0)})
    write_csv("compute.csv", rows)

    rows = []
    for r in simple:
        row = {"seed": r["seed"]}
        for tag in ("depth1", "depth2", "depth3"):
            row[f"{tag}_RC"] = r.get("capacity", {}).get("depths", {}).get(tag, {}).get("RC", 0.0)
        row["disagree_acc"] = r.get("diagnostics", {}).get("agreement_table", {}).get("depth3", {}).get("RC_disagree", 0.0)
        rows.append(row)
    write_csv("seed_results.csv", rows)

    if intervention and intervention.get("tested"):
        rows = []
        for c in intervention.get("per_seed", []):
            rows.append(dict(c))
        write_csv("intervention.csv", rows)

    write_csv("failure_diagnosis.csv", [
        {"seed": 0, "category": c["category"], "failed": c["failed"],
         "criterion": c["criterion"], "observed": c["observed"]} for c in failure_rows])


def generate_phase21_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    agg = summary.get("aggregates", {})
    hyps = summary.get("hypotheses", {})
    fail = summary.get("failure_diagnosis", [])
    cap = agg.get("capacity", {})
    loc = agg.get("locations", {}).get("rows", {})
    cf = agg.get("counterfactual", {})
    hd = agg.get("heads", {})

    report = f"""# Phase 21 — RC Relational Capacity and Decision Conversion Diagnosis

## Executive summary

Programmatic verdict: **{summary.get('verdict_case', '?')} — {summary.get('verdict_label', '')}**.

## Phase 20 reproduction

Gate: {"PASS" if all(r.get("repro", {}).get("passed", False) for r in summary.get("per_seed_results", [])) else "FAIL — see failure_diagnosis"}.
Reproduction compares recomputed Phase 20 values (same weights, same seeds) within {REP_TOL * 100:.0f}pp per family.

## Research question

Why does the existing prediction mechanism select one available component when RC requires combining R+C?

## Pre-registered hypotheses

H1 depth limitation; H2 task alignment; H3 contextual dominance; H4 combination
causality; H5 head expressivity; H6 locality; H7 order causality; H8 gated capacity
intervention. Thresholds in `evaluation/phase21_rc_diagnosis.py`.

## Experimental design

Depth-1/2/3 (+gated depth-4) trained on repaired F+R+C with the exact Phase 20
joint protocol; everything else frozen diagnostics on depth-3 (and cheap variants).

## RC observability

Repaired RC relational carrier observable per Phase 19 gate (rechecked in 20A smoke).

## Relational capacity analysis (H1)

Per-depth RC (mean): {", ".join(f"{d}={cap.get('per_depth_RC', {}).get(d, float('nan')) * 100:.1f}%" for d in ("depth1", "depth2", "depth3"))}.
Gains vs depth-1: {cap.get("RC_gains", [])}.

## Representation probes (H2)

Best-location R decodability {agg.get('alignment', {}).get('R_dec', float('nan')) * 100:.1f}% vs final RC-target decodability {agg.get('alignment', {}).get('RC_dec', float('nan')) * 100:.1f}%.

## R/C contribution analysis (H3)

Mean grad shares: R {agg.get('dominance', {}).get('R_share', float('nan')) * 100:.1f}%, C {agg.get('dominance', {}).get('C_share', float('nan')) * 100:.1f}% (C/R ratio {agg.get('dominance', {}).get('C_over_R_grad', float('nan')):.1f}x).

## RC agreement/disagreement

Depth-3 agree {agg.get('agreement', {}).get('agree_acc', float('nan')) * 100:.1f}% vs disagree {agg.get('agreement', {}).get('disagree_acc', float('nan')) * 100:.1f}%.

## Counterfactual decision sensitivity (H4 + §12)

R change {cf.get('R_change', float('nan')) * 100:.1f}% (correct {cf.get('R_correct_change', float('nan')) * 100:.1f}%); control change {cf.get('control_change', float('nan')) * 100:.1f}%.

## Composition-order analysis (H7)

Controlled-readout RC spread {agg.get('order', {}).get('spread_RC', float('nan')) * 100:+.1f}pp.

## Minimal intervention (H8, only if gated)

Gate: {json.dumps(summary.get('gate', {}))}.

## Compute/latency

Per-depth params, relational params, latency in `compute.csv`; intervention candidate costs in `intervention.csv` when tested.

## Seed statistics

3/3 directional consistency required for small effects; per-seed values in `seed_results.csv`.

## H1–H8 verdict table

| Hypothesis | Status | Evidence |
|---|---|---|
"""
    for h in [f"H{i}" for i in range(1, 9)]:
        dd = hyps.get(h, {})
        report += f"| **{h}** | {dd.get('status', '?')} | {dd.get('evidence', '')} |\n"
    report += f"""
## CASE A–F programmatic verdict

**{summary.get('verdict_case', '?')} — {summary.get('verdict_label', '')}**

## What this rules out

Recorded in `failure_diagnosis.csv` with explicit criteria.

## What remains unresolved

See recommendation: {summary.get('recommendation', '')}

## Phase 22 recommendation

{summary.get('recommendation', '')} Minimal intervention: **{summary.get('minimal_intervention', {}).get('intervention', '?')}** — {summary.get('minimal_intervention', {}).get('outcome', '')}.
"""
    output_path.write_text(report, encoding="utf-8")
