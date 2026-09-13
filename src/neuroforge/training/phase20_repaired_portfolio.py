"""Phase 20 clean portfolio re-evaluation runner (repaired benchmark ONLY).

Additive: no Phase 1-19 module is modified. Experts are freshly trained on
repaired data via local trainers that mirror the historical protocols exactly
(same optimizer/lr/budget/best-val/freeze); only the data source differs,
because pre-repair checkpoints observe a different carrier channel and are
therefore invalid. Router/composition/oracle machinery is reused unchanged
(`_train_compute_aware_router`, `_oracle_portfolio`, `StateChain`).

Staged + resumable: `stages` subset of ("experts", "routers", "finalize").
Expert/router weights are cached under `<metrics_dir>/partials/`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from neuroforge.blocks.joint_co_relational import JointCoRelationalBlock
from neuroforge.datasets.phase9_mixed_repaired import (
    CONSTRUCTION_VERSION,
    RELATIONAL_CHANNEL,
    Phase9MixedRepairedDataset,
)
from neuroforge.datasets.phase9_datasets import apply_phase9_relational_permutation
from neuroforge.evaluation.phase13_metrics import relational_destruction_accuracy
from neuroforge.evaluation.phase19_benchmark_validation import autocorr_align
from neuroforge.evaluation.phase20_repaired_portfolio import (
    FAMILIES,
    MIXED_FAMS,
    analytical_flops,
    assert_repaired,
    build_failure_diagnosis,
    build_phase20_hypotheses,
    check_zero_overlap,
    dataset_fingerprint,
    mean,
    per_family_accuracy,
    recommendation_for_case,
    select_phase20_case,
)
from neuroforge.evaluation.specialization import select_capacity_phase6
from neuroforge.models.phase11_diagnostics import StateChain, linear_probe_accuracy
from neuroforge.models.specialists import StandaloneSpecialist
from neuroforge.routing.learned_router import SampleLevelRouter
from neuroforge.training.phase12_expert_portfolio import (
    _oracle_portfolio,
    _single_expert_ceiling,
    _train_compute_aware_router,
)

EXPERTS_20 = ("mlp", "graph", "attention", "attention_v2", "joint", "joint_co", "depth3")
EXPERT_TRAIN_FAMILY = {
    "mlp": "F", "graph": "R", "attention": "C", "attention_v2": "C",
    "joint": None, "joint_co": None, "depth3": None,
}
COMPUTE_LAMBDAS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
ORDER_CHAINS = (
    ("mlp", "graph"), ("graph", "mlp"),
    ("graph", "attention_v2"), ("attention_v2", "graph"),
    ("mlp", "attention_v2"), ("attention_v2", "mlp"),
)


# ---------------------------------------------------------------------------
# Repaired-data trainers (protocol mirrors of _train_expert / _train_joint_on_mixed)
# ---------------------------------------------------------------------------
def _family_loader(
    family: str | None, seed: int, samples_per_type: int, batch_size: int, kind: str
) -> DataLoader[dict[str, torch.Tensor]]:
    """DataLoader over repaired family subsets (kind=train/val)."""
    ds = Phase9MixedRepairedDataset(samples_per_type=samples_per_type, seed=seed)
    assert_repaired(ds)
    if family is None:
        wanted = ("F", "R", "C")
    else:
        wanted = (family,)
    items = [ds.items[i] for i in range(len(ds)) if ds.items[i]["family"] in wanted]

    class _DS(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            it = items[idx]
            return {"features": it["features"], "target": it["target"]}

    return DataLoader(_DS(), batch_size=batch_size, shuffle=(kind == "train"))


def _train_specialist_repaired(
    architecture: str,
    family: str,
    depth: int,
    seed: int,
    epochs: int,
    batch_size: int,
) -> StandaloneSpecialist:
    """Mirror of `_train_expert`: same optimizer/loss/budget/best-val/freeze."""
    torch.manual_seed(seed)
    model = StandaloneSpecialist(architecture, input_dim=8, hidden_dim=24, depth=depth)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_loader = _family_loader(family, seed, 120, batch_size, "train")
    val_ds = Phase9MixedRepairedDataset(samples_per_type=40, seed=seed + 70_000)
    val_idx = [i for i, fm in enumerate(val_ds.families_list) if fm == family]
    val_x = val_ds.features[val_idx]
    val_y = val_ds.targets[val_idx]
    best, best_state, curve = -1.0, None, []
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            loss = F.cross_entropy(model(batch["features"]), batch["target"])
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(val_x).argmax(-1) == val_y).float().mean().item())
        curve.append(acc)
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _train_joint_repaired(
    architecture: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> StandaloneSpecialist:
    """Mirror of `_train_joint_on_mixed` on repaired F+R+C (best-val, frozen)."""
    torch.manual_seed(seed)
    if architecture == "depth3":
        model = StandaloneSpecialist("joint_co", input_dim=8, hidden_dim=24, depth=1)
        model.blocks[0] = JointCoRelationalBlock(
            hidden_dim=24, rel_depth=3, aggregation="mean", rel_update="linear")
    else:
        model = StandaloneSpecialist(architecture, input_dim=8, hidden_dim=24, depth=1)
    optim = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    train_loader = _family_loader(None, seed, 120, batch_size, "train")
    val_ds = Phase9MixedRepairedDataset(samples_per_type=20, seed=seed + 70_000)
    best, best_state, curve = -1.0, None, []
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            loss = F.cross_entropy(model(batch["features"]), batch["target"])
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(val_ds.features).argmax(-1) == val_ds.targets).float().mean().item())
        curve.append(acc)
        if acc > best:
            best, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _cfg_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _cached_train(kind: str, key: str, cfg: dict[str, Any], partials: Path, build) -> Any:
    h = _cfg_hash({**cfg, "benchmark": CONSTRUCTION_VERSION})
    pt, meta = partials / f"{key}.pt", partials / f"{key}.json"
    if pt.exists() and meta.exists():
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("hash") == h:
                obj = build(empty=True)
                obj.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
                obj.eval()
                return obj
        except Exception:
            pass
    obj = build(empty=False)
    partials.mkdir(parents=True, exist_ok=True)
    torch.save(obj.state_dict(), pt)
    meta.write_text(json.dumps({"hash": h, "key": key}), encoding="utf-8")
    return obj


# ---------------------------------------------------------------------------
# Diagnostic helpers on repaired data
# ---------------------------------------------------------------------------
@torch.no_grad()
def _depth3_states(expert: StandaloneSpecialist, features: torch.Tensor) -> torch.Tensor:
    expert.eval()
    enc = expert.encoder(features)
    state, _ = expert.blocks[0](enc, features)
    return state


def _stat_sr(features: torch.Tensor) -> torch.Tensor:
    x = features[:, :, RELATIONAL_CHANNEL].float()
    return ((x * torch.roll(x, shifts=-1, dims=1)).mean(dim=1) > 0).long()


def _probe_sc_hat(train_ds: Any, eval_ds: Any, seed: int) -> torch.Tensor:
    """ch3 linear probe for sc, fitted on train half of the TRAIN ds, applied to eval."""
    ch3 = train_ds.features[:, :, 3]
    sc = torch.tensor([int(train_ds.items[i]["sc"] == 1) for i in range(len(train_ds))])
    n = len(train_ds)
    cut = n // 2
    Xtr, Ytr = ch3[:cut].float(), sc[:cut]
    Y1h = torch.zeros(len(Xtr), 2)
    Y1h[torch.arange(len(Xtr)), Ytr.long()] = 1.0
    try:
        W = torch.linalg.solve(Xtr.t() @ Xtr + 1e-2 * torch.eye(12), Xtr.t() @ Y1h)
    except Exception:
        W = torch.linalg.lstsq(Xtr, Y1h).solution
    return (eval_ds.features[:, :, 3].float() @ W).argmax(dim=1)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_phase20_repaired_portfolio(
    output_dir: str | Path,
    metrics_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    seeds: tuple[int, ...] = (11, 23, 37),
    expert_epochs: int = 40,
    router_epochs: int = 30,
    samples_per_type: int = 120,
    batch_size: int = 60,
    stages: tuple[str, ...] = ("experts", "routers", "finalize"),
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
        "phase": "Phase 20 — Clean Portfolio Re-evaluation on Repaired Mixed Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "seeds": list(seeds),
        "benchmark_version": CONSTRUCTION_VERSION,
        "evaluated_families": list(FAMILIES),
        "samples_per_type": samples_per_type,
        "expert_epochs": expert_epochs,
        "router_epochs": router_epochs,
        "portfolio": list(EXPERTS_20),
        "stages": list(stages),
        "protocol": (
            "Fresh training on repaired data (mirrored protocols); pre-repair "
            "checkpoints invalid (carrier moved ch0→ch5) and never used. Router "
            "machinery reused unchanged (num_experts=7 per Phase 12 4→5 precedent). "
            "Router frozen w.r.t. architecture/loss/temperature."
        ),
    }

    selected = select_capacity_phase6()
    depths = {**selected.depths, "joint": 1, "joint_co": 1, "depth3": 1}
    per_seed: list[dict[str, Any]] = []

    for seed in seeds:
        r: dict[str, Any] = {"seed": seed}
        # ---- 20B: fresh splits (repaired only) ----
        train_ds = Phase9MixedRepairedDataset(samples_per_type=samples_per_type, seed=seed + 50_000)
        val_ds = Phase9MixedRepairedDataset(samples_per_type=40, seed=seed + 60_000)
        eval_ds = Phase9MixedRepairedDataset(samples_per_type=samples_per_type, seed=seed)
        for d in (train_ds, val_ds, eval_ds):
            assert_repaired(d)
        r["fingerprints"] = {
            "train": dataset_fingerprint(train_ds),
            "val": dataset_fingerprint(val_ds),
            "eval": dataset_fingerprint(eval_ds),
            "train_test_overlap": check_zero_overlap(train_ds, eval_ds),
            "val_test_overlap": check_zero_overlap(val_ds, eval_ds),
        }
        # ---- 20A: smoke (repaired validity spot-checks) ----
        smoke = {
            "tested": True,
            "r_align_R": autocorr_align(eval_ds.features, eval_ds.items, eval_ds.families_list, ("R",), RELATIONAL_CHANNEL),
            "r_align_RC": autocorr_align(eval_ds.features, eval_ds.items, eval_ds.families_list, ("RC",), RELATIONAL_CHANNEL),
            "r_align_FRC": autocorr_align(eval_ds.features, eval_ds.items, eval_ds.families_list, ("FRC",), RELATIONAL_CHANNEL),
        }
        smoke["passed"] = smoke["r_align_R"] >= 0.90 and smoke["r_align_RC"] >= 0.90 and smoke["r_align_FRC"] >= 0.90
        smoke["detail"] = "repaired R observable on R/RC/FRC" if smoke["passed"] else "SMOKE FAILED"
        r["smoke"] = smoke

        # ---- experts (cached) ----
        experts: dict[str, StandaloneSpecialist] = {}
        if "experts" in stages:
            for arch in ("mlp", "graph", "attention", "attention_v2"):
                cfg = {"arch": arch, "fam": EXPERT_TRAIN_FAMILY[arch], "depth": depths[arch],
                       "seed": seed, "epochs": expert_epochs, "bs": batch_size}
                experts[arch] = _cached_train(
                    "expert", f"seed{seed}_{arch}", cfg, partials,
                    lambda empty, arch=arch, cfg=cfg: (
                        StandaloneSpecialist(arch, input_dim=8, hidden_dim=24, depth=cfg["depth"])
                        if empty else _train_specialist_repaired(
                            arch, EXPERT_TRAIN_FAMILY[arch], cfg["depth"], seed, expert_epochs, batch_size)))
            for arch in ("joint", "joint_co", "depth3"):
                cfg = {"arch": arch, "seed": seed, "epochs": expert_epochs, "bs": batch_size}
                experts[arch] = _cached_train(
                    "expert", f"seed{seed}_{arch}", cfg, partials,
                    lambda empty, arch=arch: (
                        _empty_joint(arch) if empty
                        else _train_joint_repaired(arch, seed, expert_epochs, batch_size)))
        else:
            for arch in EXPERTS_20:
                experts[arch] = _load_cached(arch, seed, depths, partials)
        r["_experts"] = experts
        r["_splits"] = (train_ds, val_ds, eval_ds)
        per_seed.append(r)

    if "routers" in stages:
        for r in per_seed:
            seed = r["seed"]
            train_ds, _, eval_ds = r["_splits"]
            experts = r["_experts"]
            if not experts:
                for arch in EXPERTS_20:
                    experts[arch] = _load_cached(arch, seed, depths, partials)
            # repaired all-family router loaders
            train_loader = _dict_loader(train_ds, batch_size)
            val_ds = Phase9MixedRepairedDataset(samples_per_type=40, seed=seed + 60_000)
            val_loader = _dict_loader(val_ds, batch_size)
            flops = {n: analytical_flops(experts[n]) for n in EXPERTS_20}
            norm_costs = {n: flops[n] / max(flops.values()) for n in EXPERTS_20}
            r["_flops"] = flops
            # learned router λ=0
            cfg = {"seed": seed, "lam": 0.0, "epochs": router_epochs, "nexp": 7}
            r["_router0"] = _cached_train(
                "router", f"seed{seed}_router_lam0", cfg, partials,
                lambda empty: _build_router(empty, experts, EXPERTS_20, norm_costs, 0.0,
                                            train_loader, val_loader, router_epochs))
            # compute-aware sweep
            sweep = {}
            for lam in COMPUTE_LAMBDAS:
                if lam == 0.0:
                    sweep[lam] = r["_router0"]
                    continue
                cfg = {"seed": seed, "lam": lam, "epochs": router_epochs, "nexp": 7}
                sweep[lam] = _cached_train(
                    "router", f"seed{seed}_router_lam{lam}", cfg, partials,
                    lambda empty, lam=lam: _build_router(empty, experts, EXPERTS_20, norm_costs, lam,
                                                         train_loader, val_loader, router_epochs))
            r["_sweep"] = sweep
            r["_norm_costs"] = norm_costs

    if "finalize" not in stages:
        return {"manifest": manifest, "stages_done": [s for s in stages if s != "finalize"]}

    # ================= FINALIZE (evals only; models loaded, cached) =================
    for r in per_seed:
        seed = r["seed"]
        train_ds, _, eval_ds = r["_splits"]
        experts = r["_experts"]
        if not experts:
            for arch in EXPERTS_20:
                experts[arch] = _load_cached(arch, seed, depths, partials)
        if "_router0" not in r:
            train_loader = _dict_loader(train_ds, batch_size)
            val_ds = Phase9MixedRepairedDataset(samples_per_type=40, seed=seed + 60_000)
            val_loader = _dict_loader(val_ds, batch_size)
            flops = {n: analytical_flops(experts[n]) for n in EXPERTS_20}
            norm_costs = {n: flops[n] / max(flops.values()) for n in EXPERTS_20}
            r["_flops"] = flops
            cfg = {"seed": seed, "lam": 0.0, "epochs": router_epochs, "nexp": 7}
            r["_router0"] = _cached_train(
                "router", f"seed{seed}_router_lam0", cfg, partials,
                lambda empty: _build_router(empty, experts, EXPERTS_20, norm_costs, 0.0,
                                            train_loader, val_loader, router_epochs))
            sweep = {}
            for lam in COMPUTE_LAMBDAS:
                if lam == 0.0:
                    sweep[lam] = r["_router0"]
                    continue
                cfg = {"seed": seed, "lam": lam, "epochs": router_epochs, "nexp": 7}
                sweep[lam] = _cached_train(
                    "router", f"seed{seed}_router_lam{lam}", cfg, partials,
                    lambda empty, lam=lam: _build_router(empty, experts, EXPERTS_20, norm_costs, lam,
                                                         train_loader, val_loader, router_epochs))
            r["_sweep"] = sweep
            r["_norm_costs"] = norm_costs
        feats, targets, fams = eval_ds.features, eval_ds.targets, eval_ds.families_list
        # 20C/20D cross matrix + logits
        logits: dict[str, torch.Tensor] = {}
        cross: dict[str, dict[str, float]] = {}
        for name, exp in experts.items():
            exp.eval()
            with torch.no_grad():
                logits[name] = exp(feats)
            cross[name] = per_family_accuracy(exp, feats, targets, fams)
        r["cross"] = cross
        # 20E ceiling (new) + 20F oracle k=1..3
        r["ceiling"] = _single_expert_ceiling(logits, targets, fams)
        r["oracle"] = {f"k={k}": _oracle_portfolio(logits, targets, fams, k) for k in (1, 2, 3)}
        # 20F sequential chains (frozen machinery)
        chain_evals: dict[str, dict[str, float]] = {}
        sub = {k: experts[k] for k in ("mlp", "graph", "attention_v2")}
        for a, b in ORDER_CHAINS:
            chain = StateChain(sub, (a, b), attach="identity", attach_v2_features=True)
            chain.eval()
            with torch.no_grad():
                preds = chain(feats).argmax(-1)
            per_fam = {}
            for f in FAMILIES:
                idx = [i for i, fm in enumerate(fams) if fm == f]
                if idx:
                    per_fam[f] = float((preds[idx] == targets[idx]).float().mean().item())
            chain_evals[f"{a}->{b}"] = per_fam
        r["chains"] = chain_evals
        # random routing (seeded)
        g = torch.Generator().manual_seed(seed + 999)
        choice = torch.randint(len(EXPERTS_20), (len(feats),), generator=g)
        rand_preds = torch.stack([logits[EXPERTS_20[int(c)]][i].argmax() for i, c in enumerate(choice)])
        r["random_routing"] = _fam_acc(rand_preds, targets, fams)
        # 20H learned router eval + 20I diagnostics
        router = r["_router0"]
        router.eval()
        with torch.no_grad():
            dec = router(feats, mode="hard")
            chosen = dec.selected_experts
            ent = dec.entropy
        routed = torch.stack([logits[EXPERTS_20[int(c)]][i].argmax() for i, c in enumerate(chosen)])
        r["learned_routing"] = _fam_acc(routed, targets, fams)
        fam_expert: dict[str, Counter] = {}
        for i, fm in enumerate(fams):
            fam_expert.setdefault(fm, Counter())[EXPERTS_20[int(chosen[i].item())]] += 1
        r["routing_diag"] = {
            "utilization": dict(Counter(EXPERTS_20[int(c.item() if hasattr(c, "item") else c)] for c in chosen)),
            "mean_entropy": float(ent.mean().item()),
            "family_expert": {f: dict(c) for f, c in fam_expert.items()},
        }
        # 20J sweep eval
        sweep_evals = {}
        for lam, rt in r["_sweep"].items():
            rt.eval()
            with torch.no_grad():
                ch = rt(feats, mode="hard").selected_experts
            pr = torch.stack([logits[EXPERTS_20[int(c)]][i].argmax() for i, c in enumerate(ch)])
            acc = _fam_acc(pr, targets, fams)
            lat = _router_latency(rt, experts, feats)
            sweep_evals[f"lambda={lam}"] = {**acc, "latency_us": lat}
        r["sweep"] = sweep_evals
        # 20K/20L component semantics (repaired RC/FRC)
        r["semantics"] = _component_semantics(experts["depth3"], eval_ds, seed)
        # 20M probes on depth3
        r["probes"] = _probe_bundle(experts["depth3"], eval_ds)
        # 20N destruction: graph, joint_co, best composition
        cand_names = _best_compositions(r)
        dest = {}
        for name in ("graph", "joint_co"):
            dest[name] = relational_destruction_accuracy(experts[name], eval_ds, seed=seed)
        dest["bestcomp"] = _destruction_of_best(r, cand_names, eval_ds, seed)
        dest["bestcomp_name"] = cand_names[0] if cand_names else "none"
        r["destruction"] = dest
        # ceiling inputs for H8
        with torch.no_grad():
            r["_logits"] = {n: logits[n].clone() for n in logits}
        # measured latency per expert (for the latency aggregate + CSV)
        r["latencies"] = {}
        for name, exp in experts.items():
            exp.eval()
            sample = feats[:60]
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(5):
                    _ = exp(sample)
                r["latencies"][name] = (time.perf_counter() - t0) / 5 / len(sample) * 1e6

    # ================= aggregates =================
    agg = _aggregate(per_seed)
    hypotheses = build_phase20_hypotheses(agg)
    agg["hypotheses"] = hypotheses
    case, label = select_phase20_case(hypotheses, agg)
    failure_rows = build_failure_diagnosis(agg)

    summary: dict[str, Any] = {
        "manifest": manifest,
        "per_seed_results": _serialisable(per_seed),
        "cross_mean": agg["cross_mean"],
        "hypotheses": hypotheses,
        "verdict_case": case,
        "verdict_label": label,
        "recommendation": recommendation_for_case(case),
        "minimal_intervention": {"intervention": "none",
                                 "outcome": "NO INTERVENTION (baseline phase)",
                                 "detail": "Phase 20 establishes the post-repair baseline; fixes forbidden by 20O."},
        "failure_diagnosis": failure_rows,
        "aggregates": agg,
    }
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (m_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _write_csvs(m_dir, per_seed, failure_rows)
    from neuroforge.visualization.phase20_plots import generate_phase20_figures
    summary["generated_figures"] = generate_phase20_figures(summary, f_dir)
    with (m_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _empty_joint(arch: str) -> StandaloneSpecialist:
    if arch == "depth3":
        m = StandaloneSpecialist("joint_co", input_dim=8, hidden_dim=24, depth=1)
        m.blocks[0] = JointCoRelationalBlock(hidden_dim=24, rel_depth=3, aggregation="mean", rel_update="linear")
        return m
    return StandaloneSpecialist(arch, input_dim=8, hidden_dim=24, depth=1)


def _load_cached(arch: str, seed: int, depths: dict[str, int], partials: Path) -> StandaloneSpecialist:
    if arch == "depth3":
        m = _empty_joint(arch)
    elif arch in ("joint", "joint_co"):
        m = StandaloneSpecialist(arch, input_dim=8, hidden_dim=24, depth=1)
    else:
        m = StandaloneSpecialist(arch, input_dim=8, hidden_dim=24, depth=depths[arch])
    pt = partials / f"seed{seed}_{arch}.pt"
    m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    m.eval()
    return m


def _dict_loader(ds: Any, batch_size: int) -> DataLoader[dict[str, torch.Tensor]]:
    class _DS(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
        def __len__(self) -> int:
            return len(ds)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            it = ds.items[idx]
            return {"features": it["features"], "target": it["target"]}

    return DataLoader(_DS(), batch_size=batch_size, shuffle=True)


def _build_router(empty: bool, experts: dict[str, StandaloneSpecialist], names: tuple[str, ...],
                  norm_costs: dict[str, float], lam: float,
                  train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> SampleLevelRouter:
    router = SampleLevelRouter(input_dim=8, hidden_dim=16, num_experts=len(names))
    if empty:
        return router
    trained, _ = _train_compute_aware_router(router, experts, names, norm_costs, lam,
                                             train_loader, val_loader, epochs=epochs)
    return trained


def _fam_acc(preds: torch.Tensor, targets: torch.Tensor, fams: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in FAMILIES:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if idx:
            out[f] = float((preds[idx] == targets[idx]).float().mean().item())
    out["mixed_mean"] = statistics.mean([out.get(f, 0.0) for f in MIXED_FAMS])
    out["overall"] = float((preds == targets).float().mean().item())
    return out


@torch.no_grad()
def _router_latency(router: SampleLevelRouter, experts: dict[str, StandaloneSpecialist],
                    feats: torch.Tensor) -> float:
    import time as _t
    sample = feats[:60]
    names = list(experts.keys())
    t0 = _t.perf_counter()
    for _ in range(5):
        ch = router(sample, mode="hard").selected_experts
        for i in range(len(sample)):
            _ = experts[names[int(ch[i].item())]](sample[i:i + 1])
    return (_t.perf_counter() - t0) / 5 / len(sample) * 1e6


def _component_semantics(expert: StandaloneSpecialist, eval_ds: Any, seed: int) -> dict[str, Any]:
    """20K: agreement audit + input-level R/C swaps with change-rate (repaired RC)."""
    from collections import defaultdict

    expert.eval()
    with torch.no_grad():
        preds = expert(eval_ds.features).argmax(-1)
    rc = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    agree = [i for i in rc if eval_ds.items[i]["sr"] == eval_ds.items[i]["sc"]]
    disagree = [i for i in rc if eval_ds.items[i]["sr"] != eval_ds.items[i]["sc"]]
    out: dict[str, Any] = {
        "agree_acc": float((preds[agree] == eval_ds.targets[agree]).float().mean().item()) if agree else 0.0,
        "disagree_acc": float((preds[disagree] == eval_ds.targets[disagree]).float().mean().item()) if disagree else 0.0,
    }
    # input-level swaps on ch5 (R) / ch0:4 (C); change-rate via production head
    pools: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in rc:
        pools[(eval_ds.items[i]["sr"], eval_ds.items[i]["sc"])].append(i)
    g = torch.Generator().manual_seed(seed + 4242)
    with torch.no_grad():
        base = expert(eval_ds.features).argmax(-1)

        def response(which: str) -> float:
            ch, tot = 0, 0
            for i in disagree:
                sr, sc = eval_ds.items[i]["sr"], eval_ds.items[i]["sc"]
                key = (-sr, sc) if which == "R" else (sr, -sc)
                donors = [j for j in pools.get(key, []) if j != i]
                if not donors:
                    continue
                j = donors[torch.randint(len(donors), (1,), generator=g).item()]
                mod = eval_ds.features[i].clone()
                if which == "R":
                    mod[:, RELATIONAL_CHANNEL] = eval_ds.features[j][:, RELATIONAL_CHANNEL]
                else:
                    mod[:, 0:5] = eval_ds.features[j][:, 0:5]
                new = int(expert(mod.unsqueeze(0)).argmax(-1).item())
                ch += 1 if new != int(base[i].item()) else 0
                tot += 1
            return ch / max(1, tot)

        out["R_swap_response"] = response("R")
        out["C_swap_response"] = response("C")
    return out


def _probe_bundle(expert: StandaloneSpecialist, eval_ds: Any) -> dict[str, float]:
    """20M: F/R/C + joint probes on depth3 states and final accuracy."""
    expert.eval()
    with torch.no_grad():
        enc = expert.encoder(eval_ds.features)
        state, _ = expert.blocks[0](enc, eval_ds.features)
    out: dict[str, float] = {}
    for letter in ("F", "R", "C"):
        key = {"F": "sf", "R": "sr", "C": "sc"}[letter]
        lab = torch.tensor([int(eval_ds.items[i][key] == 1) for i in range(len(eval_ds))])
        out[f"{letter}_probe"] = linear_probe_accuracy(state, lab, 24)
    # joint (sr,sc) probe on RC
    rc = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
    jlab = torch.tensor([int(eval_ds.items[i]["sr"] == 1) * 2 + int(eval_ds.items[i]["sc"] == 1) for i in rc])
    out["joint_RC_probe"] = linear_probe_accuracy(state[rc], jlab, 24)
    with torch.no_grad():
        preds = expert(eval_ds.features).argmax(-1)
    out["final_R"] = float((preds[[i for i, fm in enumerate(eval_ds.families_list) if fm == "R"]] == eval_ds.targets[[i for i, fm in enumerate(eval_ds.families_list) if fm == "R"]]).float().mean().item())
    out["final_RC"] = float((preds[rc] == eval_ds.targets[rc]).float().mean().item())
    return out


def _best_compositions(r: dict[str, Any]) -> list[str]:
    scored = []
    for k, v in r["oracle"].items():
        pf = v.get("per_family", {})
        mm = statistics.mean([pf.get(f, 0.0) for f in MIXED_FAMS if f in pf])
        scored.append((f"oracle_{k}", mm))
    for name, pf in r["chains"].items():
        mm = statistics.mean([pf.get(f, 0.0) for f in MIXED_FAMS])
        scored.append((f"chain_{name}", mm))
    scored.sort(key=lambda t: -t[1])
    return [n for n, _ in scored]


def _destruction_of_best(r: dict[str, Any], cand_names: list[str],
                         eval_ds: Any, seed: int) -> dict[str, dict[str, float]]:
    """Destruction eval for the best composition (router/chain/oracle-logits)."""
    name = cand_names[0] if cand_names else "none"
    dest_feats = apply_phase9_relational_permutation(eval_ds.features, seed=seed)
    fams, targets = eval_ds.families_list, eval_ds.targets
    out: dict[str, dict[str, float]] = {}
    for label, feats in (("original", eval_ds.features), ("relational_permuted", dest_feats)):
        if name.startswith("oracle_"):
            k = int(name.split("=")[1])
            preds = _oracle_predict(r, k, feats)
        elif name.startswith("chain_"):
            preds = _chain_predict(r, name[len("chain_"):], feats)
        else:
            preds = torch.zeros(len(feats), dtype=torch.long)
        per_fam = {}
        for f in ("R", "RC", "FRC"):
            idx = [i for i, fm in enumerate(fams) if fm == f]
            if idx:
                per_fam[f] = float((preds[idx] == targets[idx]).float().mean().item())
        out[label] = per_fam
    out["name"] = name  # type: ignore[assignment]
    return out


def _oracle_predict(r: dict[str, Any], k: int, feats: torch.Tensor) -> torch.Tensor:
    """Best-combo logit averaging recomputed on given features (frozen experts)."""
    import itertools

    experts = r["_experts"]
    names = list(experts.keys())
    fams = r["_splits"][2].families_list
    targets = r["_splits"][2].targets
    # NOTE: combo selection uses eval-split performance (diagnostic, documented)
    with torch.no_grad():
        logits = {n: experts[n](r["_splits"][2].features) for n in names}
    best: dict[str, tuple] = {}
    for f in FAMILIES:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        best_acc, best_c = -1.0, None
        for combo in itertools.combinations(names, k):
            stacked = torch.stack([logits[n] for n in combo], dim=0).mean(dim=0)
            acc = float((stacked.argmax(-1)[idx] == targets[idx]).float().mean().item()) if idx else 0.0
            if acc > best_acc:
                best_acc, best_c = acc, combo
        best[f] = best_c
    with torch.no_grad():
        cur = {n: experts[n](feats) for n in names}
    preds = torch.zeros(len(feats), dtype=torch.long)
    for f in FAMILIES:
        idx = [i for i, fm in enumerate(fams) if fm == f]
        if idx and best[f] is not None:
            stacked = torch.stack([cur[n] for n in best[f]], dim=0).mean(dim=0)
            preds[idx] = stacked.argmax(-1)[idx]
    return preds


def _chain_predict(r: dict[str, Any], label: str, feats: torch.Tensor) -> torch.Tensor:
    a, b = label.split("->")
    sub = {k: r["_experts"][k] for k in ("mlp", "graph", "attention_v2")}
    chain = StateChain(sub, (a, b), attach="identity", attach_v2_features=True)
    chain.eval()
    with torch.no_grad():
        return chain(feats).argmax(-1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_seed)
    cross_mean = {}
    for name in EXPERTS_20:
        cross_mean[name] = {f: mean([r["cross"][name].get(f, 0.0) for r in per_seed]) for f in list(FAMILIES) + ["mixed_mean", "overall"]}

    pure = {"tested": True,
            "best_F": max(cross_mean[e]["F"] for e in EXPERTS_20),
            "best_R": max(cross_mean[e]["R"] for e in EXPERTS_20),
            "best_C": max(cross_mean[e]["C"] for e in EXPERTS_20)}
    mixed = {"tested": True,
             "best_mixed_mean": mean([max(cross_mean[e][f] for e in EXPERTS_20) for f in MIXED_FAMS]),
             "best_RC": max(cross_mean[e]["RC"] for e in EXPERTS_20),
             "best_FRC": max(cross_mean[e]["FRC"] for e in EXPERTS_20)}

    # multi-component requirement (statistic rules on repaired eval, seed 0 rep figures per seed)
    rc_margins, frc_margins = [], []
    for r in per_seed:
        eval_ds = r["_splits"][2]
        feats = eval_ds.features
        s_sr = _stat_vec(feats, eval_ds, "sr")
        s_sc = _probe_sc_vec(r, eval_ds)
        s_sf = (feats[:, :, 0].mean(dim=1) > 0).long()
        rc_idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "RC"]
        rule = torch.where(s_sr[rc_idx] == s_sc[rc_idx], s_sr[rc_idx], s_sr[rc_idx])
        c_only = s_sc[rc_idx]
        yt = (eval_ds.targets[rc_idx] == 1).long()
        rc_margins.append(float((rule == yt).float().mean().item()) - float((c_only == yt).float().mean().item()))
        frc_idx = [i for i, fm in enumerate(eval_ds.families_list) if fm == "FRC"]
        votes = torch.stack([s_sf[frc_idx], s_sr[frc_idx], s_sc[frc_idx]], dim=1)
        rule_f = (votes.sum(dim=1) >= 2).long()
        best_single = max(float(((s_sf[frc_idx] == (eval_ds.targets[frc_idx] == 1).long()).float().mean().item())),
                          float(((s_sr[frc_idx] == (eval_ds.targets[frc_idx] == 1).long()).float().mean().item())),
                          float(((s_sc[frc_idx] == (eval_ds.targets[frc_idx] == 1).long()).float().mean().item())))
        frc_margins.append(float((rule_f == (eval_ds.targets[frc_idx] == 1).long()).float().mean().item()) - best_single)
    requires_multi = {"tested": True, "rc_margin": mean(rc_margins), "frc_margin": mean(frc_margins)}

    # composition gains: best fixed comp vs best single, RC/FRC
    rc_gains, frc_gains, both_pos = [], [], 0
    for r in per_seed:
        best_single_RC = max(r["cross"][e].get("RC", 0.0) for e in EXPERTS_20)
        best_single_FRC = max(r["cross"][e].get("FRC", 0.0) for e in EXPERTS_20)
        comp_rc = max([r["oracle"][f"k={k}"].get("per_family", {}).get("RC", 0.0) for k in (2, 3)]
                      + [v.get("RC", 0.0) for v in r["chains"].values()])
        comp_frc = max([r["oracle"][f"k={k}"].get("per_family", {}).get("FRC", 0.0) for k in (2, 3)]
                       + [v.get("FRC", 0.0) for v in r["chains"].values()])
        rc_gains.append(comp_rc - best_single_RC)
        frc_gains.append(comp_frc - best_single_FRC)
        both_pos += 1 if (comp_rc > best_single_RC and comp_frc > best_single_FRC) else 0
    composition = {"tested": True, "RC_gain": mean(rc_gains), "FRC_gain": mean(frc_gains), "n_pos": both_pos}

    # routing: learned λ=0 vs best single-expert mixed
    rout_gains, rpos = [], 0
    for r in per_seed:
        best_single_mixed = max(r["cross"][e].get("mixed_mean", 0.0) for e in EXPERTS_20)
        g = r["learned_routing"].get("mixed_mean", 0.0) - best_single_mixed
        rout_gains.append(g)
        rpos += 1 if g > 0 else 0
    routing = {"tested": True, "learned_minus_best_single_mixed": mean(rout_gains), "n_pos": rpos}

    # behavior: depth3 agree/disagree
    behavior = {"tested": True,
                "agree_acc": mean([r["semantics"]["agree_acc"] for r in per_seed]),
                "disagree_acc": mean([r["semantics"]["disagree_acc"] for r in per_seed])}

    # ceiling: single-expert vs portfolio-best (router/oracle/chain by mixed)
    pf_mixed, port_mixed, port_rc, port_frc, sing_rc, sing_frc = [], [], [], [], [], []
    for r in per_seed:
        fams = r["_splits"][2].families_list
        targets = r["_splits"][2].targets
        feats = r["_splits"][2].features
        lg = {n: r["_experts"][n](feats).detach() for n in EXPERTS_20}
        from neuroforge.training.phase12_expert_portfolio import _single_expert_ceiling as _ceil
        c = _ceil(lg, targets, fams)
        sing = mean([c["ceiling_per_family"].get(f, 0.0) for f in MIXED_FAMS])
        pf_mixed.append(sing)
        sing_rc.append(c["ceiling_per_family"].get("RC", 0.0))
        sing_frc.append(c["ceiling_per_family"].get("FRC", 0.0))
        cands = {"router": r["learned_routing"].get("mixed_mean", 0.0)}
        for k in (2, 3):
            cands[f"oracle_k={k}"] = mean([r["oracle"][f"k={k}"].get("per_family", {}).get(f, 0.0) for f in MIXED_FAMS])
        best_name = max(cands, key=lambda k: cands[k])
        port_mixed.append(cands[best_name])
        if best_name == "router":
            port_rc.append(r["learned_routing"].get("RC", 0.0))
            port_frc.append(r["learned_routing"].get("FRC", 0.0))
        else:
            kk = best_name.split("=")[1]
            port_rc.append(r["oracle"][f"k={kk}"].get("per_family", {}).get("RC", 0.0))
            port_frc.append(r["oracle"][f"k={kk}"].get("per_family", {}).get("FRC", 0.0))
        r["portfolio_best_name"] = best_name
    ceiling = {"tested": True, "portfolio_minus_single_mixed": mean(port_mixed) - mean(pf_mixed),
               "portfolio_RC": mean(port_rc), "portfolio_FRC": mean(port_frc),
               "single_RC": mean(sing_rc), "single_FRC": mean(sing_frc),
               "single_mixed": mean(pf_mixed), "portfolio_mixed": mean(port_mixed)}

    # smoke
    smoke = {"tested": True, "passed": all(r["smoke"]["passed"] for r in per_seed),
             "detail": "all seeds pass repaired-validity smoke" if all(r["smoke"]["passed"] for r in per_seed) else "SMOKE FAILED"}

    # seed stability (RC per key condition)
    stability: dict[str, dict[str, float]] = {"RC": {}}
    for key, lab in (("cross_depth3", "depth3"), ("oracle", "oracle_k=2"), ("router", "learned")):
        if key == "cross_depth3":
            vals = [r["cross"]["depth3"].get("RC", 0.0) for r in per_seed]
        elif key == "oracle":
            vals = [r["oracle"]["k=2"].get("per_family", {}).get("RC", 0.0) for r in per_seed]
        else:
            vals = [r["learned_routing"].get("RC", 0.0) for r in per_seed]
        if len(vals) > 1:
            stability["RC"][lab] = statistics.stdev(vals)

    # latency (joint_co as reference; depth3 must not regress-latency vs it)
    latency = {"tested": True,
               "baseline_total_us": mean([r["latencies"].get("joint_co", 0.0) for r in per_seed]),
               "candidate_total_us": mean([r["latencies"].get("depth3", 0.0) for r in per_seed])}
    return {"n_seeds": n, "pure": pure, "mixed": mixed, "requires_multi": requires_multi,
            "composition": composition, "routing": routing, "behavior": behavior,
            "ceiling": ceiling, "smoke": smoke, "seed_stability": stability,
            "latency": latency, "cross_mean": cross_mean}


def _stat_vec(feats: torch.Tensor, eval_ds: Any, comp: str) -> torch.Tensor:
    if comp == "sr":
        x = feats[:, :, RELATIONAL_CHANNEL].float()
        return ((x * torch.roll(x, shifts=-1, dims=1)).mean(dim=1) > 0).long()
    raise ValueError(comp)


def _probe_sc_vec(r: dict[str, Any], eval_ds: Any) -> torch.Tensor:
    n = len(eval_ds)
    ch3 = eval_ds.features[:, :, 3]
    lab = torch.tensor([int(eval_ds.items[i]["sc"] == 1) for i in range(n)])
    cut = n // 2
    Xtr, Ytr = ch3[:cut].float(), lab[:cut]
    Y1h = torch.zeros(len(Xtr), 2)
    Y1h[torch.arange(len(Xtr)), Ytr.long()] = 1.0
    try:
        W = torch.linalg.solve(Xtr.t() @ Xtr + 1e-2 * torch.eye(12), Xtr.t() @ Y1h)
    except Exception:
        W = torch.linalg.lstsq(Xtr, Y1h).solution
    return (ch3.float() @ W).argmax(dim=1)


# ---------------------------------------------------------------------------
# CSV writers + report
# ---------------------------------------------------------------------------
def _serialisable(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in per_seed:
        row: dict[str, Any] = {"seed": r["seed"]}
        for k, v in r.items():
            if k.startswith("_") or k == "seed":
                continue
            row[k] = v
        out.append(row)
    return out


def _write_csvs(m_dir: Path, per_seed: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> None:
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
    for r in per_seed:
        for k in ("r_align_R", "r_align_RC", "r_align_FRC"):
            rows.append({"seed": r["seed"], "check": k, "value": r["smoke"].get(k, 0.0)})
        rows.append({"seed": r["seed"], "check": "smoke_passed", "value": float(r["smoke"]["passed"])})
    write_csv("benchmark_smoke.csv", rows)

    rows = []
    for r in per_seed:
        fp = r["fingerprints"]
        rows.append({"seed": r["seed"], "train_fp": fp["train"], "val_fp": fp["val"], "eval_fp": fp["eval"],
                     "train_test_overlap": fp["train_test_overlap"], "val_test_overlap": fp["val_test_overlap"]})
    write_csv("dataset_manifest.csv", rows)

    rows = []
    for r in per_seed:
        for name in EXPERTS_20:
            row = {"seed": r["seed"], "expert": name}
            for f in fams:
                row[f] = r["cross"][name].get(f, 0.0)
            row["mixed_mean"] = r["cross"][name].get("mixed_mean", 0.0)
            rows.append(row)
    write_csv("pure_experts.csv", rows)
    write_csv("cross_evaluation.csv", rows)

    rows = []
    for r in per_seed:
        fams_l = r["_splits"][2].families_list
        targets = r["_splits"][2].targets
        feats = r["_splits"][2].features
        lg = {n: r["_experts"][n](feats).detach() for n in EXPERTS_20}
        from neuroforge.training.phase12_expert_portfolio import _single_expert_ceiling as _ceil
        c = _ceil(lg, targets, fams_l)
        row = {"seed": r["seed"]}
        for f in fams:
            row[f] = c["ceiling_per_family"].get(f, 0.0)
        rows.append(row)
    write_csv("empirical_ceiling.csv", rows)

    rows = []
    for r in per_seed:
        for k in (1, 2, 3):
            pf = r["oracle"][f"k={k}"].get("per_family", {})
            row = {"seed": r["seed"], "policy": f"oracle_k={k}"}
            for f in fams:
                row[f] = pf.get(f, 0.0)
            rows.append(row)
        for name, pf in r["chains"].items():
            row = {"seed": r["seed"], "policy": f"chain_{name}"}
            for f in fams:
                row[f] = pf.get(f, 0.0)
            rows.append(row)
    write_csv("fixed_composition.csv", rows)

    rows = []
    for r in per_seed:
        for k in (1, 2, 3):
            pf = r["oracle"][f"k={k}"].get("per_family", {})
            row = {"seed": r["seed"], "policy": f"oracle_k={k}"}
            for f in ("R", "RC", "FRC"):
                row[f] = pf.get(f, 0.0)
            rows.append(row)
    write_csv("oracle.csv", rows)

    rows = []
    for r in per_seed:
        row = {"seed": r["seed"]}
        for f in ("R", "RC", "FRC"):
            row[f] = r["random_routing"].get(f, 0.0)
        row["mixed_mean"] = r["random_routing"].get("mixed_mean", 0.0)
        rows.append(row)
    write_csv("random_routing.csv", rows)

    rows = []
    for r in per_seed:
        row = {"seed": r["seed"]}
        for f in fams:
            row[f] = r["learned_routing"].get(f, 0.0)
        row["mixed_mean"] = r["learned_routing"].get("mixed_mean", 0.0)
        rows.append(row)
    write_csv("learned_routing.csv", rows)

    rows = []
    for r in per_seed:
        for lam, acc in r["sweep"].items():
            row = {"seed": r["seed"], "lambda": lam}
            for f in ("R", "RC", "FRC"):
                row[f] = acc.get(f, 0.0)
            row["mixed_mean"] = acc.get("mixed_mean", 0.0)
            row["latency_us"] = acc.get("latency_us", 0.0)
            rows.append(row)
    write_csv("compute_aware.csv", rows)

    rows = []
    for r in per_seed:
        d = r["routing_diag"]
        rows.append({"seed": r["seed"], "mean_entropy": d["mean_entropy"],
                     "utilization": json.dumps(d["utilization"]),
                     "family_expert": json.dumps(d["family_expert"])})
    write_csv("routing_diagnostics.csv", rows)

    rows = []
    for r in per_seed:
        s = r["semantics"]
        rows.append({"seed": r["seed"], "agree_acc": s["agree_acc"], "disagree_acc": s["disagree_acc"],
                     "R_swap_response": s["R_swap_response"], "C_swap_response": s["C_swap_response"]})
    write_csv("component_semantics.csv", rows)

    rows = []
    for r in per_seed:
        for cond in ("depth3",):
            rows.append({"seed": r["seed"], "condition": cond,
                         "R": r["cross"][cond].get("R", 0.0),
                         "RC": r["cross"][cond].get("RC", 0.0),
                         "FRC": r["cross"][cond].get("FRC", 0.0)})
    write_csv("rc_frc_analysis.csv", rows)

    rows = []
    for r in per_seed:
        for k, v in r["probes"].items():
            rows.append({"seed": r["seed"], "probe": k, "value": v})
    write_csv("representation_probe.csv", rows)

    rows = []
    for r in per_seed:
        for cond, ev in r["destruction"].items():
            if cond in ("bestcomp_name",) or not isinstance(ev, dict):
                continue
            for variant, per_fam in ev.items():
                if not isinstance(per_fam, dict):
                    continue
                row = {"seed": r["seed"], "condition": cond, "variant": variant}
                for f in ("R", "RC", "FRC"):
                    row[f] = per_fam.get(f, 0.0)
                rows.append(row)
    write_csv("causal_controls.csv", rows)

    rows = []
    for r in per_seed:
        for name in EXPERTS_20:
            rows.append({"seed": r["seed"], "expert": name,
                         "params": sum(p.numel() for p in r["_experts"][name].parameters()),
                         "flops": r["_flops"][name]})
    write_csv("compute.csv", rows)

    rows = []
    for r in per_seed:
        for name in EXPERTS_20:
            lat = r["latencies"].get(name, 0.0)
            rows.append({"seed": r["seed"], "expert": name, "latency_us": lat,
                         "throughput_per_s": 1e6 / max(lat, 1e-9)})
    write_csv("latency.csv", rows)

    rows = []
    for r in per_seed:
        rows.append({"seed": r["seed"],
                     "best_RC": max(r["cross"][e].get("RC", 0.0) for e in EXPERTS_20),
                     "best_FRC": max(r["cross"][e].get("FRC", 0.0) for e in EXPERTS_20),
                     "router_mixed": r["learned_routing"].get("mixed_mean", 0.0)})
    write_csv("seed_results.csv", rows)

    write_csv("failure_diagnosis.csv", [
        {"seed": 0, "category": c["category"], "failed": c["failed"],
         "criterion": c["criterion"], "observed": c["observed"]} for c in failure_rows])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_phase20_markdown_report(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def pct(x: float) -> str:
        return f"{float(x) * 100:.1f}%"

    cross = summary.get("cross_mean", {})
    hyps = summary.get("hypotheses", {})
    fail = summary.get("failure_diagnosis", [])
    agg = summary.get("aggregates", {})
    ceil = agg.get("ceiling", {})

    report = f"""# Phase 20 — Clean Portfolio Re-evaluation on Repaired Mixed Benchmark

## Correction first

> The mixed benchmark previously erased the relational carrier from RC/FRC inputs.
> Phase 19 repaired and validated the construction (`{CONSTRUCTION_VERSION}`).
> Phase 20 therefore constitutes the first clean re-evaluation of heterogeneous
> composition under the intended observable task semantics.

### Historical result (invalid construction)

Old mixed-composition experiments showed an RC/FRC composition failure — now
understood as substantially benchmark-induced (model at feasible optimum on
unlearnable labels).

### Repaired result

Below: fresh training on repaired data only; pre-repair checkpoints never used.

### Interpretation

Benchmark-induced changes must NOT be attributed to architecture. Accuracy deltas
vs history reflect task repair unless architecture, protocol, and benchmark are
otherwise controlled (they are not: the benchmark changed).

---

## 1. Cross-evaluation matrix (repaired, mean over seeds)

| Expert | F | R | C | FR | RC | FC | FRC | Mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for name in EXPERTS_20:
        p = cross.get(name, {})
        cells = " | ".join(pct(p.get(f, 0.0)) for f in FAMILIES)
        report += f"| **{name}** | {cells} | {pct(p.get('mixed_mean', 0.0))} |\n"
    report += f"""
## 2. New empirical ceiling (repaired benchmark, evaluated portfolio)

Single-expert mixed: {ceil.get('single_mixed', float('nan')) * 100:.1f}% → portfolio: {ceil.get('portfolio_mixed', float('nan')) * 100:.1f}% ({ceil.get('portfolio_minus_single_mixed', float('nan')) * 100:+.1f}pp).
RC: {ceil.get('single_RC', float('nan')) * 100:.1f}% → {ceil.get('portfolio_RC', float('nan')) * 100:.1f}%. FRC: {ceil.get('single_FRC', float('nan')) * 100:.1f}% → {ceil.get('portfolio_FRC', float('nan')) * 100:.1f}%.
(Historical 66.4% cited for reference only; never used as the new ceiling.)

## 3. Composition, routing, compute-aware sweep

Full per-λ and per-policy tables in `compute_aware.csv`, `fixed_composition.csv`,
`learned_routing.csv`, `random_routing.csv`, `routing_diagnostics.csv`.

## 4. RC/FRC semantics on repaired inputs

Component agreement, input-level R/C swap change-rates, probes, and destruction
in `component_semantics.csv`, `representation_probe.csv`, `causal_controls.csv`.

## 5. Hypotheses H1–H8 (programmatic)

| Hypothesis | Status | Evidence |
|---|---|---|
"""
    for h in [f"H{i}" for i in range(1, 9)]:
        d = hyps.get(h, {})
        report += f"| **{h}** | {d.get('status', '?')} | {d.get('evidence', '')} |\n"
    report += f"""
## 6. Final CASE (programmatic)

**{summary.get('verdict_case', '?')} — {summary.get('verdict_label', '')}**

Recommendation: {summary.get('recommendation', '')}

Minimal intervention: **{summary.get('minimal_intervention', {}).get('intervention', '?')}** — {summary.get('minimal_intervention', {}).get('outcome', '')}.

## 7. Failure diagnosis

| Category | Failed | Criterion | Observed |
|---|---|---|---|
"""
    for row in fail:
        report += f"| {row.get('category')} | {row.get('failed')} | {row.get('criterion')} | {row.get('observed')} |\n"
    report += """
## 8. Limitations

1. Three seeds; mean ± SD, never best-seed.
2. Specialists train on repaired pure-family subsets (120 train); joints on repaired F+R+C.
3. Router trains on all-family repaired mixed; num_experts=7 follows the Phase 12 4→5 precedent (portfolio sizing, not redesign).
4. Task changed vs history: deltas vs Phase 9–18 are repair effects unless otherwise controlled.
"""
    output_path.write_text(report, encoding="utf-8")
