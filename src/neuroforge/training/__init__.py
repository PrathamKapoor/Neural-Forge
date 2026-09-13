from .experiment import run_experiment
from .routing_diagnosis import run_routing_diagnosis
from .expert_specialization import run_expert_specialization
from .phase6_specialization import run_phase6_specialization
from .phase7_oracle import run_phase7_oracle_routing
from .phase8a_learned_routing import run_phase8a_learned_routing, verify_router_marker_neutrality
from .phase8b_compute_aware import run_phase8b_compute_aware_routing
from .phase9_generalization import generate_markdown_report, run_phase9_generalization
from .phase10_multi_expert import (
    generate_phase10_markdown_report,
    run_phase10_multi_expert,
)

__all__ = [
    "run_experiment",
    "run_routing_diagnosis",
    "run_expert_specialization",
    "run_phase6_specialization",
    "run_phase7_oracle_routing",
    "run_phase8a_learned_routing",
    "run_phase8b_compute_aware_routing",
    "run_phase9_generalization",
    "generate_markdown_report",
    "run_phase10_multi_expert",
    "generate_phase10_markdown_report",
    "verify_router_marker_neutrality",
]




