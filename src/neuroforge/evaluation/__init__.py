from .metrics import benchmark_latency, evaluate_classifier
from .phase8a_metrics import (
    calculate_counterfactual_analysis,
    calculate_oracle_recovery,
    calculate_route_accuracy,
    check_routing_collapse,
)

from .phase8b_metrics import (
    calculate_counterfactual_sacrifices,
    calculate_family_conditional_cost,
    calculate_pareto_frontier,
    check_cost_collapse,
    compute_normalized_costs,
    identify_representative_operating_points,
)

from .phase9_metrics import (
    calculate_decision_preservation,
    calculate_generalization_gap,
    calculate_mixed_counterfactual_analysis,
    calculate_single_expert_ceiling,
    diagnose_mixed_structure_failure,
)
from .composition_metrics import (
    calculate_accuracy_constrained_compute,
    calculate_active_expert_statistics,
    calculate_composition_entropy,
    calculate_counterfactual_composition_analysis,
    calculate_pair_utilization,
    calculate_triple_utilization,
    check_composition_collapses,
    diagnose_phase10_failure,
    generate_family_composition_matrix,
)

__all__ = [
    "benchmark_latency",
    "evaluate_classifier",
    "calculate_route_accuracy",
    "calculate_oracle_recovery",
    "calculate_counterfactual_analysis",
    "check_routing_collapse",
    "compute_normalized_costs",
    "calculate_pareto_frontier",
    "check_cost_collapse",
    "calculate_family_conditional_cost",
    "calculate_counterfactual_sacrifices",
    "identify_representative_operating_points",
    "calculate_generalization_gap",
    "calculate_decision_preservation",
    "calculate_single_expert_ceiling",
    "diagnose_mixed_structure_failure",
    "calculate_mixed_counterfactual_analysis",
    "calculate_active_expert_statistics",
    "calculate_pair_utilization",
    "calculate_triple_utilization",
    "calculate_composition_entropy",
    "generate_family_composition_matrix",
    "check_composition_collapses",
    "calculate_accuracy_constrained_compute",
    "calculate_counterfactual_composition_analysis",
    "diagnose_phase10_failure",
]


