import torch
from neuroforge.evaluation.phase7_metrics import calculate_oracle_regret, calculate_routing_statistics


def test_routing_statistics_utilization_and_matrix():
    chosen = ["mlp", "graph", "attention_v2", "mlp", "graph", "attention_v2"]
    families = ["feature", "relational", "contextual", "feature", "relational", "contextual"]

    stats = calculate_routing_statistics(chosen, families)

    # Utilization
    assert stats["utilization"]["mlp"] == 1.0 / 3.0
    assert stats["utilization"]["graph"] == 1.0 / 3.0
    assert stats["utilization"]["attention_v2"] == 1.0 / 3.0
    assert stats["utilization"]["attention"] == 0.0

    # Entropy for 3 equal choices is log2(3) ~ 1.58496
    assert abs(stats["routing_entropy_bits"] - 1.58496) < 0.01

    # Selection matrix
    mat = stats["selection_matrix_proportions"]
    assert mat["feature"]["mlp"] == 1.0
    assert mat["relational"]["graph"] == 1.0
    assert mat["contextual"]["attention_v2"] == 1.0


def test_oracle_regret_computation():
    targets = torch.tensor([1, 0, 1, 0, 1, 0])
    oracle_preds = torch.tensor([1, 0, 1, 0, 1, 0])  # 100%
    fixed_preds = torch.tensor([1, 0, 0, 0, 0, 0])   # misses index 2 and 4 (66.7%)
    families = ["feature", "feature", "relational", "relational", "contextual", "contextual"]

    regret = calculate_oracle_regret(oracle_preds, fixed_preds, targets, families)

    assert regret["oracle_accuracy"] == 1.0
    assert abs(regret["best_fixed_accuracy"] - (4.0 / 6.0)) < 0.01
    assert abs(regret["net_accuracy_advantage"] - (2.0 / 6.0)) < 0.01
    assert regret["oracle_win_count"] == 2
    assert regret["fixed_win_count"] == 0
    assert regret["disagreement_count"] == 2
