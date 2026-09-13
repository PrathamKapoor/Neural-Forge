"""Typed experiment configuration with early, actionable validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_samples: int = 768
    validation_samples: int = 192
    test_samples: int = 192
    sequence_length: int = 6
    input_dim: int = 8

    def __post_init__(self) -> None:
        if min(self.train_samples, self.validation_samples, self.test_samples, self.sequence_length, self.input_dim) < 1:
            raise ValueError("data sizes and dimensions must be positive")


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 24
    num_classes: int = 2
    modules: tuple[str, ...] = ("mlp", "graph", "attention")

    def __post_init__(self) -> None:
        supported = {"mlp", "graph", "attention"}
        if self.hidden_dim < 2 or self.num_classes < 2:
            raise ValueError("hidden_dim must be >= 2 and num_classes must be >= 2")
        if not self.modules or any(module not in supported for module in self.modules):
            raise ValueError(f"modules must be a non-empty subset of {sorted(supported)}")


@dataclass(frozen=True)
class RouterConfig:
    strategy: str = "soft"
    temperature: float = 1.0
    compute_penalty: float = 0.01

    def __post_init__(self) -> None:
        if self.strategy not in {"soft", "hard"}:
            raise ValueError("strategy must be 'soft' or 'hard'")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.compute_penalty < 0:
            raise ValueError("compute_penalty must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.003

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    latency_warmup: int = 5
    latency_iterations: int = 20

    def __post_init__(self) -> None:
        if self.latency_warmup < 0 or self.latency_iterations < 1:
            raise ValueError("latency warmup must be >= 0 and iterations must be positive")


@dataclass(frozen=True)
class NeuroForgeConfig:
    seed: int = 42
    device: str = "cpu"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NeuroForgeConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle) or {}
        return cls(
            seed=raw.get("seed", 42), device=raw.get("device", "cpu"),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**{**raw.get("model", {}), "modules": tuple(raw.get("model", {}).get("modules", ("mlp", "graph", "attention")))}),
            router=RouterConfig(**raw.get("router", {})), training=TrainingConfig(**raw.get("training", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
