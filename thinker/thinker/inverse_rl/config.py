"""Configuration helpers for the Offline ML-IRL pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


@dataclass
class BehaviorDataConfig:
    base_path: str = "../behavioral_data_block"
    subjects: List[int] = field(default_factory=lambda: [1])
    game_id: int = 1
    frame_stack_n: int = 4
    sequence_length: int = 40
    batch_size: int = 32
    target_size: Tuple[int, int] = (84, 84)
    grayscale: bool = True


@dataclass
class OfflineOptimizationConfig:
    gamma: float = 0.99
    entropy_coef: float = 0.01
    penalty_scale: float = 1.0
    penalty_clamp: float = 5.0
    actor_lr: float = 3e-5
    reward_lr: float = 1e-4
    actor_grad_clip: float = 5.0
    reward_grad_clip: float = 5.0
    feature_norm_momentum: float = 0.01
    replay_size: int = 512
    warmup_batches: int = 32
    replay_batches_per_step: int = 4
    actor_ckp_interval: int = 0
    max_steps: int = 1000
    log_interval: int = 50
    policy_updates_per_step: int = 1
    reward_updates_per_step: int = 1


@dataclass
class ThinkerCheckpointConfig:
    preload: str = ""
    reward_feature_source: str = "sr"
    device: str = "auto"
    config_path: Optional[str] = None


@dataclass
class OfflineMLIRLConfig:
    behavior: BehaviorDataConfig = field(default_factory=BehaviorDataConfig)
    optim: OfflineOptimizationConfig = field(default_factory=OfflineOptimizationConfig)
    thinker: ThinkerCheckpointConfig = field(default_factory=ThinkerCheckpointConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior": dataclass_to_dict(self.behavior),
            "optim": dataclass_to_dict(self.optim),
            "thinker": dataclass_to_dict(self.thinker),
        }


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in fields(obj)}


def _coerce_subjects(value: Any) -> List[int]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [int(p) for p in parts]
    if isinstance(value, Sequence):
        return [int(v) for v in value]
    return [int(value)]


def _coerce_tuple(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected pair for target_size, got {value!r}")


def _update_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    valid_fields = {f.name for f in fields(instance)}
    for key, value in updates.items():
        if key not in valid_fields:
            continue
        if isinstance(getattr(instance, key), list) and not isinstance(value, list):
            setattr(instance, key, list(value))
        else:
            setattr(instance, key, value)
    return instance


def load_offline_mlirl_config(config_path: str | Path) -> OfflineMLIRLConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Offline ML-IRL config '{path}' does not exist.")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    cfg = OfflineMLIRLConfig()
    behavior_cfg = data.get("behavior", {})
    if "subjects" in behavior_cfg:
        behavior_cfg["subjects"] = _coerce_subjects(behavior_cfg["subjects"])
    if "target_size" in behavior_cfg:
        behavior_cfg["target_size"] = _coerce_tuple(behavior_cfg["target_size"])
    _update_dataclass(cfg.behavior, behavior_cfg)

    optim_cfg = data.get("optim", {})
    _update_dataclass(cfg.optim, optim_cfg)

    thinker_cfg = data.get("thinker", {})
    _update_dataclass(cfg.thinker, thinker_cfg)

    return cfg
