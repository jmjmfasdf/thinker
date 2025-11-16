"""Offline ML-IRL trainer that reuses Thinker's actor/model components."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, Dict, List, Optional
from pathlib import Path

import numpy as np

import torch

from imitation import ThinkerPolicyAdapter

from .config import OfflineMLIRLConfig
from .datasets import DemonstrationBatch, ThinkerBehaviorDataset
from .features import FeatureScaler, extract_features
from .policy_updater import OfflinePolicyUpdater
from .reward_estimator import RewardEstimator
from .world_model_adapter import WorldModelAdapter


@dataclass
class OfflineMLIRLState:
    step: int = 0
    actor_updates: int = 0
    reward_updates: int = 0


class DemonstrationReplay:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[dict] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.buffer)

    def add_batch(self, batch: DemonstrationBatch) -> None:
        images = batch.images.astype(np.float16)
        rewards = batch.rewards.astype(np.float32)
        is_first = batch.is_first
        actions = batch.actions
        B, T = rewards.shape
        for b in range(B):
            prev_action = 0
            for t in range(T):
                action = int(actions[b, t])
                entry = {
                    "obs": images[b, t],
                    "action": action,
                    "prev_action": prev_action,
                    "reward": rewards[b, t],
                    "sequence_start": bool(is_first[b, t]),
                }
                self.buffer.append(entry)
                prev_action = action

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        samples = random.sample(self.buffer, batch_size)
        obs = np.stack([s["obs"] for s in samples], axis=0).astype(np.float32)
        actions = np.array([s["action"] for s in samples], dtype=np.int64)
        prev_actions = np.array([s["prev_action"] for s in samples], dtype=np.int64)
        rewards = np.array([s["reward"] for s in samples], dtype=np.float32)
        seq_flags = np.array([s["sequence_start"] for s in samples], dtype=bool)
        return {
            "obs": obs,
            "actions": actions,
            "prev_actions": prev_actions,
            "rewards": rewards,
            "sequence_starts": seq_flags,
        }


class OfflineMLIRLTrainer:
    def __init__(
        self,
        config: OfflineMLIRLConfig,
        dataset: ThinkerBehaviorDataset,
        policy_adapter: ThinkerPolicyAdapter,
        reward_estimator: RewardEstimator,
        reward_optimizer: torch.optim.Optimizer,
        policy_updater: OfflinePolicyUpdater,
        world_model: WorldModelAdapter,
        device: torch.device,
        save_dir: Path,
    ) -> None:
        self.cfg = config
        self.dataset = dataset
        self.policy_adapter = policy_adapter
        self.reward_estimator = reward_estimator
        self.reward_optimizer = reward_optimizer
        self.policy_updater = policy_updater
        self.world_model = world_model
        self.device = device
        self.state = OfflineMLIRLState()
        self.feature_source = config.thinker.reward_feature_source.lower()
        self.feature_scaler = FeatureScaler(momentum=config.optim.feature_norm_momentum)
        self.replay = DemonstrationReplay(capacity=config.optim.replay_size)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> List[Dict[str, float]]:
        logs: List[Dict[str, float]] = []
        max_steps = int(self.cfg.optim.max_steps)
        log_interval = max(1, int(self.cfg.optim.log_interval))
        warmup = max(1, int(self.cfg.optim.warmup_batches))
        completed_steps = 0
        for step in range(max_steps):
            new_batch = self.dataset.sample_batch(self.cfg.behavior.batch_size)
            if new_batch is None:
                raise RuntimeError("Behavior dataset is exhausted or empty.")
            self.replay.add_batch(new_batch)
            if len(self.replay) < warmup:
                if (step + 1) % log_interval == 0:
                    print(f"[offline-mlirl] Warming up replay ({len(self.replay)}/{warmup})")
                continue
            metrics = self._train_step()
            logs.append(metrics)
            completed_steps += 1
            if completed_steps % log_interval == 0:
                self._log_progress(completed_steps, metrics)
            interval = int(self.cfg.optim.actor_ckp_interval)
            if interval > 0 and completed_steps % interval == 0:
                ckp_path = self.save_dir / f"ckp_actor.tar_{completed_steps}"
                self.save_actor_checkpoint(ckp_path)
        return logs

    def _train_step(self) -> Dict[str, float]:
        batches_per_step = max(1, int(self.cfg.optim.replay_batches_per_step))
        metrics_accum: Dict[str, float] = {}
        for _ in range(batches_per_step):
            batch = self.replay.sample_batch(self.cfg.behavior.batch_size)
            metrics = self._update_with_batch(batch)
            for key, value in metrics.items():
                metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)
        for key in metrics_accum:
            metrics_accum[key] /= batches_per_step
        return metrics_accum

    def _update_with_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        torch_batch = self._to_torch(batch)
        policy_batch = self.policy_adapter.forward(
            torch_batch["obs"],
            prev_actions=torch_batch["prev_actions"],
            sequence_starts=torch_batch["sequence_starts"],
            requires_grad=True,
        )

        features = extract_features(self.policy_adapter, policy_batch, self.feature_source)
        features = self.feature_scaler(features)
        rewards = self.reward_estimator(features)
        penalty = self.world_model.compute_penalty(
            self.policy_adapter.last_tree_q,
            torch_batch["actions"],
            torch_batch["returns"],
        )
        shaped = rewards - penalty
        advantages = shaped - shaped.mean()

        actor_metrics = self.policy_updater.update(
            policy_batch,
            torch_batch["actions"],
            advantages,
        )
        self.state.actor_updates += 1

        log_probs = policy_batch.log_probs.gather(-1, torch_batch["actions"].view(-1, 1)).squeeze(-1)
        centered_rewards = rewards - rewards.mean()
        reward_loss = -(log_probs.detach() * centered_rewards).mean()
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_estimator.parameters(), self.cfg.optim.reward_grad_clip)
        self.reward_optimizer.step()
        self.state.reward_updates += 1

        self.state.step += 1
        metrics_step = self.state.step
        mean_log_prob = float(log_probs.mean().detach().cpu().item())
        mean_shaped = float(shaped.mean().detach().cpu().item())
        mean_penalty = float(penalty.mean().detach().cpu().item())
        mean_raw_reward = float(rewards.mean().detach().cpu().item())

        metrics = {
            **actor_metrics,
            "step": metrics_step,
            "reward_loss": float(reward_loss.detach().cpu().item()),
            "penalty": mean_penalty,
            "mean_reward": mean_raw_reward,
            "mean_shaped_reward": mean_shaped,
            "mean_log_prob": mean_log_prob,
            "actor_updates": self.state.actor_updates,
            "reward_updates": self.state.reward_updates,
        }
        return metrics

    def _to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        device = self.device
        obs = torch.from_numpy(batch["obs"]).to(device=device, dtype=torch.float32)
        actions = torch.from_numpy(batch["actions"]).to(device)
        prev_actions = torch.from_numpy(batch["prev_actions"]).to(device)
        rewards = torch.from_numpy(batch["rewards"]).to(device)
        seq_flags = torch.from_numpy(batch["sequence_starts"]).to(device)
        return {
            "obs": obs,
            "actions": actions,
            "prev_actions": prev_actions,
            "returns": rewards,
            "sequence_starts": seq_flags,
        }

    def build_actor_checkpoint(self) -> Dict[str, object]:
        optimizer_state = self.policy_updater.state_dict()["optimizer"]
        return {
            "step": self.state.step,
            "real_step": self.state.step,
            "tot_eps": self.state.actor_updates,
            "ret_buffers": None,
            "norm_stats": None,
            "crnorm": None,
            "actor_net_optimizer_state_dict": optimizer_state,
            "actor_net_scheduler_state_dict": None,
            "actor_net_state_dict": self.policy_updater.actor_net.state_dict(),
            "flags": vars(self.policy_adapter.flags) if hasattr(self.policy_adapter, "flags") else {},
        }

    def save_actor_checkpoint(self, path: Path) -> None:
        torch.save(self.build_actor_checkpoint(), path)

    @staticmethod
    def _log_progress(step: int, metrics: Dict[str, float]) -> None:
        summary = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[offline-mlirl] step={step} :: {summary}")
