"""IcoPro-style behavioral cloning trainer for Thinker networks."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from thinker.bc_loader import FrameStackedBehavioralDataLoader
from thinker.actor_net import ActorNet
from thinker.model_net import ModelNet
from imitation import ThinkerPolicyAdapter, compute_icopro_actor_losses


class IcoProBehaviorCloningTrainer:
    """IcoPro-inspired behavioral cloning trainer for Thinker."""

    def __init__(
        self,
        actor_net: ActorNet,
        model_net: ModelNet,
        data_loader: FrameStackedBehavioralDataLoader,
        flags,
        logger=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.actor_net = actor_net
        self.model_net = model_net
        self.data_loader = data_loader
        self.flags = flags
        self.logger = logger
        self.device = device or next(actor_net.parameters()).device
        self.mode = int(getattr(flags, "mode", 3))

        self.policy_adapter = ThinkerPolicyAdapter(self.actor_net, self.model_net, flags, self.device)

        self.batch_size = max(1, getattr(flags, "bc_batch_size", 32))
        self.eval_batch_size = max(1, getattr(flags, "bc_eval_batch_size", self.batch_size))
        self.margin_value = float(getattr(flags, "bc_margin", 0.05))
        self.margin_coef = float(getattr(flags, "bc_margin_coef", 1.0))
        self.ce_coef = float(getattr(flags, "bc_ce_coef", 1.0))
        self.tree_coef = float(getattr(flags, "bc_tree_coef", 0.0))
        self.max_grad_norm = getattr(flags, "bc_max_grad_norm", None)

        warm_batch = self._sample_batch(batch_size=self.batch_size)
        if warm_batch is None:
            raise RuntimeError("Unable to load behavioral cloning data for warm-up.")
        obs = warm_batch["obs"][:1]
        warm_actions = warm_batch["actions"][:1]
        warm_policy = self.policy_adapter.forward(obs, actions=warm_actions, requires_grad=False)
        self.num_actions = warm_policy.logits.shape[-1]

        lr_actor = getattr(flags, "bc_lr", 1e-4)
        lr_model = getattr(flags, "bc_model_lr", lr_actor)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.model_optimizer = (
            torch.optim.Adam(self.model_net.parameters(), lr=lr_model)
            if self.mode in (2, 3)
            else None
        )

        self.actor_net.train()
        self.model_net.train()
        self._best_metric = float("inf")

        base_savedir = Path(getattr(self.flags, "savedir", "./logs"))
        proj_name = getattr(self.flags, "proj_name", "bc_icopro")
        self.output_dir = base_savedir / proj_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        existing_ckpdir = Path(getattr(self.flags, "ckpdir", str(self.output_dir)))
        if existing_ckpdir.exists() and existing_ckpdir.resolve() != self.output_dir.resolve():
            for item in existing_ckpdir.iterdir():
                dest = self.output_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    if not dest.exists():
                        shutil.copy2(item, dest)
        config_source = existing_ckpdir / "config_c.yaml"
        if config_source.exists():
            shutil.copy2(config_source, self.output_dir / "config_c.yaml")

        os.environ["THINKER_LOG_DIR"] = str(self.output_dir)
        self.flags.ckpdir = str(self.output_dir)

        def _filter_value(value):
            if isinstance(value, (int, float, str, bool)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                if all(isinstance(item, (int, float, str, bool)) or item is None for item in value):
                    return list(value)
            return None

        self.serialized_flags = {}
        for key, value in vars(self.flags).items():
            filtered = _filter_value(value)
            if filtered is not None:
                self.serialized_flags[key] = filtered

        self.updates_per_epoch = max(1, getattr(self.flags, "bc_updates_per_epoch", len(self.data_loader.data_files)))

    def _sample_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        batch = self.data_loader.get_paired_batch(batch_size=batch_size)
        if batch is None:
            return None
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)
        actions_np = np.asarray(batch["actions"])
        if actions_np.ndim > 1 and actions_np.shape[-1] > 1:
            actions_np = actions_np.argmax(axis=-1)
        actions = torch.from_numpy(actions_np.astype(np.int64)).to(self.device)
        rewards = torch.from_numpy(np.asarray(batch["rewards"], dtype=np.float32)).to(self.device)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
        }

    def _actor_step(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            metrics = compute_icopro_actor_losses(
                self.policy_adapter,
                obs,
                actions,
                margin_value=self.margin_value,
                margin_coef=self.margin_coef,
                ce_coef=self.ce_coef,
                tree_coef=self.tree_coef,
                requires_grad=train,
            )
            total_loss = metrics["total_loss"]
            margin_loss = metrics["margin_loss"]
            ce_loss = metrics["ce_loss"]
            accuracy = metrics["accuracy"]
        if train:
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
        return {
            "loss": float(total_loss.detach().cpu().item()),
            "margin": float(margin_loss.detach().cpu().item()),
            "ce": float(ce_loss.detach().cpu().item()),
            "acc": float(accuracy),
        }

    def _model_step(self, batch: Dict[str, torch.Tensor], train: bool) -> float:
        if self.model_optimizer is None or not train:
            return 0.0
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]

        if getattr(self.model_net, "state_dtype_n", 0) == 0:
            obs_input = (obs * 255.0).clamp(0, 255).to(torch.uint8)
            next_input = (next_obs * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            obs_input = obs
            next_input = next_obs

        batch_size = obs.shape[0]
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        prev_actions = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        current_actions = actions.view(batch_size, 1)
        action_seq = torch.stack([prev_actions, current_actions], dim=0)
        state = self.model_net.initial_state(batch_size=batch_size, device=self.device)

        model_out = self.model_net.forward(
            env_state=obs_input,
            done=done,
            actions=action_seq,
            state=state,
            future_env_state=next_input,
            training=True,
        )

        losses: List[torch.Tensor] = []
        if getattr(model_out, "rs", None) is not None:
            predicted_rewards = model_out.rs[0]
            if predicted_rewards.ndim == 2:
                predicted_rewards = predicted_rewards[:, 0]
            losses.append(F.mse_loss(predicted_rewards.float(), rewards.float()))
        if getattr(model_out, "policy", None) is not None:
            policy_logits = model_out.policy[0].float()
            policy_logits = policy_logits.view(policy_logits.shape[0], -1)
            losses.append(F.cross_entropy(policy_logits, actions.long()))
        if getattr(model_out, "xs", None) is not None:
            predicted_next = model_out.xs[0]
            target_next = self.model_net.normalize(next_input)
            predicted_next = predicted_next.float()
            losses.append(F.mse_loss(predicted_next, target_next))

        if not losses:
            return 0.0

        model_loss = sum(losses)
        self.model_optimizer.zero_grad()
        model_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model_net.parameters(), self.max_grad_norm)
        self.model_optimizer.step()
        return float(model_loss.detach().cpu().item())

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        actor_losses, margin_losses, ce_losses, accuracies, model_losses = [], [], [], [], []
        updates = 0
        for _ in range(self.updates_per_epoch):
            batch = self._sample_batch(self.batch_size)
            if batch is None:
                continue
            updates += 1
            actor_metrics = self._actor_step(batch, train=self.mode in (1, 3))
            actor_losses.append(actor_metrics["loss"])
            margin_losses.append(actor_metrics["margin"])
            ce_losses.append(actor_metrics["ce"])
            accuracies.append(actor_metrics["acc"])
            model_loss = self._model_step(batch, train=self.mode in (2, 3))
            model_losses.append(model_loss)
        metrics = {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "margin_loss": float(np.mean(margin_losses)) if margin_losses else 0.0,
            "ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "model_loss": float(np.mean(model_losses)) if model_losses else 0.0,
            "updates": updates,
        }
        return metrics

    def train(self) -> Tuple[float, List[Dict[str, float]]]:
        history: List[Dict[str, float]] = []
        epochs = getattr(self.flags, "bc_epochs", 1)
        for epoch in range(epochs):
            metrics = self._run_epoch(epoch)
            history.append(metrics)

            reference_metric = metrics.get("actor_loss", metrics.get("model_loss", 0.0))
            if reference_metric < self._best_metric:
                self._best_metric = reference_metric
                self._save_checkpoint(epoch, metrics, best=True)

            save_interval = max(1, getattr(self.flags, "bc_save_interval", 1))
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(epoch, metrics, best=False)

            if self.logger:
                self.logger.info(
                    "[BC] epoch=%d actor=%.4f margin=%.4f ce=%.4f acc=%.4f model=%.4f updates=%d",
                    epoch + 1,
                    metrics.get("actor_loss", 0.0),
                    metrics.get("margin_loss", 0.0),
                    metrics.get("ce_loss", 0.0),
                    metrics.get("accuracy", 0.0),
                    metrics.get("model_loss", 0.0),
                    metrics.get("updates", 0),
                )

            eval_interval = getattr(self.flags, "bc_eval_interval", 0)
            if eval_interval and (epoch + 1) % eval_interval == 0:
                eval_stats = self.evaluate(
                    num_batches=getattr(self.flags, "bc_eval_batches", 1),
                    batch_size=self.eval_batch_size,
                )
                metrics.update({f"eval_{k}": v for k, v in eval_stats.items()})
                if self.logger:
                    self.logger.info(
                        "[BC] eval " + ", ".join(f"{k}={v:.4f}" for k, v in eval_stats.items())
                    )

        self.policy_adapter.close()
        return self._best_metric, history

    def evaluate(self, num_batches: int, batch_size: int) -> Dict[str, float]:
        if num_batches <= 0:
            return {"actor_loss": 0.0, "accuracy": 0.0}
        actor_losses, accuracies = [], []
        for _ in range(num_batches):
            batch = self._sample_batch(batch_size)
            if batch is None:
                continue
            metrics = self._actor_step(batch, train=False)
            actor_losses.append(metrics["loss"])
            accuracies.append(metrics["acc"])
        if not actor_losses:
            return {"actor_loss": 0.0, "accuracy": 0.0}
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "accuracy": float(np.mean(accuracies)),
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], best: bool) -> None:
        def _atomic_save(payload, file_path: Path) -> None:
            tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
            torch.save(payload, tmp_path)
            tmp_path.replace(file_path)

        base_dir = self.output_dir
        actor_payload = {
            "state_dict": self.actor_net.state_dict(),
            "flags": self.serialized_flags,
            "epoch": epoch,
            "metrics": metrics,
        }
        model_payload = {
            "state_dict": self.model_net.state_dict(),
            "flags": self.serialized_flags,
            "epoch": epoch,
            "metrics": metrics,
        }

        if best:
            _atomic_save(actor_payload, base_dir / "ckp_actor_best.tar")
            _atomic_save(model_payload, base_dir / "ckp_model_best.tar")
        else:
            _atomic_save(actor_payload, base_dir / f"ckp_actor_epoch{epoch + 1:04d}.tar")
            _atomic_save(model_payload, base_dir / f"ckp_model_epoch{epoch + 1:04d}.tar")

        if self.logger:
            status = "best" if best else "epoch"
            self.logger.info(f"Saved {status} checkpoint to {base_dir}")


def create_bc_data_loader(flags) -> FrameStackedBehavioralDataLoader:
    subjects = [int(s.strip()) for s in str(flags.bc_subjects).split(',')]
    loader = FrameStackedBehavioralDataLoader(
        base_path=flags.bc_data_path,
        subjects=subjects,
        game_id=flags.bc_game_id,
        frame_stack_n=flags.frame_stack_n,
        target_size=(84, 84),
        grayscale=flags.grayscale,
        normalize=True,
    )
    return loader


def run_pe_rlhf_training(
    flags,
    model_net: ModelNet,
    actor_net: ActorNet,
    logger=None,
) -> Tuple[float, List[Dict[str, float]]]:
    loader = create_bc_data_loader(flags)
    trainer = IcoProBehaviorCloningTrainer(
        actor_net=actor_net,
        model_net=model_net,
        data_loader=loader,
        flags=flags,
        logger=logger,
    )
    return trainer.train()
