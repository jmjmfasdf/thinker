"""Versioned training changes independent of the unchanged soft-gate schema.

Legacy flags keep the historical behavior. New critic/memory/gradient settings
are checkpointed explicitly and may not change during a resume.
"""
from collections.abc import Mapping
import math


def planning_training_contract(flags):
    values = flags if isinstance(flags, Mapping) else vars(flags)
    critic = values.get("voc_critic_parameterization", "policy_centered_v1")
    if critic not in ("policy_centered_v1", "common_contrast_v1"):
        raise ValueError("unknown voc_critic_parameterization")
    burn = values.get("icopro_burn_in_steps", 1)
    context = values.get("icopro_sampling_context", burn)
    for name, value in (("icopro_burn_in_steps", burn),
                        ("icopro_sampling_context", context)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if context < burn:
        raise ValueError("icopro_sampling_context must cover burn-in")
    scale = values.get("voc_gru_gradient_scale", 1.0)
    if isinstance(scale, bool) or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("voc_gru_gradient_scale must be finite and positive")
    if scale != 1 and not values.get("voc_gru_connected", False):
        raise ValueError("VOC GRU scaling requires voc_gru_connected")
    if critic == "common_contrast_v1":
        if not values.get("voc_dueling_q", True):
            raise ValueError("common_contrast_v1 requires dueling Q")
        if values.get("voc_actor_policy_version_barrier", False):
            raise ValueError("new critic contract cannot alter frozen atomic schemas")
    norm = values.get("actor_h_encoder_norm", "none")
    if norm not in ("none", "layer"):
        raise ValueError("actor_h_encoder_norm must be 'none' or 'layer'")
    x_norm = values.get("actor_x_encoder_norm", "none")
    if x_norm not in ("none", "layer"):
        raise ValueError("actor_x_encoder_norm must be 'none' or 'layer'")
    contract = dict(version=1, critic=critic, burn_in_steps=burn,
                sampling_context=context, gru_gradient_scale=float(scale),
                gru_connected=bool(values.get("voc_gru_connected", False)),
                train_epsilon=float(values.get("voc_train_epsilon", .02)))
    # Preserve version1 contracts byte-for-byte for existing raw encoders.
    if norm != "none":
        contract.update(version=2, actor_h_encoder_norm=norm)
    # Keep historical v1/v2 contracts unchanged; x normalization changes the
    # actor architecture and must never be enabled silently on resume.
    if x_norm != "none":
        contract.update(version=3, actor_h_encoder_norm=norm,
                        actor_x_encoder_norm=x_norm)
    return contract


def validate_planning_training_checkpoint(checkpoint, flags=None):
    saved = planning_training_contract(checkpoint.get("flags", {}))
    recorded = checkpoint.get("planning_training_contract")
    extended = (saved["critic"] != "policy_centered_v1"
                or saved["burn_in_steps"] != 1 or saved["sampling_context"] != 1
                or saved["gru_gradient_scale"] != 1 or saved["version"] != 1)
    if (recorded is not None or extended) and recorded != saved:
        raise ValueError("checkpoint planning training contract is missing or inconsistent")
    if flags is not None and saved != planning_training_contract(flags):
        raise ValueError("cannot resume with a different planning training contract")
    return saved
