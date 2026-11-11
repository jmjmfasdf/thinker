"""Incumbent coordinates using the existing persistence coordinate formulas."""
import numpy as np

COORDINATE_VARIANT = "native_source_observation_actor_policy_v3"


def window_coordinates(start_policy, terminal_policy, search_policy, incumbent, coordinates):
    """Policies refer to one native search window; no historical chunk shift."""
    n_actions = len(terminal_policy)
    result = {}
    mapping = {"margin": "persistence_evidence", "uncertainty": "switch_target_uncertainty",
               "probability": "current_action_probability", "mass_logodds": "persistence_mass_logodds"}
    for label, policy in (("start", start_policy), ("terminal", terminal_policy)):
        values = coordinates(policy, incumbent, n_actions)
        for output, source in mapping.items():
            result[f"actor_current_action_{output}_{label}"] = values[source]
    # Search coordinate is the coordinate of the mean search policy, as in the
    # historical ensemble summary; an empty imagination has no search mean.
    search = np.mean(search_policy, axis=0) if len(search_policy) else np.full(n_actions, np.nan)
    values = coordinates(search, incumbent, n_actions)
    for output, source in mapping.items():
        result[f"actor_current_action_{output}_search"] = values[source]
    result["actor_policy_start_candidate"] = int(np.argmax(start_policy))
    result["actor_current_action_coordinate_valid"] = int(np.isfinite(
        result["actor_current_action_margin_start"]) and np.isfinite(result["actor_current_action_margin_terminal"]))
    return result
