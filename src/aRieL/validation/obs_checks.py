"""Validate observation configs, live env spaces, and optional custom builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from aRieL.utils.config import (
    ALL_EVENT_FEATURES,
    ALL_GLOBAL_FEATURES,
    ObservationConfig,
)
from aRieL.validation.report import CheckResult, ValidationReport

OBS_REQUIREMENTS = """
Observation contract
--------------------
ArielEnv observations are dicts of float32 arrays:

  top-K / target mode
      obs["events"]  shape (n_actions, n_event_features)
      obs["global"]  shape (n_global_features,)

  full_set mode
      obs["planets"] shape (n_actions, n_planet_features)
      obs["global"]  shape (n_global_features,)

Requirements
  * keys match the action type
  * dtypes are float32 (or castable without object dtype)
  * all values finite (no NaN / ±Inf)
  * shapes match env.observation_space
  * info["action_mask"] is a 1-D bool array of length n_actions

Event feature names must be computed in observation_builder (top-K) or you
must supply a compatible custom builder.  Known stock names are listed in
ALL_EVENT_FEATURES / ALL_GLOBAL_FEATURES.  Full-set planet features live in
planet_feature_builder.PLANET_FEATURE_NAMES — see docs/CUSTOMISING.md.
"""


def validate_observation_config(cfg: ObservationConfig) -> ValidationReport:
    """Check feature-name lists against the stock vocabulary."""
    report = ValidationReport(title="Observation config")

    if not cfg.event_features:
        report.add(
            CheckResult(
                "event_features_nonempty",
                "warn",
                "event_features is empty (ok only if you never use top-K mode)",
            )
        )
    else:
        unknown = [f for f in cfg.event_features if f not in ALL_EVENT_FEATURES]
        if unknown:
            report.add(
                CheckResult(
                    "event_features_known",
                    "fail",
                    f"unknown event features: {unknown}",
                    details=f"stock names: {ALL_EVENT_FEATURES}",
                    hint=(
                        "Either pick names from ALL_EVENT_FEATURES, or implement "
                        "them in observation_builder._build_event_row and add them "
                        "to ALL_EVENT_FEATURES."
                    ),
                )
            )
        else:
            report.add(
                CheckResult(
                    "event_features_known",
                    "ok",
                    f"{len(cfg.event_features)} event features are known",
                )
            )
        dupes = _duplicates(cfg.event_features)
        if dupes:
            report.add(
                CheckResult(
                    "event_features_unique",
                    "fail",
                    f"duplicate event features: {dupes}",
                )
            )
        else:
            report.add(CheckResult("event_features_unique", "ok", "no duplicate event features"))

    if not cfg.global_features:
        report.add(
            CheckResult(
                "global_features_nonempty",
                "fail",
                "global_features is empty",
                hint="Include at least fraction_elapsed / tier fractions.",
            )
        )
    else:
        unknown_g = [f for f in cfg.global_features if f not in ALL_GLOBAL_FEATURES]
        if unknown_g:
            report.add(
                CheckResult(
                    "global_features_known",
                    "fail",
                    f"unknown global features: {unknown_g}",
                    details=f"stock names: {ALL_GLOBAL_FEATURES}",
                    hint="Implement new globals in observation_builder._build_global.",
                )
            )
        else:
            report.add(
                CheckResult(
                    "global_features_known",
                    "ok",
                    f"{len(cfg.global_features)} global features are known",
                )
            )

    if cfg.min_bin_targets < 0:
        report.add(
            CheckResult(
                "min_bin_targets",
                "fail",
                f"min_bin_targets={cfg.min_bin_targets} must be >= 0",
            )
        )
    else:
        report.add(
            CheckResult(
                "min_bin_targets",
                "ok",
                f"min_bin_targets={cfg.min_bin_targets}",
            )
        )

    return report


def validate_observation_dict(
    obs: Mapping[str, Any],
    *,
    action_type: str = "topk",
    observation_space: Any | None = None,
    action_mask: np.ndarray | None = None,
    n_actions: int | None = None,
) -> ValidationReport:
    """Validate a single observation dict (+ optional mask / Gymnasium space)."""
    report = ValidationReport(title="Observation dict")

    if action_type == "full_set":
        required_keys = ("planets", "global")
    else:
        required_keys = ("events", "global")

    missing = [k for k in required_keys if k not in obs]
    if missing:
        report.add(
            CheckResult(
                "keys",
                "fail",
                f"missing keys {missing} for action_type={action_type!r}",
                hint=OBS_REQUIREMENTS.strip(),
            )
        )
        return report
    report.add(CheckResult("keys", "ok", f"has required keys {required_keys}"))

    for key in required_keys:
        report.extend(_check_array(key, obs[key], observation_space=observation_space))

    if action_mask is not None:
        report.extend(_check_action_mask(action_mask, obs, action_type, n_actions))

    if observation_space is not None:
        try:
            # Gymnasium spaces have .contains
            contains = observation_space.contains(obs)
            if contains:
                report.add(
                    CheckResult(
                        "space_contains",
                        "ok",
                        "obs is contained in observation_space",
                    )
                )
            else:
                report.add(
                    CheckResult(
                        "space_contains",
                        "fail",
                        "obs is NOT contained in observation_space",
                        hint="Check dtypes (float32), shapes, and value ranges.",
                    )
                )
        except Exception as exc:
            report.add(
                CheckResult(
                    "space_contains",
                    "warn",
                    f"observation_space.contains raised: {exc}",
                )
            )

    return report


def validate_env_observation(
    env: Any,
    *,
    seed: int = 0,
    n_steps: int = 3,
) -> ValidationReport:
    """Reset/step a live env and validate obs + masks each time."""
    report = ValidationReport(title="Live env observation")
    try:
        obs, info = env.reset(seed=seed)
    except Exception as exc:
        report.add(CheckResult("reset", "fail", f"env.reset raised: {exc}"))
        return report
    report.add(CheckResult("reset", "ok", "env.reset succeeded"))

    action_type = getattr(getattr(env, "cfg", None), "action", None)
    action_type = getattr(action_type, "type", "topk")

    sub = validate_observation_dict(
        obs,
        action_type=action_type,
        observation_space=getattr(env, "observation_space", None),
        action_mask=info.get("action_mask"),
        n_actions=getattr(env, "n_actions", None),
    )
    report.extend(sub.checks)

    for step in range(n_steps):
        mask = info.get("action_mask")
        if mask is None or not np.any(mask):
            report.add(
                CheckResult(
                    f"step_{step}",
                    "warn",
                    "no valid actions; stopping early",
                )
            )
            break
        action = int(np.flatnonzero(mask)[0])
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:
            report.add(CheckResult(f"step_{step}", "fail", f"env.step raised: {exc}"))
            break
        if not np.isfinite(float(reward)):
            report.add(
                CheckResult(
                    f"step_{step}_reward",
                    "fail",
                    f"non-finite reward {reward!r}",
                )
            )
        sub = validate_observation_dict(
            obs,
            action_type=action_type,
            observation_space=getattr(env, "observation_space", None),
            action_mask=info.get("action_mask"),
            n_actions=getattr(env, "n_actions", None),
        )
        # Collapse per-step noise: only keep failures/warns + one ok marker
        fails = [c for c in sub.checks if c.status != "ok"]
        if fails:
            report.extend(fails)
            report.add(
                CheckResult(f"step_{step}", "fail", "observation checks failed after step")
            )
            break
        report.add(CheckResult(f"step_{step}", "ok", "obs + mask ok"))
        if terminated or truncated:
            report.add(CheckResult(f"step_{step}_done", "ok", "episode ended"))
            break

    return report


def validate_obs_builder(
    builder: Callable[..., Mapping[str, Any]],
    *,
    sample_kwargs: Mapping[str, Any] | None = None,
    action_type: str = "topk",
    mean_budget_us: float = 5_000.0,
    n_timing: int = 200,
    check_speed: bool = True,
) -> ValidationReport:
    """Validate a custom observation builder callable.

    ``builder(**sample_kwargs)`` must return an obs dict.  Provide whatever
    positional/keyword args your builder needs via ``sample_kwargs``.

    For the stock builder, prefer ``validate_env_observation`` instead.
    """
    import time

    report = ValidationReport(title="Custom observation builder")
    sample_kwargs = dict(sample_kwargs or {})

    try:
        obs = builder(**sample_kwargs)
    except TypeError:
        # allow positional-only builders that ignore kwargs
        try:
            obs = builder()
        except Exception as exc:
            report.add(
                CheckResult(
                    "call",
                    "fail",
                    f"builder raised: {exc}",
                    hint="Pass sample_kwargs= with the arguments your builder needs.",
                )
            )
            return report
    except Exception as exc:
        report.add(CheckResult("call", "fail", f"builder raised: {exc}"))
        return report

    if not isinstance(obs, Mapping):
        report.add(
            CheckResult(
                "return_type",
                "fail",
                f"builder returned {type(obs).__name__}, expected dict",
            )
        )
        return report
    report.add(CheckResult("call", "ok", "builder returned a mapping"))
    report.extend(
        validate_observation_dict(obs, action_type=action_type).checks
    )

    if check_speed:
        # Warm-up
        for _ in range(5):
            builder(**sample_kwargs)
        samples = []
        for _ in range(n_timing):
            t0 = time.perf_counter()
            builder(**sample_kwargs)
            samples.append((time.perf_counter() - t0) * 1e6)
        mean = float(np.mean(samples))
        p95 = float(np.percentile(samples, 95))
        detail = f"mean={mean:.1f} µs  p95={p95:.1f} µs  budget_mean={mean_budget_us:.0f} µs"
        if mean > mean_budget_us:
            report.add(
                CheckResult(
                    "speed",
                    "fail",
                    f"obs builder too slow (mean={mean:.1f} µs)",
                    details=detail,
                    hint=(
                        "Obs builders run every step. Prefer numpy over pandas row "
                        "loops; cache static features."
                    ),
                )
            )
        elif mean > 0.5 * mean_budget_us:
            report.add(
                CheckResult(
                    "speed",
                    "warn",
                    f"obs builder acceptable but heavy (mean={mean:.1f} µs)",
                    details=detail,
                )
            )
        else:
            report.add(
                CheckResult(
                    "speed",
                    "ok",
                    f"obs builder within budget (mean={mean:.1f} µs)",
                    details=detail,
                )
            )

    return report


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _duplicates(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for x in items:
        if x in seen and x not in dupes:
            dupes.append(x)
        seen.add(x)
    return dupes


def _check_array(
    key: str,
    value: Any,
    *,
    observation_space: Any | None,
) -> list[CheckResult]:
    out: list[CheckResult] = []
    try:
        arr = np.asarray(value)
    except Exception as exc:
        return [
            CheckResult(f"{key}_array", "fail", f"cannot convert to array: {exc}")
        ]

    if arr.dtype == object:
        out.append(
            CheckResult(
                f"{key}_dtype",
                "fail",
                f"{key} has object dtype",
                hint="Use numeric float32 arrays only.",
            )
        )
        return out

    if arr.dtype != np.float32:
        out.append(
            CheckResult(
                f"{key}_dtype",
                "warn",
                f"{key} dtype is {arr.dtype}, expected float32",
                hint="Cast with np.asarray(..., dtype=np.float32).",
            )
        )
    else:
        out.append(CheckResult(f"{key}_dtype", "ok", f"{key} dtype=float32"))

    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        out.append(
            CheckResult(
                f"{key}_finite",
                "fail",
                f"{key} contains {n_bad} non-finite values",
                hint="Replace NaN/Inf before returning the observation.",
            )
        )
    else:
        out.append(CheckResult(f"{key}_finite", "ok", f"{key} all finite"))

    if observation_space is not None and key in getattr(observation_space, "spaces", {}):
        expected = observation_space[key].shape
        if arr.shape != expected:
            out.append(
                CheckResult(
                    f"{key}_shape",
                    "fail",
                    f"{key} shape {arr.shape} != space shape {expected}",
                )
            )
        else:
            out.append(
                CheckResult(f"{key}_shape", "ok", f"{key} shape {arr.shape}")
            )
    else:
        out.append(
            CheckResult(f"{key}_shape", "ok", f"{key} shape {arr.shape}")
        )

    return out


def _check_action_mask(
    mask: Any,
    obs: Mapping[str, Any],
    action_type: str,
    n_actions: int | None,
) -> list[CheckResult]:
    out: list[CheckResult] = []
    arr = np.asarray(mask)
    if arr.ndim != 1:
        out.append(
            CheckResult(
                "action_mask_rank",
                "fail",
                f"action_mask rank {arr.ndim} != 1",
            )
        )
        return out
    if arr.dtype != bool and arr.dtype != np.bool_:
        out.append(
            CheckResult(
                "action_mask_dtype",
                "warn",
                f"action_mask dtype {arr.dtype}, expected bool",
            )
        )
    else:
        out.append(CheckResult("action_mask_dtype", "ok", "action_mask dtype=bool"))

    if action_type == "full_set":
        candidate_n = int(np.asarray(obs["planets"]).shape[0])
    else:
        candidate_n = int(np.asarray(obs["events"]).shape[0])

    expected_n = n_actions if n_actions is not None else candidate_n
    if arr.shape[0] != expected_n:
        out.append(
            CheckResult(
                "action_mask_len",
                "fail",
                f"action_mask length {arr.shape[0]} != expected {expected_n}",
            )
        )
    else:
        out.append(
            CheckResult(
                "action_mask_len",
                "ok",
                f"action_mask length {arr.shape[0]}",
            )
        )
    return out
