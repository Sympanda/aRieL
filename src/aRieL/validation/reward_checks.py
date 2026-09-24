"""Validate reward configs, callables, and latency budgets."""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from aRieL.rewards.compute_reward import compute_reward
from aRieL.rewards.context import reward_info_from_step
from aRieL.utils.config import RewardConfig, load_reward_config, resolve_reward_preset
from aRieL.validation.report import CheckResult, ValidationReport
from aRieL.validation.step_result import (
    REQUIRED_STEP_RESULT_KEYS,
    STEP_RESULT_REQUIREMENTS,
    default_bin_context,
    make_step_result,
    step_result_battery,
)

# Defaults chosen relative to the stock reward (~2 µs/call on a laptop).
# Custom rewards should stay well under a millisecond — they run every env step.
DEFAULT_REWARD_MEAN_US = 100.0
DEFAULT_REWARD_P95_US = 250.0
DEFAULT_REWARD_WARN_MEAN_US = 50.0


RewardFn = Callable[..., Any]


def validate_reward_config(
    cfg: RewardConfig | str | Path | Mapping[str, Any],
    *,
    strict: bool = True,
) -> ValidationReport:
    """Check a ``RewardConfig`` / YAML / mapping for loadability and sanity."""
    report = ValidationReport(title="Reward config")

    try:
        if isinstance(cfg, RewardConfig):
            reward_cfg = cfg
            report.add(CheckResult("load", "ok", "RewardConfig instance accepted"))
        elif isinstance(cfg, Mapping):
            # Build via YAML-equivalent path: write-through load_reward_config needs a file;
            # construct with dataclass fields instead.
            reward_cfg = _mapping_to_reward_config(dict(cfg), strict=strict)
            report.add(CheckResult("load", "ok", "mapping converted to RewardConfig"))
        else:
            path = resolve_reward_preset(cfg) if not Path(cfg).is_file() else Path(cfg)
            reward_cfg = load_reward_config(path, strict=strict)
            report.add(CheckResult("load", "ok", f"loaded {path}"))
    except Exception as exc:
        report.add(
            CheckResult(
                "load",
                "fail",
                f"could not load reward config: {exc}",
                hint=(
                    "Use a complete YAML under aRieL/configs/reward/ as a template.\n"
                    "Strict mode requires every RewardConfig field to be present.\n"
                    f"Known fields: {', '.join(f.name for f in fields(RewardConfig))}"
                ),
            )
        )
        return report

    # Numeric sanity
    bad_nonfinite = []
    for f in fields(RewardConfig):
        val = getattr(reward_cfg, f.name)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if not math.isfinite(float(val)):
                bad_nonfinite.append(f.name)
    if bad_nonfinite:
        report.add(
            CheckResult(
                "finite_weights",
                "fail",
                f"non-finite numeric fields: {bad_nonfinite}",
            )
        )
    else:
        report.add(CheckResult("finite_weights", "ok", "all numeric fields are finite"))

    if reward_cfg.overhead_penalty_mode not in ("absolute", "fraction"):
        report.add(
            CheckResult(
                "overhead_mode",
                "fail",
                f"overhead_penalty_mode={reward_cfg.overhead_penalty_mode!r}",
                hint="Must be 'absolute' or 'fraction'.",
            )
        )
    else:
        report.add(
            CheckResult(
                "overhead_mode",
                "ok",
                f"overhead_penalty_mode={reward_cfg.overhead_penalty_mode!r}",
            )
        )

    if reward_cfg.reward_mode not in ("absolute", "relative"):
        report.add(
            CheckResult(
                "reward_mode",
                "fail",
                f"reward_mode={reward_cfg.reward_mode!r}",
                hint="Must be 'absolute' or 'relative'.",
            )
        )
    else:
        report.add(CheckResult("reward_mode", "ok", f"reward_mode={reward_cfg.reward_mode!r}"))

    # Smoke: stock compute_reward must accept this config
    try:
        totals, before, after, hosts = default_bin_context()
        r = compute_reward(
            make_step_result(), reward_cfg, totals, before, after, hosts
        )
        if not _is_finite_number(r):
            report.add(
                CheckResult(
                    "stock_compute_smoke",
                    "fail",
                    f"stock compute_reward returned non-finite {r!r}",
                )
            )
        else:
            report.add(
                CheckResult(
                    "stock_compute_smoke",
                    "ok",
                    f"stock compute_reward(smoke) → {float(r):.4g}",
                )
            )
    except Exception as exc:
        report.add(
            CheckResult(
                "stock_compute_smoke",
                "fail",
                f"stock compute_reward raised: {exc}",
            )
        )

    return report


def validate_reward_fn(
    fn: RewardFn,
    cfg: RewardConfig | None = None,
    *,
    mean_budget_us: float = DEFAULT_REWARD_MEAN_US,
    p95_budget_us: float = DEFAULT_REWARD_P95_US,
    warn_mean_us: float = DEFAULT_REWARD_WARN_MEAN_US,
    n_timing: int = 2000,
    check_speed: bool = True,
) -> ValidationReport:
    """Validate a custom reward callable against the env contract + speed gate.

    Parameters
    ----------
    fn:
        Callable with the same signature as ``compute_reward`` (see
        ``STEP_RESULT_REQUIREMENTS``).
    cfg:
        Config passed through; defaults to ``RewardConfig()``.
    mean_budget_us / p95_budget_us:
        Hard fail thresholds for mean / 95th-percentile latency (microseconds).
    warn_mean_us:
        Soft warning if mean exceeds this but stays under the fail budget.
    n_timing:
        Number of timed calls for the speed check.
    check_speed:
        Set False to skip the latency gate.
    """
    report = ValidationReport(title="Custom reward function")
    cfg = cfg or RewardConfig()

    report.add(_check_signature(fn))
    report.extend(_check_correctness(fn, cfg))
    if check_speed:
        report.add(
            _check_speed(
                fn,
                cfg,
                mean_budget_us=mean_budget_us,
                p95_budget_us=p95_budget_us,
                warn_mean_us=warn_mean_us,
                n_timing=n_timing,
            )
        )
    return report


def validate_reward(
    cfg: RewardConfig | str | Path | Mapping[str, Any] | None = None,
    fn: RewardFn | None = None,
    **speed_kwargs: Any,
) -> ValidationReport:
    """Run config and/or callable checks; combine into one report."""
    if cfg is None and fn is None:
        report = ValidationReport(title="Reward validation")
        report.add(
            CheckResult(
                "input",
                "fail",
                "pass cfg= and/or fn=",
                hint="Example: validate_reward(cfg='default', fn=my_reward)",
            )
        )
        return report

    combined = ValidationReport(title="Reward validation")
    reward_cfg: RewardConfig | None = None

    if cfg is not None:
        cfg_report = validate_reward_config(cfg)
        combined.extend(cfg_report.checks)
        if isinstance(cfg, RewardConfig):
            reward_cfg = cfg
        elif cfg_report.ok:
            if isinstance(cfg, Mapping):
                reward_cfg = _mapping_to_reward_config(dict(cfg), strict=False)
            else:
                path = resolve_reward_preset(cfg) if not Path(cfg).is_file() else Path(cfg)
                reward_cfg = load_reward_config(path, strict=False)

    if fn is not None:
        fn_report = validate_reward_fn(fn, reward_cfg or RewardConfig(), **speed_kwargs)
        combined.extend(fn_report.checks)

    return combined


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _mapping_to_reward_config(data: dict, *, strict: bool) -> RewardConfig:
    """Build RewardConfig from a plain dict (with optional 'reward' nesting)."""
    if "reward" in data and isinstance(data["reward"], dict):
        data = data["reward"]
    required = {f.name for f in fields(RewardConfig)}
    provided = set(data)
    if strict:
        missing = sorted(required - provided)
        unknown = sorted(provided - required)
        if missing or unknown:
            parts = []
            if missing:
                parts.append("missing: " + ", ".join(missing))
            if unknown:
                parts.append("unknown: " + ", ".join(unknown))
            raise ValueError("; ".join(parts))
    kwargs: dict[str, Any] = {}
    defaults = RewardConfig()
    tuple_fields = {
        "t1_milestone_fractions",
        "t2_milestone_fractions",
        "science_weight_tiers",
    }
    for f in fields(RewardConfig):
        if f.name in data:
            val = data[f.name]
            if f.name in tuple_fields and isinstance(val, list):
                val = tuple(val)
            kwargs[f.name] = val
        else:
            kwargs[f.name] = getattr(defaults, f.name)
    return RewardConfig(**kwargs)


def _check_signature(fn: RewardFn) -> CheckResult:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        return CheckResult(
            "signature",
            "warn",
            f"could not introspect signature ({exc}); will call positionally",
            hint=(
                "Preferred simple form: def my_reward(info: RewardInfo) -> float\n"
                + STEP_RESULT_REQUIREMENTS.strip()
            ),
        )

    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    ]
    n_required = sum(
        1 for p in params if p.default is p.empty and p.kind != p.VAR_POSITIONAL
    )
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)

    if n_required <= 1 or has_varargs:
        return CheckResult(
            "signature",
            "ok",
            f"simple RewardInfo-style signature: {fn.__name__}{sig}",
            hint="Expected: def my_reward(info: RewardInfo) -> float",
        )
    if n_required <= 5:
        return CheckResult(
            "signature",
            "ok",
            f"legacy compute_reward-style signature: {fn.__name__}{sig}",
            hint="Preferred for new code: def my_reward(info: RewardInfo) -> float",
        )
    if n_required > 6:
        return CheckResult(
            "signature",
            "fail",
            f"too many required args ({n_required}): {fn.__name__}{sig}",
            hint=(
                "Use def my_reward(info: RewardInfo) -> float\n"
                + STEP_RESULT_REQUIREMENTS.strip()
            ),
        )
    return CheckResult(
        "signature",
        "warn",
        f"unexpected required-arg count ({n_required}): {fn.__name__}{sig}",
        hint="Preferred: def my_reward(info: RewardInfo) -> float",
    )


def _is_simple_reward_fn(fn: RewardFn) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = [
        p for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    n_required = sum(1 for p in params if p.default is p.empty)
    return n_required <= 1


def _call_reward(fn: RewardFn, step_result: dict, cfg: RewardConfig) -> Any:
    totals, before, after, hosts = default_bin_context()
    if _is_simple_reward_fn(fn):
        info = reward_info_from_step(
            step_result,
            bin_totals=totals,
            bin_observed_before=before,
            bin_observed_after=after,
            host_tier1_before=hosts,
        )
        return fn(info)
    try:
        return fn(step_result, cfg, totals, before, after, hosts)
    except TypeError:
        return fn(step_result, cfg, totals, before, after)


def _check_correctness(fn: RewardFn, cfg: RewardConfig) -> list[CheckResult]:
    out: list[CheckResult] = []

    # Missing required keys should not hard-crash if the author uses .get;
    # but crashing is a fail for production rewards.
    try:
        sparse = {"missed": True}
        val = _call_reward(fn, sparse, cfg)
        if not _is_finite_number(val):
            out.append(
                CheckResult(
                    "sparse_step_result",
                    "fail",
                    f"returned non-finite {val!r} on minimal missed step_result",
                    hint="Use defaults / getattr-style access for optional fields.",
                )
            )
        else:
            out.append(
                CheckResult(
                    "sparse_step_result",
                    "ok",
                    "handles minimal missed step_result without crashing",
                )
            )
    except Exception as exc:
        out.append(
            CheckResult(
                "sparse_step_result",
                "fail",
                f"crashed on minimal step_result: {exc}",
                hint=(
                    "For simple rewards, read attributes on RewardInfo with defaults "
                    "already filled.\n"
                    "For legacy rewards, use step_result.get(key, default).\n"
                    f"Required keys for full path: {', '.join(REQUIRED_STEP_RESULT_KEYS)}"
                ),
            )
        )

    failures: list[str] = []
    for i, sr in enumerate(step_result_battery()):
        try:
            val = _call_reward(fn, sr, cfg)
            if not _is_finite_number(val):
                failures.append(f"case[{i}] non-finite {val!r}")
        except Exception as exc:
            failures.append(f"case[{i}] raised {type(exc).__name__}: {exc}")
    if failures:
        out.append(
            CheckResult(
                "battery",
                "fail",
                f"{len(failures)}/{len(step_result_battery())} cases failed",
                details="\n".join(failures[:8]),
                hint=STEP_RESULT_REQUIREMENTS.strip(),
            )
        )
    else:
        out.append(
            CheckResult(
                "battery",
                "ok",
                f"all {len(step_result_battery())} synthetic step_results returned finite floats",
            )
        )
    return out


def _check_speed(
    fn: RewardFn,
    cfg: RewardConfig,
    *,
    mean_budget_us: float,
    p95_budget_us: float,
    warn_mean_us: float,
    n_timing: int,
) -> CheckResult:
    cases = step_result_battery()
    # Warm-up
    for sr in cases:
        _call_reward(fn, sr, cfg)

    samples: list[float] = []
    # Interleave cases so timing isn't dominated by one branch.
    for i in range(n_timing):
        sr = cases[i % len(cases)]
        t0 = time.perf_counter()
        _call_reward(fn, sr, cfg)
        samples.append((time.perf_counter() - t0) * 1e6)

    arr = np.asarray(samples, dtype=np.float64)
    mean = float(arr.mean())
    p95 = float(np.percentile(arr, 95))
    detail = (
        f"mean={mean:.2f} µs  p50={float(np.percentile(arr, 50)):.2f} µs  "
        f"p95={p95:.2f} µs  n={n_timing}\n"
        f"budgets: warn_mean={warn_mean_us:.0f}  fail_mean={mean_budget_us:.0f}  "
        f"fail_p95={p95_budget_us:.0f} µs\n"
        f"stock compute_reward is typically ~2 µs/call on a laptop."
    )
    hint = (
        "Keep rewards pure functions of step_result + small dicts.\n"
        "Avoid pandas, disk I/O, Python loops over the full catalogue, and "
        "large temporary arrays inside the reward."
    )

    if mean > mean_budget_us or p95 > p95_budget_us:
        return CheckResult(
            "speed",
            "fail",
            f"too slow (mean={mean:.1f} µs, p95={p95:.1f} µs)",
            details=detail,
            hint=hint,
        )
    if mean > warn_mean_us:
        return CheckResult(
            "speed",
            "warn",
            f"acceptable but slow (mean={mean:.1f} µs)",
            details=detail,
            hint=hint,
        )
    return CheckResult(
        "speed",
        "ok",
        f"within budget (mean={mean:.1f} µs, p95={p95:.1f} µs)",
        details=detail,
    )
