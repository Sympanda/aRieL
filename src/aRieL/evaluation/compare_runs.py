"""
Run episodes and compare baselines.

``run_episode(env, agent)``       → EpisodeStats
``compare_baselines(env, agents)`` → pd.DataFrame of summary stats
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from aRieL.evaluation.metrics import EpisodeStats, compute_stats
from aRieL.evaluation.population_coverage import coverage_gini

if TYPE_CHECKING:
    from aRieL.baselines.base import BaselineAgent
    from aRieL.envs.env import ArielEnv


def run_episode_with_log(
    env: "ArielEnv",
    agent: "BaselineAgent",
    seed: int | None = None,
    max_steps: int = 100_000,
) -> tuple[EpisodeStats, pd.DataFrame]:
    """Run one episode and return both ``EpisodeStats`` and a per-step log.

    The per-step log is a DataFrame with columns:
        step, mission_day, reward, cumulative_reward,
        tier1_completed, tier2_completed, tier3_completed

    Useful for plotting reward curves and tier-completion trajectories.
    """
    obs, info = env.reset(seed=seed)
    agent.reset()

    terminated = truncated = False
    step = 0
    records: list[dict] = []
    cumulative_reward = 0.0

    while not (terminated or truncated) and step < max_steps:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        cumulative_reward += reward

        summary = info.get("mission_summary", {})
        records.append({
            "step":               step,
            "mission_day":        summary.get("current_time", 0.0) - env.state.clock.mission_start,
            "reward":             reward,
            "cumulative_reward":  cumulative_reward,
            "tier1_completed":    summary.get("tier1_completed", 0),
            "tier2_completed":    summary.get("tier2_completed", 0),
            "tier3_completed":    summary.get("tier3_completed", 0),
            "invalid_action":     info.get("invalid_action", False),
        })

    log_df = pd.DataFrame(records)
    return compute_stats(env.state), log_df


def run_episode(
    env: "ArielEnv",
    agent: "BaselineAgent",
    seed: int | None = None,
    max_steps: int = 100_000,
    verbose: bool = False,
) -> EpisodeStats:
    """Run one complete episode and return statistics.

    Parameters
    ----------
    env:
        A ready ArielEnv instance.
    agent:
        Any ``BaselineAgent`` (or anything with ``.act(obs, info) → int``).
    seed:
        If provided, passed to ``env.reset(seed=seed)``.
    max_steps:
        Safety cap to prevent infinite loops.
    verbose:
        Print a one-line progress update every 1000 steps.

    Returns
    -------
    EpisodeStats
    """
    obs, info = env.reset(seed=seed)
    agent.reset()

    terminated = truncated = False
    step = 0

    while not (terminated or truncated) and step < max_steps:
        action = agent.act(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        step += 1

        if verbose and step % 1000 == 0:
            summary = info.get("mission_summary", {})
            print(
                f"  step {step:>6}  "
                f"t={summary.get('fraction_elapsed', 0):.2f}  "
                f"T1={summary.get('tier1_completed', '?')}  "
                f"T2={summary.get('tier2_completed', '?')}"
            )

    return compute_stats(env.state)


def compare_baselines(
    env: "ArielEnv",
    agents: dict[str, "BaselineAgent"],
    n_episodes: int = 1,
    seed_start: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run each agent for n_episodes and return a summary DataFrame.

    Parameters
    ----------
    env:
        Shared env instance (reset between episodes).
    agents:
        Dict mapping agent name → BaselineAgent instance.
    n_episodes:
        Number of independent episodes per agent.
    seed_start:
        First seed; subsequent seeds are ``seed_start + episode_index``.
    verbose:
        Print per-agent progress.

    Returns
    -------
    pd.DataFrame
        One row per (agent, episode) with all EpisodeStats fields plus
        ``agent``, ``episode``, ``wall_time_s`` columns.
        For multi-episode runs the caller can ``.groupby("agent").mean()``.
    """
    records = []

    for agent_name, agent in agents.items():
        for ep in range(n_episodes):
            seed = seed_start + ep
            if verbose:
                print(f"\n[{agent_name}] episode {ep+1}/{n_episodes} (seed={seed})")

            t0 = time.perf_counter()
            stats = run_episode(env, agent, seed=seed, verbose=verbose)
            elapsed = time.perf_counter() - t0

            row = stats.to_dict()
            row["agent"]        = agent_name
            row["episode"]      = ep
            row["wall_time_s"]  = round(elapsed, 3)
            row["gini_t1"]      = coverage_gini(env.state, tier=1)
            row["gini_t2"]      = coverage_gini(env.state, tier=2)
            records.append(row)

    return pd.DataFrame(records)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-episode results into a mean ± std summary.

    Returns one row per agent with mean values of all numeric columns.
    """
    numeric_cols = results.select_dtypes(include=[float, int]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "episode"]

    grouped = results.groupby("agent")[numeric_cols]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df  = grouped.std().add_suffix("_std").fillna(0)

    combined = pd.concat([mean_df, std_df], axis=1).sort_index(axis=1)
    return combined.reset_index()
