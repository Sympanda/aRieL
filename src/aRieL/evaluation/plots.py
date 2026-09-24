"""Diagnostic and summary plots for the Ariel RL mission scheduler.

Three main use-cases
────────────────────
1. **Episode-level** – inspect the result of a single finished episode:
   - ``plot_episode_summary``   4-panel overview (tiers, time budget, bin coverage)
   - ``plot_schedule_timeline`` Gantt chart of the observation sequence
   - ``plot_coverage_heatmap``  Radius × temperature completion heat-map
   - ``plot_activity_timeline`` Horizontal bar chart of mission activities by month
                                (slew=red, T1/T2/T3 obs=blue shades, idle=grey)
   - ``plot_reward_curve``      Smoothed per-step reward + cumulative reward over time

2. **Comparison** – compare multiple agents / baselines:
   - ``plot_agent_comparison``  Grouped bar charts with ± std error bars

3. **Training** – track evolution over RL training episodes:
   - ``plot_training_curves``       Episode reward, length, and optional RL losses
   - ``plot_scientific_objectives`` Tier rates, bin coverage, Gini over training

Colour palette
──────────────
All plots use the Paul Tol colourblind-safe palette:
  - slew:  vivid red  (#EE6677)
  - T1:    sky blue   (#66CCEE)
  - T2:    mid blue   (#4477AA)
  - T3:    navy blue  (#004488)
  - idle:  mid grey   (#BBBBBB)
  - fail:  orange     (#EE7733)

All functions return ``(fig, axes)`` so callers can save or further customise.

Quick usage
───────────
>>> from aRieL.evaluation.plots import (
...     plot_episode_summary, plot_schedule_timeline,
...     plot_coverage_heatmap, plot_agent_comparison,
...     plot_training_curves, plot_scientific_objectives,
...     plot_activity_timeline, plot_reward_curve,
... )
>>> fig, axes = plot_episode_summary(env.state)
>>> fig.savefig("episode_summary.png", dpi=150, bbox_inches="tight")
>>>
>>> stats, log_df = run_episode_with_log(env, agent)
>>> fig, ax = plot_reward_curve(log_df, agent_name="SmartGreedy")
>>> fig, ax = plot_activity_timeline(log_df, agent_name="SmartGreedy")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from aRieL.simulator.mission_state import MissionState

# ── colour palette (Paul Tol colourblind-safe) ─────────────────────────────
# Tier colours use a single-hue blue ramp so they are distinguishable even
# under deuteranopia / protanopia (the most common forms).  Slew is a vivid
# red so it stands apart from the blue observation segments and the neutral
# grey idle bars.

_C = {
    # Observation tiers — light → dark blue ramp
    "t1":      "#88CCEE",   # sky-blue    – Tier 1 (easiest / most common)
    "t2":      "#4477AA",   # mid-blue    – Tier 2
    "t3":      "#114477",   # dark navy   – Tier 3

    # Activity types in the timeline
    "slew":    "#CC3311",   # vivid red   – telescope slewing
    "missed":  "#882255",   # dark plum   – failed / missed obs (shouldn't appear after mask fix)
    "idle":    "#BBBBBB",   # mid-grey    – waiting for next window

    # Other uses
    "science": "#4477AA",   # mid-blue    – science time bars
    "none":    "#EEEEEE",   # light grey  – incomplete / background
    "reward":  "#AA3377",   # magenta     – reward curves
    "entropy": "#009988",   # teal        – entropy / diversity
}

_TIER_COLORS = [_C["t1"], _C["t2"], _C["t3"]]

# ── shared style helper ─────────────────────────────────────────────────────

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.titleweight":  "bold",
        "axes.labelsize":    10,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        100,
    })


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Simple centred moving average for smoothing training curves."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values.astype(float), window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Episode summary (4-panel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_episode_summary(
    state: "MissionState",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (15, 10),
) -> tuple[plt.Figure, np.ndarray]:
    """4-panel overview of a completed (or mid-episode) mission.

    Panels
    ──────
    A (top-left)    Tier completion funnel — completed vs eligible per tier
    B (top-right)   Time budget — science / slew / idle breakdown
    C (bottom-left) Population bin coverage — tier-1 rate per bin
    D (bottom-right) Key metrics scorecard

    Parameters
    ----------
    state:
        A finished ``MissionState`` (after ``env.step`` returned
        ``terminated=True``), or any mid-episode snapshot.
    title:
        Optional suptitle string.
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, axes  (2-D numpy array, shape (2, 2))
    """
    _apply_style()
    from aRieL.evaluation.metrics import compute_stats
    from aRieL.evaluation.population_coverage import coverage_table

    stats = compute_stats(state)
    cov   = coverage_table(state)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    # ── A: Tier completion funnel ────────────────────────────────────────
    ax = axes[0, 0]
    tiers     = ["Tier 1", "Tier 2", "Tier 3"]
    completed = [stats.tier1_completed, stats.tier2_completed, stats.tier3_completed]
    # Funnel bars use the live eligible counts; % annotations use initial denom.
    eligible  = [stats.tier1_eligible,  stats.tier2_eligible,  stats.tier3_eligible]
    rates     = [stats.tier1_rate, stats.tier2_rate, stats.tier3_rate]
    remaining = [max(e - c, 0) for e, c in zip(eligible, completed)]

    y = np.arange(len(tiers))
    bar_h = 0.5
    ax.barh(y, completed, bar_h, color=[_C["t1"], _C["t2"], _C["t3"]],
            label="Completed", zorder=3)
    ax.barh(y, remaining, bar_h, left=completed, color=_C["none"],
            label="Not completed", zorder=3)
    # Dotted line at initial catalogue size (100 % reference)
    ax.axvline(stats.n_initial_targets, color="#555", linestyle=":",
               linewidth=1.4, zorder=5, label="Initial catalogue (100%)")
    if stats.total_targets != stats.n_initial_targets:
        ax.axvline(stats.total_targets, color="#888", linestyle="--",
                   linewidth=1.2, zorder=5, label="Final catalogue")
    ax.set_yticks(y)
    ax.set_yticklabels(tiers)
    ax.set_xlabel("Number of targets")
    ax.set_title("A — Tier completion")
    ax.legend(loc="lower right", framealpha=0.8)
    # Annotate fraction vs initial catalogue (may exceed 100 %)
    for i, (c, e, r) in enumerate(zip(completed, eligible, rates)):
        ax.text(c + max(remaining[i], 1) * 0.02, i,
                f"{c}/{stats.n_initial_targets} ({r:.0%} of initial)",
                va="center", fontsize=9)
    ax.set_xlim(0, max(max(eligible), stats.n_initial_targets) * 1.25)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", alpha=0)

    # ── B: Time budget ───────────────────────────────────────────────────
    ax = axes[0, 1]
    clk = state.clock
    science = clk.used_science_time
    slew    = clk.used_slew_time
    elapsed = clk.elapsed_time
    idle    = max(0.0, elapsed - science - slew)
    labels  = ["Science", "Slew", "Idle"]
    sizes   = [science, slew, idle]
    colors  = [_C["science"], _C["slew"], _C["idle"]]

    # Only show non-zero slices
    nz = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0.01]
    if nz:
        lbls, szs, clrs = zip(*nz)
        wedges, texts, autotexts = ax.pie(
            szs, labels=lbls, colors=clrs, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 9},
        )
        for at in autotexts:
            at.set_fontsize(8)
    ax.set_title("B — Time budget")
    ax.text(0, -1.35, f"Total elapsed: {elapsed:.1f} d  |  "
            f"Mission: {clk.mission_end - clk.mission_start:.0f} d",
            ha="center", fontsize=8.5, color="#555")

    # ── C: Population bin coverage ───────────────────────────────────────
    ax = axes[1, 0]
    if not cov.empty:
        bins_sorted = cov.sort_values("tier1_rate", ascending=True)
        n_bins = len(bins_sorted)
        y2 = np.arange(n_bins)
        ax.barh(y2, bins_sorted["tier1_rate"], 0.5, color=_C["t1"],
                label="Tier 1", alpha=0.85, zorder=3)
        ax.barh(y2, bins_sorted["tier2_rate"], 0.5, left=0, color=_C["t2"],
                alpha=0, zorder=2)   # invisible – just for legend marker below

        # Overlay Tier 2 as a dot marker
        ax.scatter(bins_sorted["tier2_rate"], y2, color=_C["t2"],
                   s=25, zorder=4, label="Tier 2", marker="D")
        ax.set_yticks(y2)
        ax.set_yticklabels(
            [b.replace("_", " ") for b in bins_sorted["population_bin"]],
            fontsize=8,
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Completion rate")
        ax.set_title("C — Population bin coverage")
        ax.legend(loc="lower right", framealpha=0.8)
        ax.grid(axis="x", alpha=0.3)
        ax.grid(axis="y", alpha=0)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        ax.set_title("C — Population bin coverage")

    # ── D: Scorecard ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    ax.set_title("D — Key metrics", pad=8)

    rows = [
        ("Observations",  f"{stats.n_observations}"),
        ("Missed events", f"{stats.n_missed}  ({stats.miss_rate:.1%})"),
        ("Science time",  f"{stats.used_science_days:.2f} d"),
        ("Slew time",     f"{stats.used_slew_days:.2f} d"),
        ("Sci. efficiency",f"{stats.science_efficiency:.1%}"),
        ("T1 of initial", f"{stats.tier1_rate:.1%}"
                          + (f"  (ceiling {stats.catalogue_ceiling_rate:.1%})"
                             if stats.catalogue_ceiling_rate > 1.0 + 1e-9 else "")),
        ("Bins covered",  f"{stats.n_bins_with_t1} / {stats.n_bins_total}"
                          f"  ({stats.bin_coverage:.1%})"),
        ("Gini (T1)",     f"{stats.coverage_gini_t1:.3f}"),
        ("Gini (T2)",     f"{stats.coverage_gini_t2:.3f}"),
        ("Mission used",  f"{elapsed:.1f} / "
                          f"{clk.mission_end - clk.mission_start:.0f} d"),
    ]
    col_widths = [0.55, 0.45]
    table_data = [[r[0], r[1]] for r in rows]
    tbl = ax.table(
        cellText=table_data,
        colWidths=col_widths,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if col == 0:
            cell.set_facecolor("#F5F5F5")
        else:
            cell.set_facecolor("white")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Schedule timeline (Gantt chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_schedule_timeline(
    state: "MissionState",
    max_targets: Optional[int] = 40,
    color_by: str = "tier",
    figsize: tuple[float, float] = (16, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Gantt chart of all observations in an episode.

    Each row = one observed target.  Each bar = one observation event.
    Missed events are shown as red crosses.

    Parameters
    ----------
    state:
        A finished ``MissionState`` with a populated ``obs_log``.
    max_targets:
        Cap the number of targets displayed (those observed most often are
        shown first).  Set to ``None`` to show all.
    color_by:
        ``"tier"``   – colour bars by the tier reached after the observation.
        ``"target"`` – each target gets a distinct colour.
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, ax
    """
    _apply_style()
    log = state.obs_log_df()

    if log.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No observations recorded\n(obs_log is empty)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="#888")
        ax.set_title("Schedule timeline")
        return fig, ax

    mission_days = state.clock.mission_end - state.clock.mission_start

    # Select targets to display
    obs_counts = log.groupby("target_id").size().sort_values(ascending=False)
    if max_targets and len(obs_counts) > max_targets:
        top_targets = obs_counts.head(max_targets).index.tolist()
        log = log[log["target_id"].isin(top_targets)]

    targets_ordered = (
        log.groupby("target_id")["mission_day"].min()
        .sort_values()
        .index.tolist()
    )

    target_y = {tid: i for i, tid in enumerate(targets_ordered)}
    n_targets = len(targets_ordered)

    # Colour maps
    if color_by == "tier":
        tier_colors = {0: _C["none"], 1: _C["t1"], 2: _C["t2"], 3: _C["t3"]}
    elif color_by == "target":
        cmap = plt.cm.get_cmap("tab20", n_targets)
        target_colors = {tid: cmap(i) for i, tid in enumerate(targets_ordered)}

    fig, ax = plt.subplots(figsize=figsize)
    bar_h      = 0.6   # observation bar height
    slew_h     = 0.25  # slew band height (centred on same y)
    idle_h     = 0.35  # idle band height (slightly taller than slew)

    SLEW_CLR = "#aaaaaa"   # grey  — slewing
    IDLE_CLR = "#f5a623"   # amber — idle wait before window

    for _, row in log.iterrows():
        tid      = row["target_id"]
        ypos     = target_y[tid]
        mid      = row["mission_day"]
        dur      = max(float(row.get("obs_duration_days", 0)), 0.05)
        idle_d   = max(float(row.get("idle_days",  0)), 0.0)
        slew_d   = max(float(row.get("slew_days",  0)), 0.0)

        # Reconstruct timeline: slew → idle → observation (obs centred on window_mid)
        obs_start  = mid - dur / 2
        idle_start = obs_start - idle_d
        slew_start = idle_start - slew_d

        if row["missed"]:
            ax.scatter(mid, ypos, marker="x", color=_C["missed"],
                       s=60, zorder=5, linewidths=1.5)
            # Still show the slew/idle cost of a missed event
            if slew_d > 0:
                ax.barh(ypos, slew_d, slew_h, left=slew_start,
                        color=SLEW_CLR, alpha=0.5, zorder=2)
            if idle_d > 0:
                ax.barh(ypos, idle_d, idle_h, left=idle_start,
                        color=IDLE_CLR, alpha=0.5, zorder=2)
        else:
            if color_by == "tier":
                clr = tier_colors.get(int(row["tier_after"]), _C["none"])
            else:
                clr = target_colors[tid]

            # Slew band
            if slew_d > 0.001:
                ax.barh(ypos, slew_d, slew_h, left=slew_start,
                        color=SLEW_CLR, alpha=0.7, zorder=2)
            # Idle band (on-target wait before window opens)
            if idle_d > 0.001:
                ax.barh(ypos, idle_d, idle_h, left=idle_start,
                        color=IDLE_CLR, alpha=0.85, zorder=3,
                        hatch="///", edgecolor="white", linewidth=0.0)
            # Observation bar
            ax.barh(ypos, dur, bar_h, left=obs_start, color=clr,
                    edgecolor="white", linewidth=0.3, zorder=4, alpha=0.9)

    ax.set_yticks(range(n_targets))
    ax.set_yticklabels(targets_ordered, fontsize=max(6, 9 - n_targets // 10))
    ax.set_xlim(0, mission_days)
    ax.set_xlabel("Mission day")
    ax.set_ylabel("Target")
    ax.set_title("Observation schedule timeline  (slew=grey · idle=amber · obs=tier colour)")
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", alpha=0)

    # Legend
    if color_by == "tier":
        legend_patches = [
            mpatches.Patch(color=_C["t1"],  label="Tier 1 reached"),
            mpatches.Patch(color=_C["t2"],  label="Tier 2 reached"),
            mpatches.Patch(color=_C["t3"],  label="Tier 3 reached"),
            mpatches.Patch(color=_C["none"],label="No tier yet"),
            mpatches.Patch(color=IDLE_CLR,  label="Idle wait",  hatch="///", edgecolor="white"),
            mpatches.Patch(color=SLEW_CLR,  label="Slew"),
            plt.Line2D([0], [0], marker="x", color=_C["missed"],
                       linestyle="None", markersize=8, label="Missed"),
        ]
        ax.legend(handles=legend_patches, loc="upper right",
                  framealpha=0.9, fontsize=9)

    # Summary stats as a text box
    n_obs  = int((~log["missed"]).sum())
    n_miss = int(log["missed"].sum())
    ax.text(
        0.01, 1.01,
        f"Observations: {n_obs}   Missed: {n_miss}   "
        f"Targets observed: {n_targets}",
        transform=ax.transAxes, fontsize=9, color="#444",
    )

    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Population coverage heat-map
# ═══════════════════════════════════════════════════════════════════════════

def plot_coverage_heatmap(
    state: "MissionState",
    tier: int = 1,
    figsize: tuple[float, float] = (10, 5),
    cmap: str = "Blues",
) -> tuple[plt.Figure, plt.Axes]:
    """2-D heat-map of tier completion across the planet population grid.

    Rows = planet radius class (sub-earth → Jupiter).
    Columns = planet temperature class (cold → ultra-hot).
    Cell values = fraction of targets in that cell at the given tier.

    A completely dark/empty cell means that class was never completed;
    a bright cell means all eligible targets reached this tier.

    Parameters
    ----------
    state:
        Finished ``MissionState``.
    tier:
        Which tier to visualise (1, 2, or 3).
    figsize:
        Figure size in inches.
    cmap:
        Matplotlib colour map name.

    Returns
    -------
    fig, ax
    """
    _apply_style()
    from aRieL.evaluation.population_coverage import (
        coverage_matrix,
        coverage_counts_matrix,
    )

    mat = coverage_matrix(state, tier=tier)

    if mat.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No coverage data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#888")
        return fig, ax

    completed_mat, total_mat = coverage_counts_matrix(state, tier=tier)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, vmin=0, vmax=1, cmap=cmap, aspect="auto")

    # Annotate cells with "completed / total" counts
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val        = mat.values[r, c]
            completed  = int(completed_mat.values[r, c])
            total      = int(total_mat.values[r, c])
            text_color = "white" if val > 0.55 else "#333333"
            # Two-line label: fraction bar on top, count on bottom
            label = f"{completed}/{total}"
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(
        [c.replace("_", "-") for c in mat.columns],
        rotation=30, ha="right",
    )
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([r.replace("_", "-") for r in mat.index])
    ax.set_xlabel("Planet temperature class  →  hotter")
    ax.set_ylabel("Planet radius class  →  larger")
    ax.set_title(f"Tier {tier} completion by population class  (colour = fraction completed)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Completion fraction", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.text(0.5, -0.18,
            "Cell label = completed / total targets in that radius × temperature class",
            ha="center", transform=ax.transAxes, fontsize=8.5, color="#555")

    fig.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Agent comparison (grouped bars)
# ═══════════════════════════════════════════════════════════════════════════

def plot_agent_comparison(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    figsize: tuple[float, float] = (14, 9),
) -> tuple[plt.Figure, np.ndarray]:
    """Grouped bar charts comparing multiple agents across episode statistics.

    Expects the output of ``compare_baselines(env, agents, n_episodes=N)``.
    Each bar shows the per-agent mean; error bars show ± 1 std over episodes.

    Parameters
    ----------
    df:
        DataFrame with columns: ``agent``, and metric columns.
        (Output of ``evaluation.compare_runs.compare_baselines``.)
    metrics:
        List of column names to plot.  Defaults to a useful subset.
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, axes  (1-D numpy array)
    """
    _apply_style()

    default_metrics = [
        ("tier1_rate",           "T1 completion (% all targets)",        True),
        ("tier2_of_eligible",    "T2 completion (% T2-eligible targets)", True),
        ("tier3_of_eligible",    "T3 completion (% T3-eligible targets)", True),
        ("bin_coverage",         "Population bin coverage",               True),
        ("science_efficiency",   "Science efficiency (excl. idle+slew)",  True),
        ("miss_rate",            "Miss rate (lower=better)",              False),
    ]

    if metrics is not None:
        plot_specs = [(m, m.replace("_", " ").title(), True) for m in metrics]
    else:
        plot_specs = [(m, lbl, hi) for m, lbl, hi in default_metrics
                      if m in df.columns]

    if not plot_specs:
        raise ValueError(f"None of the default metrics found in df.  "
                         f"Available: {df.columns.tolist()}")

    # RL agents (those not in the canonical greedy set) sort to the end.
    _GREEDY = {"RandomValid", "GreedyValue", "GreedyBalanced",
               "EarliestDeadline", "SmartGreedy"}
    all_agents = df["agent"].unique().tolist()
    greedy_agents = [a for a in all_agents if a in _GREEDY]
    rl_agents     = [a for a in all_agents if a not in _GREEDY]
    agents = sorted(greedy_agents) + sorted(rl_agents)
    n_agents = len(agents)
    n_metrics = len(plot_specs)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=figsize, squeeze=False)
    agent_colors = plt.cm.tab10(np.linspace(0, 0.9, n_agents))

    for idx, (col, label, higher_is_better) in enumerate(plot_specs):
        ax = axes[idx // n_cols, idx % n_cols]
        grouped = df.groupby("agent")[col]
        means  = grouped.mean()
        stds   = grouped.std().fillna(0)

        x = np.arange(n_agents)
        bars = ax.bar(
            x,
            [means.get(a, 0) for a in agents],
            0.6,
            yerr=[stds.get(a, 0) for a in agents],
            color=agent_colors,
            capsize=4,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.set_xticks(x)
        ax.set_xticklabels(agents, rotation=15, ha="right")
        ax.set_title(label)
        vals = [means.get(a, 0) for a in agents]
        y_top = max(vals) if vals else 1.0
        # Allow >100 % when catalogue-growth rates exceed 1.0
        ax.set_ylim(0, max(1.15, y_top * 1.25 + 0.05))
        if ("rate" in col or "efficiency" in col or "coverage" in col
                or "eligible" in col):
            ax.set_ylabel("Fraction")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0%}")
            )
            if col in ("tier1_rate", "tier2_rate", "tier3_rate"):
                ax.axhline(1.0, color="#555", linestyle=":", linewidth=1.2,
                           zorder=2, label="100% of initial")
                if "catalogue_ceiling_rate" in df.columns:
                    ceil = float(df["catalogue_ceiling_rate"].mean())
                    if ceil > 1.0 + 1e-9:
                        ax.axhline(ceil, color="#888", linestyle="--",
                                   linewidth=1.2, zorder=2,
                                   label=f"Final catalogue ({ceil:.0%})")
        ax.grid(axis="y", alpha=0.35)
        ax.grid(axis="x", alpha=0)

        # Annotate bars
        for bar_, a in zip(bars, agents):
            v = means.get(a, 0)
            ax.text(bar_.get_x() + bar_.get_width() / 2, bar_.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

        # Arrow hinting direction
        hint = "↑ better" if higher_is_better else "↓ better"
        ax.text(0.98, 0.97, hint, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#777")

    # Hide unused axes
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    # Global legend
    legend_patches = [
        mpatches.Patch(color=agent_colors[i], label=a)
        for i, a in enumerate(agents)
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=n_agents, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, fontsize=10)
    fig.suptitle("Agent comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# 5.  RL training curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    log_df: pd.DataFrame,
    smooth: int = 20,
    figsize: tuple[float, float] = (16, 10),
) -> tuple[plt.Figure, np.ndarray]:
    """Multi-panel plot of RL training metrics over episodes.

    Detects which columns are present and only plots those.

    Expected log_df columns
    ───────────────────────
    Required (at least one):
      episode         – episode index
      episode_reward  – total reward per episode

    Common optional (shown if present):
      episode_length  – number of steps
      policy_loss     – actor / policy gradient loss
      value_loss      – critic loss
      entropy         – policy entropy (higher = more exploration)
      learning_rate   – current lr (shown if varies)
      kl_divergence   – KL penalty (PPO)

    Parameters
    ----------
    log_df:
        One row per training episode.
    smooth:
        Moving-average window for the smoothed overlay (0 = no smoothing).
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, axes
    """
    _apply_style()

    rl_metrics = [
        ("episode_reward",  "Episode reward",      _C["reward"],  True),
        ("episode_length",  "Episode length (steps)", "#546E7A", True),
        ("policy_loss",     "Policy loss",          "#D32F2F", False),
        ("value_loss",      "Value loss",           "#E64A19", False),
        ("entropy",         "Entropy",              _C["entropy"], True),
        ("kl_divergence",   "KL divergence",        "#5D4037", False),
        ("learning_rate",   "Learning rate",        "#616161", None),
    ]

    present = [(col, lbl, clr, hi)
               for col, lbl, clr, hi in rl_metrics
               if col in log_df.columns]

    if not present:
        raise ValueError(
            f"No recognised RL metric columns found in log_df.  "
            f"Got: {log_df.columns.tolist()}"
        )

    x_col = "step" if "step" in log_df.columns else "episode"
    x     = log_df[x_col].to_numpy()
    x_lbl = "Training step" if x_col == "step" else "Episode"

    n = len(present)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=figsize, squeeze=False)

    for idx, (col, label, color, higher_is_better) in enumerate(present):
        ax = axes[idx // n_cols, idx % n_cols]
        y  = log_df[col].to_numpy(dtype=float)

        ax.plot(x, y, color=color, alpha=0.35, linewidth=0.8, label="Raw")
        if smooth > 1:
            ys = _smooth(y, smooth)
            ax.plot(x, ys, color=color, linewidth=2.0, label=f"Smooth (w={smooth})")

        ax.set_xlabel(x_lbl)
        ax.set_title(label)
        if higher_is_better is not None:
            hint = "↑ better" if higher_is_better else "↓ better"
            ax.text(0.98, 0.97, hint, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#777")
        ax.grid(alpha=0.3)

        if smooth > 1:
            ax.legend(fontsize=8)

    # Hide unused
    for idx in range(len(present), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("RL training curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Scientific objectives over training
# ═══════════════════════════════════════════════════════════════════════════

def plot_scientific_objectives(
    log_df: pd.DataFrame,
    smooth: int = 20,
    figsize: tuple[float, float] = (16, 10),
) -> tuple[plt.Figure, np.ndarray]:
    """Track how scientific objectives improve over training episodes.

    Plots tier completion rates, population coverage, Gini diversity,
    schedule efficiency, and miss rate over the training run.

    Expected log_df columns
    ───────────────────────
    Any subset of (auto-detected):
      tier1_rate / tier2_rate / tier3_rate
      tier1_of_eligible / tier2_of_eligible / tier3_of_eligible
      bin_coverage
      coverage_gini_t1 / coverage_gini_t2
      science_efficiency
      miss_rate
      fraction_elapsed

    Parameters
    ----------
    log_df:
        One row per training episode (same format as ``compare_baselines``
        output or your custom training logger).
    smooth:
        Moving-average window for the smoothed overlay.
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, axes
    """
    _apply_style()
    x_col = "step" if "step" in log_df.columns else "episode"
    x     = log_df[x_col].to_numpy(dtype=float)
    x_lbl = "Training step" if x_col == "step" else "Episode"

    # ── Panel specs: (columns_to_overlay, title, y_label, colors) ────────
    panel_specs = []

    # Tier rates on one panel
    tier_cols = [c for c in ("tier1_rate", "tier2_rate", "tier3_rate")
                 if c in log_df.columns]
    if not tier_cols:
        tier_cols = [c for c in ("tier1_of_eligible", "tier2_of_eligible",
                                  "tier3_of_eligible")
                     if c in log_df.columns]
    if tier_cols:
        panel_specs.append((
            tier_cols,
            "Tier completion rates",
            "Completion rate",
            [_C["t1"], _C["t2"], _C["t3"]],
        ))

    # Population bin coverage
    if "bin_coverage" in log_df.columns:
        panel_specs.append((
            ["bin_coverage"],
            "Population bin coverage",
            "Fraction covered",
            ["#26A69A"],
        ))

    # Gini coefficient (both tiers on one panel if available)
    gini_cols = [c for c in ("coverage_gini_t1", "coverage_gini_t2")
                 if c in log_df.columns]
    if gini_cols:
        panel_specs.append((
            gini_cols,
            "Coverage Gini coefficient\n(lower = more uniform)",
            "Gini",
            ["#AB47BC", "#EC407A"],
        ))

    # Science efficiency
    if "science_efficiency" in log_df.columns:
        panel_specs.append((
            ["science_efficiency"],
            "Science efficiency",
            "Fraction",
            [_C["science"]],
        ))

    # Miss rate
    if "miss_rate" in log_df.columns:
        panel_specs.append((
            ["miss_rate"],
            "Miss rate\n(lower = better)",
            "Fraction",
            [_C["missed"]],
        ))

    # n_observations (raw count) — useful to see if agent is becoming greedier
    if "n_observations" in log_df.columns:
        panel_specs.append((
            ["n_observations"],
            "Observations per episode",
            "Count",
            ["#78909C"],
        ))

    if not panel_specs:
        raise ValueError(
            f"No recognised scientific metric columns in log_df.  "
            f"Got: {log_df.columns.tolist()}"
        )

    n = len(panel_specs)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=figsize, squeeze=False)

    for idx, (cols, title, ylabel, colors) in enumerate(panel_specs):
        ax = axes[idx // n_cols, idx % n_cols]

        for col, clr in zip(cols, colors):
            y = log_df[col].to_numpy(dtype=float)
            lbl = col.replace("_", " ")
            ax.plot(x, y, color=clr, alpha=0.3, linewidth=0.8)
            if smooth > 1:
                ax.plot(x, _smooth(y, smooth), color=clr,
                        linewidth=2.0, label=lbl)
            else:
                ax.plot(x, y, color=clr, linewidth=1.5, label=lbl)

        ax.set_xlabel(x_lbl)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        if any("rate" in c or "coverage" in c or "efficiency" in c
               for c in cols):
            y_max = 1.05
            for col in cols:
                y_max = max(y_max, float(np.nanmax(log_df[col].to_numpy(dtype=float))) * 1.1)
            ax.set_ylim(-0.02, y_max)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0%}")
            )
            # Reference lines for catalogue-growth coverage plots
            if any(c.startswith("tier") and c.endswith("_rate") for c in cols):
                ax.axhline(1.0, color="#555", linestyle=":", linewidth=1.2,
                           zorder=1, label="100% of initial")
                if "catalogue_ceiling_rate" in log_df.columns:
                    # Use the last episode's ceiling (schedule is seed-dependent
                    # but typically similar); fall back to column mean.
                    ceil = float(log_df["catalogue_ceiling_rate"].iloc[-1])
                    if ceil > 1.0 + 1e-9:
                        ax.axhline(
                            ceil, color="#888", linestyle="--", linewidth=1.2,
                            zorder=1,
                            label=f"Final catalogue ({ceil:.0%})",
                        )
                ax.legend(fontsize=8)
        elif any("gini" in c for c in cols):
            ax.set_ylim(-0.02, 1.05)

        if len(cols) > 1:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused
    for idx in range(len(panel_specs), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle("Scientific objectives over training", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Sky coverage scatter (bonus: where did the telescope point?)
# ═══════════════════════════════════════════════════════════════════════════

def plot_sky_coverage(
    state: "MissionState",
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """RA/Dec scatter showing which targets were observed and to what tier.

    Useful for spotting whether the scheduler has spatial biases.

    Parameters
    ----------
    state:
        Finished ``MissionState``.
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, ax
    """
    _apply_style()
    targets  = state.targets
    progress = state.progress

    merged = targets.set_index("target_id").join(
        progress[["tier1_done", "tier2_done", "tier3_done", "current_tier"]]
    ).reset_index()

    fig, ax = plt.subplots(figsize=figsize)

    # Not observed
    never = merged[merged["current_tier"] == 0]
    ax.scatter(never["ra"], never["dec"], c=_C["none"], s=12,
               edgecolors="#BBBBBB", linewidths=0.4, zorder=2,
               label="Not observed", alpha=0.7)

    # Observed by tier (plot higher tiers on top)
    for tier, color, label in [
        (1, _C["t1"], "Tier 1"),
        (2, _C["t2"], "Tier 2"),
        (3, _C["t3"], "Tier 3"),
    ]:
        sub = merged[merged["current_tier"] >= tier]
        if not sub.empty:
            ax.scatter(sub["ra"], sub["dec"], c=color, s=25,
                       edgecolors="white", linewidths=0.5, zorder=tier + 2,
                       label=label, alpha=0.9)

    ax.set_xlabel("Right Ascension (deg)")
    ax.set_ylabel("Declination (deg)")
    ax.set_title("Sky coverage — target tier completion")
    ax.invert_xaxis()   # RA increases right-to-left by convention
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 7.  Monthly activity timeline
# ═══════════════════════════════════════════════════════════════════════════

def plot_activity_timeline(
    state: "MissionState",
    month_days: float = 30.0,
    figsize: tuple[float, float] = (16, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar chart showing the **actual ordered sequence** of
    activities for each month of the mission.

    Each row = one calendar month (default 30 days).
    Within each row, coloured segments appear left-to-right in the order
    the scheduler executed them — slew, then observation (or failed slew),
    then the next slew, and so on.  Gaps are idle time waiting for the
    next transit window.

    Activity colours
    ----------------
    - **T1 progress** (blue)   — obs where the target is working toward Tier 1
    - **T2 progress** (orange) — obs working toward Tier 2
    - **T3 progress** (green)  — obs working toward Tier 3
    - **Slew**        (amber)  — telescope re-pointing before a successful obs
    - **Failed slew** (red)    — slew to an event that was fully missed
      (arrived after ``window_end``; no science done, slew cost paid)
    - **Idle**        (grey)   — waiting for the next transit/eclipse window

    Note on "failed": a missed event is fully missed — if the slew takes
    longer than the observation window allows, zero science is obtained.
    There is no partial observation.

    Parameters
    ----------
    state:
        Finished (or mid-episode) ``MissionState``.
    month_days:
        Duration of each row in days (default 30).
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, ax
    """
    _apply_style()
    log_df = state.obs_log_df()

    total_days = state.clock.mission_end - state.clock.mission_start
    n_months   = max(1, int(np.ceil(total_days / month_days)))

    # ── Reconstruct ordered timeline from execution log ───────────────────
    # In the simulator execute_observation runs as:
    #   1. clock.skip_to(window_start)  if clock < window_start  → IDLE
    #   2. clock.advance(slew_days)                               → SLEW
    #   3. clock.advance(obs_duration_days)  if not missed        → OBS
    #
    # window_start ≈ mission_day - obs_duration_days / 2
    # We reconstruct idle gaps by tracking the running clock and comparing
    # it to each window_start.

    style = {
        "t1_obs":  (_C["t1"],      "T1 progress"),
        "t2_obs":  (_C["t2"],      "T2 progress"),
        "t3_obs":  (_C["t3"],      "T3 progress"),
        "slew":    (_C["slew"],    "Slew"),
        "failed":  (_C["missed"],  "Failed slew"),
        "idle":    (_C["idle"],    "Idle / waiting"),
    }

    # segments: list of (clock_start, clock_end, category)
    segments: list[tuple[float, float, str]] = []
    clock = 0.0

    if not log_df.empty:
        for _, row in log_df.iterrows():
            slew_dur = float(row["slew_days"])
            obs_dur  = float(row["obs_duration_days"])
            missed   = bool(row["missed"])
            tier     = int(row.get("tier_before", 0))

            # Approximate window_start from window midpoint and duration
            window_start = float(row["mission_day"]) - obs_dur / 2.0

            # Idle: clock waits at current position until window opens
            slew_start = max(clock, window_start)
            if slew_start > clock + 1e-4:
                segments.append((clock, slew_start, "idle"))
            clock = slew_start

            # Slew (always paid, tagged "failed" if the event was missed)
            if slew_dur > 1e-4:
                cat = "failed" if missed else "slew"
                segments.append((clock, clock + slew_dur, cat))
            clock += slew_dur

            # Observation (only if not missed)
            if not missed and obs_dur > 1e-4:
                tier_cat = {0: "t1_obs", 1: "t2_obs", 2: "t3_obs"}.get(tier, "t1_obs")
                segments.append((clock, clock + obs_dur, tier_cat))
                clock += obs_dur

    # Any remaining mission time not covered by observations
    if clock < total_days - 0.05:
        segments.append((clock, total_days, "idle"))

    # ── Plot: one row per month ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    seen_cats: set[str] = set()
    for seg_start, seg_end, cat in segments:
        color, label = style[cat]

        # A segment can span multiple months — split it at month boundaries.
        t = seg_start
        while t < seg_end - 1e-6:
            m = int(t / month_days)
            if m >= n_months:
                break
            m_end   = (m + 1) * month_days
            chunk   = min(seg_end, m_end) - t
            x_left  = t - m * month_days

            legend_label = label if cat not in seen_cats else "_nolegend_"
            seen_cats.add(cat)

            ax.barh(m, chunk, left=x_left, height=0.72,
                    color=color, alpha=0.88, label=legend_label, edgecolor="none")
            t = min(seg_end, m_end)

    ax.set_yticks(np.arange(n_months))
    ax.set_yticklabels([f"Month {i+1}" for i in range(n_months)], fontsize=9)
    ax.set_xlabel("Days within month")
    ax.set_xlim(0, month_days)
    ax.set_title("Mission activity — sequential order within each month")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9, ncol=3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", alpha=0)

    fig.tight_layout()
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════
# 9.  Reward curve over time
# ═══════════════════════════════════════════════════════════════════════════

def plot_reward_curve(
    logs: dict[str, pd.DataFrame],
    x_axis: str = "mission_day",
    smooth_window: int = 20,
    figsize: tuple[float, float] = (13, 7),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot per-step and cumulative reward over the episode.

    Parameters
    ----------
    logs:
        Dict of ``{agent_name: log_df}`` where each ``log_df`` comes from
        ``run_episode_with_log``.  Pass a single-entry dict to plot one agent.
    x_axis:
        Column to use as the x-axis.  One of ``"step"`` | ``"mission_day"``.
    smooth_window:
        Moving-average window size for the per-step reward trace.
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, axes  (shape (2,))
    """
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.12)

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(logs), 1)))

    for (name, log_df), color in zip(logs.items(), colors):
        if log_df.empty or x_axis not in log_df.columns:
            continue
        x = log_df[x_axis].values
        raw_r = log_df["reward"].values
        cum_r = log_df["cumulative_reward"].values

        smoothed = _smooth(raw_r, smooth_window)

        # Top: per-step reward (smoothed)
        axes[0].plot(x, smoothed, color=color, lw=1.5, label=name, alpha=0.9)
        axes[0].fill_between(x, 0, smoothed, color=color, alpha=0.08)

        # Bottom: cumulative reward
        axes[1].plot(x, cum_r, color=color, lw=2.0, label=name, alpha=0.9)

    axes[0].axhline(0, color="#999", lw=0.8, ls="--")
    axes[0].set_ylabel("Reward per step\n(smoothed)")
    axes[0].set_title("Per-step reward")
    axes[0].legend(loc="upper left", framealpha=0.85)

    axes[1].axhline(0, color="#999", lw=0.8, ls="--")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].set_title("Cumulative reward")
    xlabel = "Mission day" if x_axis == "mission_day" else "Step"
    axes[1].set_xlabel(xlabel)

    fig.suptitle("Reward over episode", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════
# 10. Action timeline — what is the scheduler doing each day?
# ═══════════════════════════════════════════════════════════════════════════

def plot_action_timeline(
    state: "MissionState",
    max_targets: int = 50,
    figsize: tuple[float, float] = (16, 9),
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel timeline showing observations and tier completions over mission days.

    Top panel — Gantt-style observation blocks
        Each horizontal bar is one target.  Colour shows the tier reached by
        the *end* of that observation.  Missed events are shown in red.

    Bottom panel — cumulative tier counts over time
        Running total of how many targets reach T1/T2/T3 as mission days pass.

    Parameters
    ----------
    state:
        Finished (or mid-episode) ``MissionState``.
    max_targets:
        Limit the Gantt to the N most-observed targets (keeps it readable).
    figsize:
        Figure size in inches.

    Returns
    -------
    fig, axes  (shape (2,))
    """
    _apply_style()
    log_df = state.obs_log_df()
    if log_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        for ax in axes:
            ax.text(0.5, 0.5, "No observations recorded", ha="center",
                    va="center", transform=ax.transAxes, fontsize=12)
        return fig, axes

    mission_days = state.clock.mission_end - state.clock.mission_start

    # ── pick the most-observed targets ────────────────────────────────────
    top_targets = (
        log_df[~log_df["missed"]]
        .groupby("target_id")
        .size()
        .nlargest(max_targets)
        .index.tolist()
    )
    gantt_df = log_df[log_df["target_id"].isin(top_targets)].copy()
    target_order = (
        gantt_df.groupby("target_id")["mission_day"].min()
        .sort_values()
        .index.tolist()
    )
    target_y = {t: i for i, t in enumerate(target_order)}

    fig, axes = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.subplots_adjust(hspace=0.3)

    # ── Top: Gantt ─────────────────────────────────────────────────────────
    ax = axes[0]
    tier_colors = {0: _C["none"], 1: _C["t1"], 2: _C["t2"], 3: _C["t3"]}

    for _, row in gantt_df.iterrows():
        y  = target_y[row["target_id"]]
        x0 = float(row["mission_day"])
        dur = float(row["obs_duration_days"]) if not row["missed"] else float(row.get("obs_duration_days", 0.1))
        color = _C["missed"] if row["missed"] else tier_colors.get(int(row["tier_after"]), _C["t1"])
        ax.barh(y, max(dur, 0.15), left=x0, height=0.65,
                color=color, alpha=0.85, edgecolor="none")

    # Slew lines between consecutive observations for the same target
    for tid in top_targets:
        trows = gantt_df[gantt_df["target_id"] == tid].sort_values("mission_day")
        y = target_y[tid]
        for i in range(len(trows) - 1):
            x_end = trows.iloc[i]["mission_day"] + trows.iloc[i]["obs_duration_days"]
            x_next = trows.iloc[i + 1]["mission_day"]
            if x_next > x_end:
                ax.plot([x_end, x_next], [y, y], color="#CCCCCC", lw=0.5, zorder=0)

    ax.set_yticks(list(target_y.values()))
    ax.set_yticklabels(list(target_y.keys()), fontsize=7)
    ax.set_xlim(0, mission_days)
    ax.set_ylabel("Target")
    ax.set_title(f"Observation timeline — top {len(top_targets)} targets by observation count")

    legend_patches = [
        mpatches.Patch(color=_C["t1"],     label="Tier 1 reached"),
        mpatches.Patch(color=_C["t2"],     label="Tier 2 reached"),
        mpatches.Patch(color=_C["t3"],     label="Tier 3 reached"),
        mpatches.Patch(color=_C["missed"], label="Missed"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.85)

    # ── Bottom: cumulative tier counts ────────────────────────────────────
    ax2 = axes[1]

    # Build running totals from obs_log by scanning the full log chronologically
    full = log_df.sort_values("mission_day")
    days = full["mission_day"].values

    t1_cum, t2_cum, t3_cum = [], [], []
    t1_seen, t2_seen, t3_seen = set(), set(), set()

    for _, row in full.iterrows():
        if row["missed"]:
            pass
        else:
            tid = row["target_id"]
            ta  = int(row["tier_after"])
            if ta >= 1:
                t1_seen.add(tid)
            if ta >= 2:
                t2_seen.add(tid)
            if ta >= 3:
                t3_seen.add(tid)
        t1_cum.append(len(t1_seen))
        t2_cum.append(len(t2_seen))
        t3_cum.append(len(t3_seen))

    ax2.plot(days, t1_cum, color=_C["t1"], lw=2.0, label="T1 targets")
    ax2.plot(days, t2_cum, color=_C["t2"], lw=2.0, label="T2 targets")
    ax2.plot(days, t3_cum, color=_C["t3"], lw=2.0, label="T3 targets")
    ax2.fill_between(days, 0, t1_cum, color=_C["t1"], alpha=0.10)

    ax2.set_xlim(0, mission_days)
    ax2.set_xlabel("Mission day")
    ax2.set_ylabel("Targets completed")
    ax2.set_title("Cumulative tier completions over mission time")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    return fig, axes
