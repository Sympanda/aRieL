"""
Greedy Hill-Climbing baseline.

Emulates the "greedy hill-climbing" approach described in the Ariel mission
scheduling literature (Morales et al. 2021; Nakhjiri et al. 2023) as a
comparison baseline for the meta-heuristic planners.

Algorithm
---------
The agent scores each candidate event as a **linear combination** of its
observation features:

    score(event) = w · features(event)

This generalises SmartGreedy's fixed formula to a learnable weight vector.

Before the first episode the agent calls ``fit()`` which runs a hill-climbing
loop:

1. **Initialise** ``w`` from SmartGreedy-inspired heuristics.
2. **Evaluate** the current weights by running one full episode and recording
   either cumulative **environment reward** (same ``RewardConfig`` as RL) or
   the weighted tier score  T1 + 3·T2 + 10·T3.
3. **Perturb** ``w`` by adding Gaussian noise (``noise_scale``).
4. **Accept** the perturbation if the fitness improves; reject otherwise.
5. Repeat for ``n_iter`` iterations, tracking the best ``w`` found.

At inference time ``act()`` scores the top-K candidates with the best found
``w`` and picks the highest-scoring valid action — identical in speed to any
other greedy agent.

Typical runtime
---------------
    n_iter = 100, 1-year episode  →  ≈ 2-4 minutes on CPU
    n_iter = 50                   →  ≈ 1-2 minutes

Usage
-----
    hc = HillClimbingGreedy(obs_cfg=env.cfg.observation, env=env, n_iter=100)
    hc.fit(verbose=True)   # run optimisation once

    # Afterwards ``hc`` is a standard BaselineAgent
    stats, log = run_episode_with_log(env, hc, seed=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aRieL.baselines.base import BaselineAgent

if TYPE_CHECKING:
    from aRieL.envs.env import ArielEnv
    from aRieL.utils.config import ObservationConfig


class HillClimbingGreedy(BaselineAgent):
    """Linear-scoring greedy agent whose weights are hill-climb optimised.

    Parameters
    ----------
    obs_cfg:
        ``ObservationConfig`` from the environment (to map feature names →
        column indices in ``obs["events"]``).
    env:
        The ``ArielEnv`` instance used as an oracle during ``fit()``.
        The same env can be used for evaluation afterwards — every episode
        is started with an explicit ``env.reset(seed=...)``.
    n_iter:
        Number of hill-climbing perturbation steps.  Each step runs one full
        episode, so total optimisation time ≈ ``n_iter × episode_time``.
    noise_scale:
        Standard deviation of the Gaussian weight perturbation each iteration.
        Smaller values (0.05) give finer search; larger values (0.3) explore
        more aggressively.
    seed:
        RNG seed for perturbations (passed to ``BaselineAgent``).
    """

    def __init__(
        self,
        obs_cfg: "ObservationConfig",
        env: "ArielEnv",
        n_iter:      int   = 100,
        noise_scale: float = 0.15,
        seed:        int   = 0,
        fitness:     str   = "tier_score",
    ) -> None:
        super().__init__(seed)
        if fitness not in ("tier_score", "env_reward"):
            raise ValueError(
                f"fitness must be 'tier_score' or 'env_reward', got {fitness!r}"
            )
        self.env         = env
        self.n_iter      = n_iter
        self.noise_scale = noise_scale
        self.fitness     = fitness
        self.weights: np.ndarray | None = None

        # Map feature name → column index in the per-action obs matrix
        # (top-K ``events`` or full-set ``planets``).
        if env is not None and getattr(env.cfg.action, "type", "") == "full_set":
            from aRieL.envs.planet_feature_builder import PLANET_FEATURE_NAMES
            feats = list(PLANET_FEATURE_NAMES)
        else:
            feats = list(obs_cfg.event_features) if obs_cfg is not None else []
        self._feat_idx: dict[str, int] = {f: i for i, f in enumerate(feats)}
        self._n_features = len(feats)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _initial_weights(self) -> np.ndarray:
        """SmartGreedy-inspired starting weights."""
        w = np.zeros(self._n_features)
        idx = self._feat_idx
        # Positive drivers
        for name, val in [
            ("science_weight",       1.0),
            ("base_science_value",   0.5),
            ("progress_in_tier",     0.5),
            ("window_urgency_norm",  0.3),
            ("tier_goal_norm",       0.2),
        ]:
            if name in idx:
                w[idx[name]] = val
        # Cost penalties (event-space names and full-set planet aliases)
        for name, val in [
            ("slew_time_days",          -0.5),
            ("slew_norm",               -0.5),
            ("total_time_cost_days",    -0.3),
            ("idle_if_selected_norm",   -0.2),
            ("duration_days",           -0.1),
            ("block_duration_norm",     -0.1),
            ("time_to_block_end_norm",  -0.3),
        ]:
            if name in idx:
                w[idx[name]] = val
        return w

    def _evaluate(self, weights: np.ndarray, seed: int) -> float:
        """Run one full episode with *weights* and return the fitness."""
        obs, info = self.env.reset(seed=seed)
        terminated = truncated = False
        step = 0
        total_reward = 0.0
        while not (terminated or truncated) and step < 200_000:
            action = self._pick(obs, info, weights)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            step += 1

        if self.fitness == "env_reward":
            return total_reward

        prog = self.env.state.progress
        t1 = int(prog["tier1_done"].sum())
        t2 = int(prog["tier2_done"].sum())
        t3 = int(prog["tier3_done"].sum())
        return float(t1 * 1 + t2 * 3 + t3 * 10)

    def _pick(self, obs: dict, info: dict, weights: np.ndarray) -> int:
        """Argmax linear score over valid actions."""
        from aRieL.baselines.base import candidate_matrix

        events = candidate_matrix(obs)
        valid = np.where(info["action_mask"])[0]
        if len(valid) == 0:
            return 0
        if events.shape[1] != len(weights):
            raise ValueError(
                f"Hill-climb weights have {len(weights)} features but obs has "
                f"{events.shape[1]} columns — refit in this action space."
            )
        scores = events @ weights                    # (K,)
        masked = np.full(len(scores), -np.inf)
        masked[valid] = scores[valid]
        return int(np.argmax(masked))

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def fit(self, opt_seed: int = 0, verbose: bool = True) -> float:
        """Run the hill-climbing weight search.

        Parameters
        ----------
        opt_seed:
            Base seed for optimisation episodes.  Each iteration uses
            ``opt_seed + iteration`` so the noise sees different episode
            randomness across iterations.
        verbose:
            Print progress whenever a new best is found.

        Returns
        -------
        float
            Best tier fitness found.
        """
        w     = self._initial_weights()
        best  = self._evaluate(w, seed=opt_seed)
        best_w = w.copy()

        if verbose:
            print(f"  [HC] init  score={best:.1f}")

        for i in range(self.n_iter):
            noise     = self.rng.normal(0.0, self.noise_scale, size=self._n_features)
            candidate = best_w + noise
            score     = self._evaluate(candidate, seed=opt_seed + i + 1)

            if score > best:
                best   = score
                best_w = candidate.copy()
                if verbose:
                    print(f"  [HC] iter {i+1:3d}  score={best:.1f}  ✓ improved")

        self.weights = best_w
        if verbose:
            print(f"  [HC] done   best={best:.1f}  ({self.n_iter} iters)")
        return best

    # ------------------------------------------------------------------
    # BaselineAgent interface
    # ------------------------------------------------------------------

    def act(self, obs: dict, info: dict) -> int:
        if self.weights is None:
            # Fallback if fit() was never called: pick random valid
            valid = self._valid_indices(info)
            return int(self.rng.choice(valid)) if len(valid) > 0 else 0
        return self._pick(obs, info, self.weights)

    def reset(self) -> None:
        """No episode state to reset — weights persist across episodes."""
