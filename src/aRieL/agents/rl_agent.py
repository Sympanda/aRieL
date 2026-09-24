"""
RLAgentWrapper: adapts a trained MaskablePPO model to the BaselineAgent
interface so it can be used anywhere a baseline agent is expected —
``run_episode``, ``run_episode_with_log``, ``compare_baselines``,
``save_plots`` in run_short_episode.py, etc.

Usage
-----
    from sb3_contrib import MaskablePPO
    from aRieL.agents.rl_agent import RLAgentWrapper

    model = MaskablePPO.load("outputs/my_run/final_model")
    agent = RLAgentWrapper(model, name="MLP-PPO")

    # Drop-in for any baseline agent
    stats, log_df = run_episode_with_log(env, agent, seed=0)

Loading a saved model
---------------------
    model = MaskablePPO.load(
        "outputs/my_run/final_model",
        custom_objects={
            "policy_class": ArielTransformerPolicy,   # or ArielMlpPolicy
        }
    )

    # Alternatively, the policy is saved inside the zip — SB3 usually
    # reconstructs it automatically without custom_objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass   # avoid hard dependency on SB3 at import time


class RLAgentWrapper:
    """
    Wraps a trained ``MaskablePPO`` model as a drop-in BaselineAgent.

    Parameters
    ----------
    model :
        A loaded ``MaskablePPO`` (or any SB3 policy with a ``.predict``
        method that accepts ``action_masks``).
    deterministic : bool
        If True, always take the argmax action (greedy).  Set False to
        sample from the policy distribution (useful for diversity analysis).
    name : str
        Label used in comparison tables and plot legends.
    """

    def __init__(self, model, deterministic: bool = True, name: str = "RLAgent"):
        self.model = model
        self.deterministic = deterministic
        self.name = name

    # ------------------------------------------------------------------
    # BaselineAgent interface
    # ------------------------------------------------------------------

    def act(self, obs: dict, info: dict) -> int:
        """Select an action from obs + info, honouring the action mask."""
        mask = info.get("action_mask")
        action, _ = self.model.predict(
            obs,
            action_masks=mask,
            deterministic=self.deterministic,
        )
        return int(action)

    def reset(self) -> None:
        """No internal state to reset between episodes."""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        mode = "det" if self.deterministic else "stoch"
        return f"RLAgentWrapper(name={self.name!r}, mode={mode})"

    @classmethod
    def load(
        cls,
        path: str,
        name: str = "RLAgent",
        deterministic: bool = True,
        device: str = "auto",
    ) -> "RLAgentWrapper":
        """Load a saved MaskablePPO model and wrap it.

        Parameters
        ----------
        path :
            Path to the ``.zip`` file saved by ``model.save()``.
            The ``.zip`` extension is optional.
        name :
            Display name for tables / plots.
        deterministic :
            Greedy (True) or sampled (False) action selection.
        device :
            PyTorch device string: ``"auto"``, ``"mps"``, ``"cuda"``, ``"cpu"``.

        Returns
        -------
        RLAgentWrapper
        """
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(path, device=device)
        return cls(model, deterministic=deterministic, name=name)
