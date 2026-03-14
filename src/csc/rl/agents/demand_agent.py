"""RL agent for demand forecasting — produces DemandPlan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from csc.orchestrator.state import SharedState
from csc.rl.action_mappers import map_demand_action
from csc.rl.base_agent import BaseRLAgent
from csc.rl.obs_builders import build_demand_obs, IndexMap


class DemandRLAgent(BaseRLAgent):
    """RL agent that predicts enrollment and kit demand per trial-site pair."""

    def __init__(self, state: SharedState, model_path: Path | None = None):
        super().__init__(state, model_path)
        self._trial_site_map: IndexMap | None = None

    @property
    def agent_name(self) -> str:
        return "demand_forecast"

    def get_input_keys(self) -> list[str]:
        return ["trials", "sites", "enrollment_forecasts", "materials"]

    def get_output_key(self) -> str:
        return "demand_plan"

    def build_observation(self) -> np.ndarray:
        obs, self._trial_site_map = build_demand_obs(self.state)
        return obs

    def map_action_to_output(self, action: np.ndarray) -> BaseModel:
        return map_demand_action(
            action=action,
            state=self.state,
            trial_site_map=self._trial_site_map,
        )

    def create_env(self, seed: int = 42) -> Any:
        from csc.rl.envs.demand_env import DemandForecastEnv
        return DemandForecastEnv(seed=seed)
