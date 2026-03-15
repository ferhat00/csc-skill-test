"""RL agent for capacity allocation — produces CapacityPlans + PortfolioPlan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from csc.orchestrator.state import SharedState
from csc.rl.action_mappers import map_capacity_action
from csc.rl.base_agent import BaseRLAgent
from csc.rl.obs_builders import build_capacity_obs, IndexMap


class CapacityRLAgent(BaseRLAgent):
    """RL agent that allocates plant/depot capacity across trials."""

    def __init__(self, state: SharedState, model_path: Path | None = None):
        super().__init__(state, model_path)
        self._loc_map: IndexMap | None = None
        self._trl_map: IndexMap | None = None

    @property
    def agent_name(self) -> str:
        return "capacity_allocation"

    def get_input_keys(self) -> list[str]:
        return ["trials", "plants", "depots", "equipment_lines", "materials"]

    def get_output_key(self) -> str:
        return "depot_capacity_plan"

    def build_observation(self) -> np.ndarray:
        obs, self._loc_map, self._trl_map = build_capacity_obs(self.state)
        return obs

    def map_action_to_output(self, action: np.ndarray) -> BaseModel:
        depot_plan, plant_plan, portfolio_plan = map_capacity_action(
            action=action,
            state=self.state,
            loc_map=self._loc_map,
            trl_map=self._trl_map,
        )
        # Store the additional outputs
        self.state.set("plant_capacity_plan", plant_plan)
        self.state.set("portfolio_plan", portfolio_plan)
        return depot_plan

    def create_env(self, seed: int = 42) -> Any:
        from csc.rl.envs.capacity_env import CapacityAllocationEnv
        return CapacityAllocationEnv(seed=seed)
