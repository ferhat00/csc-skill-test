"""RL agent for inventory & safety stock — produces SupplyPlan (orders + projections)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from csc.orchestrator.state import SharedState
from csc.rl.action_mappers import map_inventory_action
from csc.rl.base_agent import BaseRLAgent
from csc.rl.obs_builders import build_inventory_obs, IndexMap


class InventoryRLAgent(BaseRLAgent):
    """RL agent that decides reorder quantities and safety stock targets."""

    def __init__(self, state: SharedState, model_path: Path | None = None):
        super().__init__(state, model_path)
        self._inv_map: IndexMap | None = None

    @property
    def agent_name(self) -> str:
        return "inventory_safety_stock"

    def get_input_keys(self) -> list[str]:
        return ["inventory_positions", "materials", "demand_plan", "supply_plan"]

    def get_output_key(self) -> str:
        return "supply_plan"

    def build_observation(self) -> np.ndarray:
        obs, self._inv_map = build_inventory_obs(self.state)
        return obs

    def map_action_to_output(self, action: np.ndarray) -> BaseModel:
        return map_inventory_action(
            action=action,
            state=self.state,
            inv_map=self._inv_map,
        )

    def create_env(self, seed: int = 42) -> Any:
        from csc.rl.envs.inventory_env import InventoryEnv
        return InventoryEnv(seed=seed)
