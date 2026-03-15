"""RL agent for batch sizing & scheduling — produces SupplyPlan (batches)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from csc.orchestrator.state import SharedState
from csc.rl.action_mappers import map_batch_action
from csc.rl.base_agent import BaseRLAgent
from csc.rl.obs_builders import build_batch_obs, IndexMap


class BatchRLAgent(BaseRLAgent):
    """RL agent that decides batch sizes and production scheduling."""

    def __init__(self, state: SharedState, model_path: Path | None = None):
        super().__init__(state, model_path)
        self._line_map: IndexMap | None = None
        self._mat_map: IndexMap | None = None

    @property
    def agent_name(self) -> str:
        return "batch_scheduling"

    def get_input_keys(self) -> list[str]:
        return ["equipment_lines", "materials", "inventory_positions", "demand_plan", "changeover_rules"]

    def get_output_key(self) -> str:
        return "supply_plan"

    def build_observation(self) -> np.ndarray:
        obs, self._line_map, self._mat_map = build_batch_obs(self.state)
        return obs

    def map_action_to_output(self, action: np.ndarray) -> BaseModel:
        return map_batch_action(
            action=action,
            state=self.state,
            line_map=self._line_map,
            mat_map=self._mat_map,
        )

    def create_env(self, seed: int = 42) -> Any:
        from csc.rl.envs.batch_env import BatchSchedulingEnv
        return BatchSchedulingEnv(seed=seed)
