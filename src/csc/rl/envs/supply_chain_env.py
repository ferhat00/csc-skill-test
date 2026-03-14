"""Multi-agent PettingZoo AEC environment wrapping all 4 single-agent envs."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

from csc.data.master_generator import MasterGenerator
from csc.rl.envs.batch_env import BatchSchedulingEnv
from csc.rl.envs.capacity_env import CapacityAllocationEnv
from csc.rl.envs.demand_env import DemandForecastEnv
from csc.rl.envs.inventory_env import InventoryEnv


AGENT_ORDER = [
    "demand_forecaster",
    "capacity_allocator",
    "batch_scheduler",
    "inventory_manager",
]

# Map agent names to their single-agent env classes
ENV_CLASSES = {
    "demand_forecaster": DemandForecastEnv,
    "capacity_allocator": CapacityAllocationEnv,
    "batch_scheduler": BatchSchedulingEnv,
    "inventory_manager": InventoryEnv,
}


class SupplyChainMultiAgentEnv(AECEnv):
    """Multi-agent AEC environment for joint supply chain optimization.

    Agents act in sequence each timestep:
    1. demand_forecaster — predicts enrollment and kit demand
    2. capacity_allocator — distributes plant/depot capacity across trials
    3. batch_scheduler — schedules production batches
    4. inventory_manager — sets reorder points and safety stock

    Each agent sees the outputs of previously-acting agents embedded in its
    observation via a shared coordination vector.
    """

    metadata = {"render_modes": [], "name": "supply_chain_v0"}

    def __init__(self, seed: int = 42, horizon_months: int = 24):
        super().__init__()
        self._seed = seed
        self._horizon = horizon_months

        self.possible_agents = list(AGENT_ORDER)
        self.agents = list(AGENT_ORDER)

        # Create sub-environments
        self._sub_envs: dict[str, Any] = {}
        for agent_name in AGENT_ORDER:
            self._sub_envs[agent_name] = ENV_CLASSES[agent_name](seed=seed, horizon_months=horizon_months)

        # Coordination vector: each agent appends a summary of its output
        self._coord_dim = 16  # 4 floats per agent
        self._coord_vector = np.zeros(len(AGENT_ORDER) * self._coord_dim, dtype=np.float32)

        self._step_count = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Per-agent tracking
        self._cumulative_rewards: dict[str, float] = {a: 0.0 for a in self.agents}
        self.rewards: dict[str, float] = {a: 0.0 for a in self.agents}
        self.terminations: dict[str, bool] = {a: False for a in self.agents}
        self.truncations: dict[str, bool] = {a: False for a in self.agents}
        self.infos: dict[str, dict] = {a: {} for a in self.agents}

        self._actions_taken_this_round = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        sub_obs = self._sub_envs[agent].observation_space
        coord_size = len(AGENT_ORDER) * self._coord_dim
        total_dim = sub_obs.shape[0] + coord_size
        return spaces.Box(low=0.0, high=1e6, shape=(total_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._sub_envs[agent].action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        episode_seed = seed if seed is not None else self._seed

        self.agents = list(AGENT_ORDER)
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self._coord_vector = np.zeros(len(AGENT_ORDER) * self._coord_dim, dtype=np.float32)
        self._step_count = 0
        self._actions_taken_this_round = 0

        for agent_name in AGENT_ORDER:
            self._sub_envs[agent_name].reset(seed=episode_seed)

        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def observe(self, agent: str) -> np.ndarray:
        sub_obs = self._sub_envs[agent]._build_obs()
        return np.concatenate([sub_obs, self._coord_vector]).astype(np.float32)

    def step(self, action: np.ndarray) -> None:
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Step the sub-environment
        sub_env = self._sub_envs[agent]
        _, reward, terminated, truncated, info = sub_env.step(action)

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        self.infos[agent] = info

        # Update coordination vector with summary of this agent's action
        agent_idx = AGENT_ORDER.index(agent)
        coord_start = agent_idx * self._coord_dim
        # Store action statistics as coordination signal
        action_summary = np.zeros(self._coord_dim, dtype=np.float32)
        action_flat = action.flatten()
        if len(action_flat) > 0:
            action_summary[0] = float(np.mean(action_flat))
            action_summary[1] = float(np.std(action_flat))
            action_summary[2] = float(np.min(action_flat))
            action_summary[3] = float(np.max(action_flat))
        self._coord_vector[coord_start:coord_start + self._coord_dim] = action_summary

        self._actions_taken_this_round += 1

        # After all agents have acted, advance the round
        if self._actions_taken_this_round >= len(self.agents):
            self._step_count += 1
            self._actions_taken_this_round = 0

            if self._step_count >= self._horizon:
                for a in self.agents:
                    self.terminations[a] = True

        # Advance to next agent
        self.agent_selection = self._agent_selector.next()

    def _was_dead_step(self, action) -> None:
        """Handle steps for terminated/truncated agents."""
        if action is not None:
            pass
        self.agent_selection = self._agent_selector.next()
