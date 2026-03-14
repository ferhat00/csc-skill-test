"""Gymnasium environment for Capacity Allocation RL agent."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from csc.data.master_generator import MasterGenerator
from csc.orchestrator.state import SharedState
from csc.rl.obs_builders import MAX_LOCATIONS, MAX_TRIALS, LOC_FEATURES, TRL_FEATURES
from csc.rl.rewards import capacity_allocation_reward


class CapacityAllocationEnv(gym.Env):
    """RL environment for learning capacity allocation across trials.

    Observation: location stats + trial stats.
    Action: allocation fraction matrix [n_locations * n_trials].
    Reward: -infeasibility - imbalance + priority_alignment.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42, horizon_months: int = 24):
        super().__init__()
        self._seed = seed
        self._horizon = horizon_months
        self._step_count = 0

        self._obs_dim = MAX_LOCATIONS * LOC_FEATURES + MAX_TRIALS * TRL_FEATURES
        self._act_dim = MAX_LOCATIONS * MAX_TRIALS

        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self._act_dim,), dtype=np.float32,
        )

        self._state: SharedState | None = None
        self._n_locations = 0
        self._n_trials = 0
        self._loc_capacities: np.ndarray = np.zeros(MAX_LOCATIONS)
        self._trial_priorities: np.ndarray = np.zeros(MAX_TRIALS)
        self._trial_demand: np.ndarray = np.zeros(MAX_TRIALS)
        self._trial_inventory: np.ndarray = np.zeros(MAX_TRIALS)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self._seed + self._step_count

        gen = MasterGenerator(seed=episode_seed, num_sites=30)
        gen.generate()

        self._state = SharedState()
        self._state.trials = gen.trials
        self._state.sites = gen.sites
        self._state.plants = gen.plants
        self._state.depots = gen.depots
        self._state.materials = gen.materials
        self._state.equipment_lines = gen.equipment_lines
        self._state.inventory_positions = gen.inventory_positions

        # Build location capacities
        locations = []
        for p in gen.plants:
            locations.append(p.annual_capacity_kg / 12)
        for d in gen.depots:
            locations.append(float(d.storage_capacity_pallets))
        self._n_locations = min(len(locations), MAX_LOCATIONS)
        self._loc_capacities[:self._n_locations] = locations[:self._n_locations]

        # Build trial priorities and demand
        priority_map = {"phase_iii": 3.0, "phase_ii": 2.0, "phase_i": 1.0}
        rng = np.random.default_rng(episode_seed)
        self._n_trials = min(len(gen.trials), MAX_TRIALS)

        for i, trial in enumerate(gen.trials[:self._n_trials]):
            self._trial_priorities[i] = priority_map.get(trial.phase.value, 1.0)
            self._trial_demand[i] = trial.planned_enrollment * 2  # rough kit demand
            self._trial_inventory[i] = rng.uniform(50, 500)

        self._step_count = 0
        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        rng = np.random.default_rng(self._seed + self._step_count * 200)

        # Parse allocation matrix and softmax-normalize per location
        raw = action[:self._n_locations * self._n_trials].reshape(self._n_locations, self._n_trials)
        alloc = np.zeros_like(raw)
        for loc_i in range(self._n_locations):
            row = raw[loc_i]
            exp_row = np.exp(row - np.max(row))
            alloc[loc_i] = exp_row / (exp_row.sum() + 1e-8)

        # Compute utilization per location
        utilization = alloc.sum(axis=1)

        # Update trial inventory based on allocation and demand noise
        demand_noise = 1.0 + rng.normal(0, 0.15, self._n_trials)
        actual_demand = self._trial_demand[:self._n_trials] * np.maximum(demand_noise, 0.1) / 12

        # Capacity allocated to each trial -> production
        trial_production = np.zeros(self._n_trials)
        for loc_i in range(self._n_locations):
            for trl_i in range(self._n_trials):
                trial_production[trl_i] += alloc[loc_i, trl_i] * self._loc_capacities[loc_i] / self._n_trials

        self._trial_inventory[:self._n_trials] += trial_production
        self._trial_inventory[:self._n_trials] -= np.minimum(
            self._trial_inventory[:self._n_trials],
            actual_demand,
        )

        # Days of supply
        dos = np.where(
            actual_demand > 0,
            self._trial_inventory[:self._n_trials] / actual_demand * 30,
            np.full(self._n_trials, 999.0),
        )

        reward = capacity_allocation_reward(
            utilization=utilization,
            days_of_supply=dos,
            allocation_fractions=alloc.flatten(),
            priority_scores=self._trial_priorities[:self._n_trials],
        )

        terminated = self._step_count >= self._horizon
        truncated = False

        return self._build_obs(), reward, terminated, truncated, {}

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        n_loc_feats = MAX_LOCATIONS * LOC_FEATURES

        for i in range(self._n_locations):
            base = i * LOC_FEATURES
            obs[base + 0] = self._loc_capacities[i]
            obs[base + 1] = 0  # utilization updated during step
            obs[base + 2] = 2  # placeholder num_lines
            obs[base + 3] = 0  # pending campaigns

        for i in range(self._n_trials):
            base = n_loc_feats + i * TRL_FEATURES
            obs[base + 0] = self._trial_priorities[i]
            obs[base + 1] = self._trial_demand[i] / 12
            obs[base + 2] = self._trial_demand[i] / 4  # 3-month demand
            obs[base + 3] = self._trial_inventory[i]
            dos = self._trial_inventory[i] / max(self._trial_demand[i] / 12, 0.01) * 30
            obs[base + 4] = min(dos, 999)

        return obs
