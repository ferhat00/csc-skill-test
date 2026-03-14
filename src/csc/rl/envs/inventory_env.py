"""Gymnasium environment for Inventory & Safety Stock RL agent."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from csc.data.master_generator import MasterGenerator
from csc.orchestrator.state import SharedState
from csc.rl.obs_builders import MAX_INVENTORY_POSITIONS, INV_FEATURES
from csc.rl.rewards import inventory_reward


class InventoryEnv(gym.Env):
    """RL environment for learning reorder points and safety stock levels.

    Observation: per material-location — on_hand, in_transit, on_order, allocated,
                 available, days_of_supply, monthly_demand_estimate, lead_time.
    Action: per material-location — reorder_quantity, safety_stock_target_months.
    Reward: -stockout - holding - expiry + service_level.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42, horizon_months: int = 24):
        super().__init__()
        self._seed = seed
        self._horizon = horizon_months
        self._step_count = 0

        self._obs_dim = MAX_INVENTORY_POSITIONS * INV_FEATURES
        self._act_dim = MAX_INVENTORY_POSITIONS * 2  # reorder_qty + safety_target per position

        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.tile([0.0, 0.5], MAX_INVENTORY_POSITIONS),
            high=np.tile([500.0, 6.0], MAX_INVENTORY_POSITIONS),
            dtype=np.float32,
        )

        self._state: SharedState | None = None
        self._on_hand: np.ndarray = np.zeros(MAX_INVENTORY_POSITIONS)
        self._demand: np.ndarray = np.zeros(MAX_INVENTORY_POSITIONS)
        self._lead_times: np.ndarray = np.zeros(MAX_INVENTORY_POSITIONS)
        self._shelf_life: np.ndarray = np.zeros(MAX_INVENTORY_POSITIONS)
        self._n_positions = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self._seed + self._step_count

        # Generate fresh scenario
        gen = MasterGenerator(seed=episode_seed, num_sites=30)
        gen.generate()

        self._state = SharedState()
        self._state.trials = gen.trials
        self._state.sites = gen.sites
        self._state.depots = gen.depots
        self._state.plants = gen.plants
        self._state.materials = gen.materials
        self._state.equipment_lines = gen.equipment_lines
        self._state.inventory_positions = gen.inventory_positions
        self._state.enrollment_forecasts = gen.enrollment_forecasts

        self._n_positions = min(len(gen.inventory_positions), MAX_INVENTORY_POSITIONS)
        self._step_count = 0

        # Initialize tracking arrays
        self._on_hand = np.zeros(MAX_INVENTORY_POSITIONS)
        self._demand = np.zeros(MAX_INVENTORY_POSITIONS)
        self._lead_times = np.zeros(MAX_INVENTORY_POSITIONS)
        self._shelf_life = np.full(MAX_INVENTORY_POSITIONS, 24.0)

        for i, pos in enumerate(gen.inventory_positions[:MAX_INVENTORY_POSITIONS]):
            self._on_hand[i] = pos.on_hand
            self._demand[i] = max(1.0, pos.on_hand / max(pos.days_of_supply, 1) * 30)
            self._lead_times[i] = 30.0
            self._shelf_life[i] = 24.0

        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Apply stochastic demand noise (±15% CV)
        rng = np.random.default_rng(self._seed + self._step_count)
        noise = 1.0 + rng.normal(0, 0.15, MAX_INVENTORY_POSITIONS)
        actual_demand = self._demand * np.maximum(noise, 0.1)

        # Process reorder actions
        reorder_qty = np.maximum(action[0::2], 0)[:MAX_INVENTORY_POSITIONS]

        # Consume inventory
        consumed = np.minimum(self._on_hand[:MAX_INVENTORY_POSITIONS], actual_demand[:MAX_INVENTORY_POSITIONS])
        self._on_hand[:MAX_INVENTORY_POSITIONS] -= consumed

        # Receive reorders (simplified: arrive after lead time modeled as fraction)
        arrival_frac = np.clip(30.0 / np.maximum(self._lead_times[:MAX_INVENTORY_POSITIONS], 1), 0, 1)
        arrivals = reorder_qty[:MAX_INVENTORY_POSITIONS] * arrival_frac
        self._on_hand[:MAX_INVENTORY_POSITIONS] += arrivals

        # Expiry: reduce shelf life, expire old stock
        self._shelf_life -= 1.0
        expired = np.zeros(MAX_INVENTORY_POSITIONS)
        expired_mask = self._shelf_life <= 0
        expired[expired_mask] = self._on_hand[expired_mask]
        self._on_hand[expired_mask] = 0
        self._shelf_life[expired_mask] = 24.0

        # Days of supply
        safe_demand = np.maximum(actual_demand, 1e-8)
        dos = np.where(
            actual_demand > 0,
            self._on_hand / safe_demand * 30,
            np.full(MAX_INVENTORY_POSITIONS, 999.0),
        )

        reward = inventory_reward(
            on_hand=self._on_hand[:self._n_positions],
            demand=actual_demand[:self._n_positions],
            expired=expired[:self._n_positions],
            days_of_supply=dos[:self._n_positions],
        )

        terminated = self._step_count >= self._horizon
        truncated = False

        return self._build_obs(), reward, terminated, truncated, {}

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        for i in range(self._n_positions):
            base = i * INV_FEATURES
            obs[base + 0] = self._on_hand[i]
            obs[base + 1] = 0  # in_transit
            obs[base + 2] = 0  # on_order
            obs[base + 3] = 0  # allocated
            obs[base + 4] = self._on_hand[i]  # available
            dos = self._on_hand[i] / max(self._demand[i], 0.01) * 30
            obs[base + 5] = dos
            obs[base + 6] = self._demand[i]
            obs[base + 7] = self._lead_times[i]
        return obs
