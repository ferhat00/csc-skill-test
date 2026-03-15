"""Gymnasium environment for Batch Sizing & Scheduling RL agent."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from csc.data.master_generator import MasterGenerator
from csc.orchestrator.state import SharedState
from csc.rl.obs_builders import MAX_EQUIPMENT_LINES, MAX_MATERIALS, EQL_FEATURES, MAT_FEATURES
from csc.rl.rewards import batch_scheduling_reward


class BatchSchedulingEnv(gym.Env):
    """RL environment for learning batch sizing and production scheduling.

    Observation: equipment line stats + material stats + globals.
    Action: per line — material_to_produce, num_batches, batch_size_multiplier.
    Reward: -unmet_demand - changeover - idle + on_time.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42, horizon_months: int = 24):
        super().__init__()
        self._seed = seed
        self._horizon = horizon_months
        self._step_count = 0

        self._line_obs_dim = MAX_EQUIPMENT_LINES * EQL_FEATURES
        self._mat_obs_dim = MAX_MATERIALS * MAT_FEATURES
        self._obs_dim = self._line_obs_dim + self._mat_obs_dim + 2  # + month_idx + total_demand
        self._act_dim = MAX_EQUIPMENT_LINES * 3  # mat_idx, num_batches, size_mult per line

        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.tile([0.0, 0.0, 0.5], MAX_EQUIPMENT_LINES),
            high=np.tile([float(MAX_MATERIALS), 5.0, 1.5], MAX_EQUIPMENT_LINES),
            dtype=np.float32,
        )

        self._state: SharedState | None = None
        self._n_lines = 0
        self._n_materials = 0
        self._material_demand: np.ndarray = np.zeros(MAX_MATERIALS)
        self._material_inventory: np.ndarray = np.zeros(MAX_MATERIALS)
        self._material_batch_sizes: np.ndarray = np.zeros(MAX_MATERIALS)
        self._line_current_mat: np.ndarray = np.zeros(MAX_EQUIPMENT_LINES, dtype=int)
        self._line_utilization: np.ndarray = np.zeros(MAX_EQUIPMENT_LINES)

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
        self._state.changeover_rules = gen.changeover_rules
        self._state.inventory_positions = gen.inventory_positions

        self._n_lines = min(len(gen.equipment_lines), MAX_EQUIPMENT_LINES)

        # Collect all materials
        all_mats = []
        cat = gen.materials
        for ds in cat.drug_substances:
            all_mats.append({"id": ds.id, "batch": ds.batch_size_kg, "inv": 0.0})
        for dp in cat.drug_products:
            all_mats.append({"id": dp.id, "batch": float(dp.batch_size_units), "inv": 0.0})
        for pp in cat.primary_packs:
            all_mats.append({"id": pp.id, "batch": float(pp.pack_size), "inv": 0.0})
        for fg in cat.finished_goods:
            all_mats.append({"id": fg.id, "batch": float(fg.kits_per_patient_visit * 100), "inv": 0.0})
        self._n_materials = min(len(all_mats), MAX_MATERIALS)

        # Set inventory from positions
        inv_map = {}
        for pos in gen.inventory_positions:
            inv_map[pos.material_id] = inv_map.get(pos.material_id, 0) + pos.on_hand
        for i, m in enumerate(all_mats[:MAX_MATERIALS]):
            self._material_batch_sizes[i] = m["batch"]
            self._material_inventory[i] = inv_map.get(m["id"], 0)

        # Generate base demand
        rng = np.random.default_rng(episode_seed)
        self._material_demand[:self._n_materials] = rng.uniform(10, 200, self._n_materials)
        self._line_current_mat[:] = 0
        self._line_utilization[:] = 0
        self._step_count = 0

        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        rng = np.random.default_rng(self._seed + self._step_count * 100)

        # Stochastic demand noise
        noise = 1.0 + rng.normal(0, 0.15, MAX_MATERIALS)
        actual_demand = self._material_demand * np.maximum(noise, 0.1)

        # Process actions: schedule batches
        produced = np.zeros(MAX_MATERIALS)
        changeover_days = np.zeros(MAX_EQUIPMENT_LINES)
        idle_days = np.zeros(MAX_EQUIPMENT_LINES)
        on_time_count = 0

        for i in range(self._n_lines):
            base = i * 3
            mat_idx = int(np.clip(round(action[base + 0]), 0, max(self._n_materials - 1, 0)))
            num_batches = int(np.clip(round(action[base + 1]), 0, 5))
            size_mult = float(np.clip(action[base + 2], 0.5, 1.5))

            line = self._state.equipment_lines[i]
            available_days = line.available_days_per_month

            if num_batches == 0:
                idle_days[i] = available_days
                continue

            # Check changeover
            if self._line_current_mat[i] != mat_idx and self._line_current_mat[i] != 0:
                changeover_days[i] = 3  # default changeover
                available_days -= 3

            # Produce with yield variability (±5%)
            yield_noise = 1.0 + rng.normal(0, 0.05)
            batch_size = self._material_batch_sizes[mat_idx] * size_mult * max(yield_noise, 0.5)
            days_per_batch = max(1, line.capacity_per_day)
            max_batches = max(1, int(available_days / days_per_batch))
            actual_batches = min(num_batches, max_batches)

            produced[mat_idx] += batch_size * actual_batches
            self._line_current_mat[i] = mat_idx
            used_days = actual_batches * days_per_batch + changeover_days[i]
            self._line_utilization[i] = used_days / line.available_days_per_month
            idle_days[i] = max(0, available_days - actual_batches * days_per_batch)

            if actual_batches == num_batches:
                on_time_count += actual_batches

        # Update inventory
        self._material_inventory[:MAX_MATERIALS] += produced[:MAX_MATERIALS]
        self._material_inventory[:MAX_MATERIALS] -= np.minimum(
            self._material_inventory[:MAX_MATERIALS],
            actual_demand[:MAX_MATERIALS],
        )

        unmet = np.maximum(actual_demand[:self._n_materials] - self._material_inventory[:self._n_materials], 0)

        reward = batch_scheduling_reward(
            unmet_demand=unmet,
            changeover_days=changeover_days[:self._n_lines],
            idle_capacity_days=idle_days[:self._n_lines],
            on_time_batches=on_time_count,
        )

        terminated = self._step_count >= self._horizon
        truncated = False

        return self._build_obs(), reward, terminated, truncated, {}

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        for i in range(self._n_lines):
            line = self._state.equipment_lines[i]
            base = i * EQL_FEATURES
            obs[base + 0] = line.capacity_per_day
            obs[base + 1] = line.available_days_per_month
            obs[base + 2] = float(self._line_current_mat[i])
            obs[base + 3] = 0  # changeover placeholder
            obs[base + 4] = self._line_utilization[i]

        for i in range(self._n_materials):
            base = self._line_obs_dim + i * MAT_FEATURES
            obs[base + 0] = self._material_demand[i]
            obs[base + 1] = self._material_inventory[i]
            obs[base + 2] = self._material_batch_sizes[i]

        obs[self._line_obs_dim + self._mat_obs_dim] = self._step_count
        obs[self._line_obs_dim + self._mat_obs_dim + 1] = float(np.sum(self._material_demand[:self._n_materials]))

        return obs
