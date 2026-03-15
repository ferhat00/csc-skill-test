"""Gymnasium environment for Demand Forecasting RL agent."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from csc.data.master_generator import MasterGenerator
from csc.orchestrator.state import SharedState
from csc.rl.obs_builders import MAX_TRIAL_SITE_PAIRS, TSP_FEATURES
from csc.rl.rewards import demand_forecast_reward


class DemandForecastEnv(gym.Env):
    """RL environment for learning demand prediction policies.

    Observation: per trial-site — enrollment stats, trial metadata.
    Action: per trial-site — predicted_new_patients, predicted_kit_demand.
    Reward: -forecast_MAE - bias_penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 42, horizon_months: int = 24):
        super().__init__()
        self._seed = seed
        self._horizon = horizon_months
        self._step_count = 0

        self._obs_dim = MAX_TRIAL_SITE_PAIRS * TSP_FEATURES
        self._act_dim = MAX_TRIAL_SITE_PAIRS * 2  # patients + kits per pair

        self.observation_space = spaces.Box(
            low=0.0, high=1e6, shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=0.0, high=200.0, shape=(self._act_dim,), dtype=np.float32,
        )

        self._state: SharedState | None = None
        self._trial_site_pairs: list[tuple] = []
        self._enrollment_curves: dict[tuple, list[float]] = {}
        self._n_pairs = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episode_seed = seed if seed is not None else self._seed + self._step_count

        gen = MasterGenerator(seed=episode_seed, num_sites=30)
        gen.generate()

        self._state = SharedState()
        self._state.trials = gen.trials
        self._state.sites = gen.sites
        self._state.materials = gen.materials
        self._state.enrollment_forecasts = gen.enrollment_forecasts

        # Build trial-site pairs and enrollment curves
        self._trial_site_pairs = []
        self._enrollment_curves = {}

        therapy_map = {"oncology": 0, "immunology": 1, "neuroscience": 2, "rare_disease": 3}
        phase_map = {"phase_i": 1, "phase_ii": 2, "phase_iii": 3}

        for trial in gen.trials:
            for site_id in trial.sites:
                if len(self._trial_site_pairs) >= MAX_TRIAL_SITE_PAIRS:
                    break
                pair = (trial.id, site_id)
                self._trial_site_pairs.append(pair)

                # Extract monthly enrollment sequence
                forecasts = sorted(
                    [f for f in gen.enrollment_forecasts if f.trial_id == trial.id and f.site_id == site_id],
                    key=lambda f: f.month,
                )
                self._enrollment_curves[pair] = [f.forecasted_new_patients for f in forecasts]

        self._n_pairs = len(self._trial_site_pairs)
        self._step_count = 0

        return self._build_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        rng = np.random.default_rng(self._seed + self._step_count * 1000)

        # Get actual values for this month with noise
        actual_patients = np.zeros(MAX_TRIAL_SITE_PAIRS)
        actual_kits = np.zeros(MAX_TRIAL_SITE_PAIRS)

        for i, pair in enumerate(self._trial_site_pairs):
            curve = self._enrollment_curves.get(pair, [])
            month_idx = min(self._step_count - 1, len(curve) - 1) if curve else 0
            base_patients = curve[month_idx] if month_idx < len(curve) else 0

            # Stochastic perturbation (±15% CV)
            noise = 1.0 + rng.normal(0, 0.15)
            actual_patients[i] = max(0, base_patients * noise)
            actual_kits[i] = actual_patients[i] * 2  # ~2 kits per patient visit

        # Extract predictions
        pred_patients = np.maximum(action[0::2][:MAX_TRIAL_SITE_PAIRS], 0)
        pred_kits = np.maximum(action[1::2][:MAX_TRIAL_SITE_PAIRS], 0)

        predicted = np.concatenate([pred_patients[:self._n_pairs], pred_kits[:self._n_pairs]])
        actual = np.concatenate([actual_patients[:self._n_pairs], actual_kits[:self._n_pairs]])

        reward = demand_forecast_reward(predicted=predicted, actual=actual)

        terminated = self._step_count >= self._horizon
        truncated = False

        return self._build_obs(), reward, terminated, truncated, {}

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        therapy_map = {"oncology": 0, "immunology": 1, "neuroscience": 2, "rare_disease": 3}
        phase_map = {"phase_i": 1, "phase_ii": 2, "phase_iii": 3}

        for i, (trial_id, site_id) in enumerate(self._trial_site_pairs):
            trial = next((t for t in self._state.trials if t.id == trial_id), None)
            if trial is None:
                continue

            curve = self._enrollment_curves.get((trial_id, site_id), [])
            month_idx = min(self._step_count, len(curve) - 1) if curve else 0
            cum_enrolled = sum(curve[:month_idx + 1]) if curve else 0

            base = i * TSP_FEATURES
            obs[base + 0] = cum_enrolled
            obs[base + 1] = max(0, cum_enrolled * 0.8)  # estimated active
            obs[base + 2] = self._step_count  # months since start
            obs[base + 3] = trial.planned_enrollment
            obs[base + 4] = cum_enrolled / max(trial.planned_enrollment, 1)
            # Rolling rate
            start = max(0, month_idx - 2)
            recent = curve[start:month_idx + 1] if curve else [0]
            obs[base + 5] = sum(recent) / max(len(recent), 1)
            obs[base + 6] = phase_map.get(trial.phase.value, 0)
            obs[base + 7] = therapy_map.get(trial.therapy_area.value, 0)
            site = next((s for s in self._state.sites if s.id == site_id), None)
            obs[base + 8] = site.max_patients if site else 0
            obs[base + 9] = trial.overage_pct

        return obs
