"""Reward functions for all RL agent scopes."""

from __future__ import annotations

import numpy as np


# ── Inventory & Safety Stock ─────────────────────────────────────────────────

def inventory_reward(
    on_hand: np.ndarray,
    demand: np.ndarray,
    expired: np.ndarray,
    days_of_supply: np.ndarray,
    w_stockout: float = 10.0,
    w_holding: float = 0.01,
    w_expiry: float = 5.0,
    w_service: float = 1.0,
) -> float:
    """Compute reward for inventory management decisions.

    Args:
        on_hand: Current inventory per material-location.
        demand: Demand this period per material-location.
        expired: Units expired this period per material-location.
        days_of_supply: Current days of supply per material-location.
        w_stockout: Weight for stockout penalty.
        w_holding: Weight for holding cost.
        w_expiry: Weight for expiry waste.
        w_service: Weight for service level bonus.
    """
    shortfall = np.maximum(demand - on_hand, 0)
    stockout_penalty = np.sum(shortfall) * w_stockout
    holding_cost = np.sum(on_hand) * w_holding
    expiry_waste = np.sum(expired) * w_expiry
    service_bonus = np.sum(days_of_supply > 14) * w_service

    return float(-stockout_penalty - holding_cost - expiry_waste + service_bonus)


# ── Demand Forecasting ───────────────────────────────────────────────────────

def demand_forecast_reward(
    predicted: np.ndarray,
    actual: np.ndarray,
    w_mae: float = 1.0,
    w_bias: float = 5.0,
) -> float:
    """Compute reward for demand forecasting accuracy.

    Args:
        predicted: Predicted values (enrollment or kit demand).
        actual: Actual realized values.
        w_mae: Weight for mean absolute error.
        w_bias: Weight for systematic bias penalty.
    """
    mae = np.mean(np.abs(predicted - actual)) * w_mae
    bias = np.abs(np.mean(predicted - actual)) * w_bias

    return float(-mae - bias)


# ── Batch Sizing & Scheduling ────────────────────────────────────────────────

def batch_scheduling_reward(
    unmet_demand: np.ndarray,
    changeover_days: np.ndarray,
    idle_capacity_days: np.ndarray,
    on_time_batches: int,
    w_unmet: float = 10.0,
    w_changeover: float = 2.0,
    w_idle: float = 0.5,
    w_on_time: float = 2.0,
) -> float:
    """Compute reward for batch sizing and scheduling decisions.

    Args:
        unmet_demand: Unmet demand per material (units short).
        changeover_days: Changeover days incurred per line.
        idle_capacity_days: Unused capacity days per line.
        on_time_batches: Number of batches completed before required date.
        w_unmet: Weight for unmet demand penalty.
        w_changeover: Weight for changeover cost.
        w_idle: Weight for idle capacity cost.
        w_on_time: Weight for on-time completion bonus.
    """
    unmet_penalty = np.sum(unmet_demand) * w_unmet
    changeover_cost = np.sum(changeover_days) * w_changeover
    idle_cost = np.sum(idle_capacity_days) * w_idle
    on_time_bonus = on_time_batches * w_on_time

    return float(-unmet_penalty - changeover_cost - idle_cost + on_time_bonus)


# ── Capacity Allocation ──────────────────────────────────────────────────────

def capacity_allocation_reward(
    utilization: np.ndarray,
    days_of_supply: np.ndarray,
    allocation_fractions: np.ndarray,
    priority_scores: np.ndarray,
    w_infeasible: float = 100.0,
    w_imbalance: float = 2.0,
    w_priority: float = 5.0,
) -> float:
    """Compute reward for capacity allocation decisions.

    Args:
        utilization: Utilization fraction per location (should be <= 1.0).
        days_of_supply: Days of supply per trial.
        allocation_fractions: Capacity allocated per trial (flattened).
        priority_scores: Priority score per trial (higher = more important).
        w_infeasible: Penalty per location exceeding 100% utilization.
        w_imbalance: Weight for supply imbalance across trials.
        w_priority: Weight for priority alignment bonus.
    """
    infeasibility = np.sum(utilization > 1.0) * w_infeasible
    imbalance = np.var(days_of_supply) * w_imbalance

    # Priority alignment: correlation between allocation and priority
    if len(priority_scores) > 1 and np.std(allocation_fractions) > 0:
        # Sum allocations per trial across locations
        n_trials = len(priority_scores)
        n_locations = len(utilization)
        alloc_matrix = allocation_fractions.reshape(n_locations, n_trials)
        trial_alloc = alloc_matrix.sum(axis=0)
        corr = np.corrcoef(trial_alloc, priority_scores)[0, 1]
        priority_bonus = (corr if not np.isnan(corr) else 0.0) * w_priority
    else:
        priority_bonus = 0.0

    return float(-infeasibility - imbalance + priority_bonus)
