"""Observation space construction: SharedState -> numpy arrays with index<->UUID maps."""

from __future__ import annotations

from datetime import date
from uuid import UUID

import numpy as np

from csc.orchestrator.state import SharedState

# Fixed maximum dimensions (matching seed data constraints)
MAX_INVENTORY_POSITIONS = 100
MAX_TRIAL_SITE_PAIRS = 150
MAX_EQUIPMENT_LINES = 40
MAX_MATERIALS = 80
MAX_LOCATIONS = 10
MAX_TRIALS = 20

# Feature counts per entity
INV_FEATURES = 8   # on_hand, in_transit, on_order, allocated, available, dos, demand_est, lead_time
TSP_FEATURES = 10  # cum_enrolled, cum_active, months_since_fsfv, planned, progress, hist_rate, phase, therapy, max_pts, overage
EQL_FEATURES = 5   # cap_per_day, avail_days, current_mat_idx, changeover, utilization
MAT_FEATURES = 3   # pending_demand, inventory, nominal_batch
LOC_FEATURES = 4   # total_capacity, utilization, num_lines, pending_campaigns
TRL_FEATURES = 5   # priority, demand, demand_3m, inventory, dos


class IndexMap:
    """Bidirectional mapping between sequential indices and UUIDs."""

    def __init__(self) -> None:
        self._idx_to_uuid: dict[int, UUID] = {}
        self._uuid_to_idx: dict[UUID, int] = {}

    def add(self, idx: int, uuid: UUID) -> None:
        self._idx_to_uuid[idx] = uuid
        self._uuid_to_idx[uuid] = idx

    def uuid(self, idx: int) -> UUID:
        return self._idx_to_uuid[idx]

    def idx(self, uuid: UUID) -> int:
        return self._uuid_to_idx[uuid]

    def __len__(self) -> int:
        return len(self._idx_to_uuid)


def build_inventory_obs(state: SharedState, current_month: date | None = None) -> tuple[np.ndarray, IndexMap]:
    """Build observation for inventory agent.

    Returns (obs_vector, index_map) where index_map maps position index -> (material_id, location_id).
    """
    obs = np.zeros(MAX_INVENTORY_POSITIONS * INV_FEATURES, dtype=np.float32)
    imap = IndexMap()

    positions = state.inventory_positions or []
    for i, pos in enumerate(positions[:MAX_INVENTORY_POSITIONS]):
        imap.add(i, pos.material_id)
        base = i * INV_FEATURES
        obs[base + 0] = pos.on_hand
        obs[base + 1] = pos.in_transit
        obs[base + 2] = pos.on_order
        obs[base + 3] = pos.allocated
        obs[base + 4] = pos.available
        obs[base + 5] = pos.days_of_supply
        # Estimate monthly demand from demand plan if available
        obs[base + 6] = _estimate_demand(state, pos.material_id, pos.location_id)
        # Estimate lead time from material specs
        obs[base + 7] = _estimate_lead_time(state, pos.material_id)

    return obs, imap


def build_demand_obs(state: SharedState, current_month: date | None = None) -> tuple[np.ndarray, IndexMap]:
    """Build observation for demand forecasting agent.

    Returns (obs_vector, trial_site_map) where map[i] -> (trial_id, site_id).
    """
    obs = np.zeros(MAX_TRIAL_SITE_PAIRS * TSP_FEATURES, dtype=np.float32)
    imap = IndexMap()

    therapy_map = {"oncology": 0, "immunology": 1, "neuroscience": 2, "rare_disease": 3}
    phase_map = {"phase_i": 1, "phase_ii": 2, "phase_iii": 3}

    idx = 0
    for trial in state.trials:
        for site_id in trial.sites:
            if idx >= MAX_TRIAL_SITE_PAIRS:
                break

            # Find enrollment forecasts for this trial-site pair
            forecasts = [
                f for f in state.enrollment_forecasts
                if f.trial_id == trial.id and f.site_id == site_id
            ]

            # Use the UUID pair as a composite key (store trial_id)
            imap.add(idx, trial.id)

            base = idx * TSP_FEATURES
            if forecasts:
                latest = max(forecasts, key=lambda f: f.month)
                obs[base + 0] = latest.cumulative_enrolled
                obs[base + 1] = latest.cumulative_active
                if current_month:
                    months_since = (current_month.year - trial.fsfv.year) * 12 + (current_month.month - trial.fsfv.month)
                    obs[base + 2] = max(0, months_since)
                obs[base + 5] = _rolling_rate(forecasts)
            obs[base + 3] = trial.planned_enrollment
            obs[base + 4] = (obs[base + 0] / max(trial.planned_enrollment, 1))
            obs[base + 6] = phase_map.get(trial.phase.value, 0)
            obs[base + 7] = therapy_map.get(trial.therapy_area.value, 0)
            site = next((s for s in state.sites if s.id == site_id), None)
            obs[base + 8] = site.max_patients if site else 0
            obs[base + 9] = trial.overage_pct

            idx += 1

    return obs, imap


def build_batch_obs(state: SharedState) -> tuple[np.ndarray, IndexMap, IndexMap]:
    """Build observation for batch scheduling agent.

    Returns (obs_vector, line_map, material_map).
    """
    n_line_feats = MAX_EQUIPMENT_LINES * EQL_FEATURES
    n_mat_feats = MAX_MATERIALS * MAT_FEATURES
    obs = np.zeros(n_line_feats + n_mat_feats + 2, dtype=np.float32)
    line_map = IndexMap()
    mat_map = IndexMap()

    # Equipment lines
    for i, line in enumerate(state.equipment_lines[:MAX_EQUIPMENT_LINES]):
        line_map.add(i, line.id)
        base = i * EQL_FEATURES
        obs[base + 0] = line.capacity_per_day
        obs[base + 1] = line.available_days_per_month
        obs[base + 2] = 0  # current material index (0 = idle)
        obs[base + 3] = 0  # changeover days (default)
        obs[base + 4] = 0  # utilization placeholder

    # Materials (all stages flattened)
    all_materials = _collect_all_materials(state)
    for i, (mat_id, mat_info) in enumerate(all_materials[:MAX_MATERIALS]):
        mat_map.add(i, mat_id)
        base = n_line_feats + i * MAT_FEATURES
        obs[base + 0] = mat_info.get("pending_demand", 0)
        obs[base + 1] = mat_info.get("inventory", 0)
        obs[base + 2] = mat_info.get("nominal_batch", 0)

    # Global features
    obs[n_line_feats + n_mat_feats] = 0  # month index
    obs[n_line_feats + n_mat_feats + 1] = sum(
        m.get("pending_demand", 0) for _, m in all_materials
    )

    return obs, line_map, mat_map


def build_capacity_obs(state: SharedState) -> tuple[np.ndarray, IndexMap, IndexMap]:
    """Build observation for capacity allocation agent.

    Returns (obs_vector, location_map, trial_map).
    """
    n_loc_feats = MAX_LOCATIONS * LOC_FEATURES
    n_trl_feats = MAX_TRIALS * TRL_FEATURES
    obs = np.zeros(n_loc_feats + n_trl_feats, dtype=np.float32)
    loc_map = IndexMap()
    trl_map = IndexMap()

    # Locations (plants + depots)
    locations = []
    for p in state.plants:
        locations.append((p.id, p.equipment_lines, p.annual_capacity_kg / 12))
    for d in state.depots:
        locations.append((d.id, d.packaging_lines + d.labeling_lines, d.storage_capacity_pallets))

    for i, (loc_id, num_lines, capacity) in enumerate(locations[:MAX_LOCATIONS]):
        loc_map.add(i, loc_id)
        base = i * LOC_FEATURES
        obs[base + 0] = capacity
        obs[base + 1] = 0  # utilization placeholder
        obs[base + 2] = num_lines
        obs[base + 3] = 0  # pending campaigns placeholder

    # Trials
    priority_map = {"phase_iii": 3, "phase_ii": 2, "phase_i": 1}
    for i, trial in enumerate(state.trials[:MAX_TRIALS]):
        trl_map.add(i, trial.id)
        base = n_loc_feats + i * TRL_FEATURES
        obs[base + 0] = priority_map.get(trial.phase.value, 1)
        obs[base + 1] = 0  # demand placeholder
        obs[base + 2] = 0  # 3-month demand placeholder
        obs[base + 3] = 0  # inventory placeholder
        obs[base + 4] = 30  # days until stockout placeholder

    return obs, loc_map, trl_map


# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_demand(state: SharedState, material_id: UUID, location_id: UUID) -> float:
    """Estimate monthly demand for a material at a location from demand plan."""
    if state.demand_plan is None:
        return 0.0
    matching = [
        sd for sd in state.demand_plan.site_demands
        if sd.finished_good_id == material_id
    ]
    if not matching:
        return 0.0
    return sum(sd.quantity_with_overage for sd in matching) / max(len(set(sd.month for sd in matching)), 1)


def _estimate_lead_time(state: SharedState, material_id: UUID) -> float:
    """Estimate total lead time for a material from material specs."""
    cat = state.materials
    for ds in cat.drug_substances:
        if ds.id == material_id:
            return ds.manufacturing_lead_time_days + ds.qc_release_time_days
    for dp in cat.drug_products:
        if dp.id == material_id:
            return dp.manufacturing_lead_time_days + dp.qc_release_time_days
    for pp in cat.primary_packs:
        if pp.id == material_id:
            return pp.packaging_lead_time_days
    for fg in cat.finished_goods:
        if fg.id == material_id:
            return fg.labeling_lead_time_days
    return 30.0  # default


def _rolling_rate(forecasts: list) -> float:
    """Compute rolling 3-month average enrollment rate."""
    sorted_f = sorted(forecasts, key=lambda f: f.month, reverse=True)
    recent = sorted_f[:3]
    if not recent:
        return 0.0
    return sum(f.forecasted_new_patients for f in recent) / len(recent)


def _collect_all_materials(state: SharedState) -> list[tuple[UUID, dict]]:
    """Collect all materials with inventory and demand info."""
    cat = state.materials
    result = []

    inv_by_mat: dict[UUID, float] = {}
    for pos in (state.inventory_positions or []):
        inv_by_mat[pos.material_id] = inv_by_mat.get(pos.material_id, 0) + pos.on_hand

    for ds in cat.drug_substances:
        result.append((ds.id, {
            "pending_demand": 0,
            "inventory": inv_by_mat.get(ds.id, 0),
            "nominal_batch": ds.batch_size_kg,
        }))
    for dp in cat.drug_products:
        result.append((dp.id, {
            "pending_demand": 0,
            "inventory": inv_by_mat.get(dp.id, 0),
            "nominal_batch": dp.batch_size_units,
        }))
    for pp in cat.primary_packs:
        result.append((pp.id, {
            "pending_demand": 0,
            "inventory": inv_by_mat.get(pp.id, 0),
            "nominal_batch": pp.pack_size,
        }))
    for fg in cat.finished_goods:
        result.append((fg.id, {
            "pending_demand": 0,
            "inventory": inv_by_mat.get(fg.id, 0),
            "nominal_batch": fg.kits_per_patient_visit,
        }))

    return result
