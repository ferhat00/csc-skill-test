"""Tool functions for the Demand Review Agent."""

from __future__ import annotations

import math
from datetime import date
from uuid import UUID

from csc.orchestrator.state import SharedState


def get_tool_definitions() -> list[dict]:
    """Return Anthropic tool definitions for the Demand Review Agent."""
    return [
        {
            "name": "get_trial_summary",
            "description": "Get a summary of all trials including protocol numbers, phases, enrollment targets, timelines, and site counts. Use this first to understand the portfolio.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "forecast_enrollment",
            "description": "Forecast monthly enrollment for a specific trial using S-curve modeling. Returns month-by-month projected new patients and cumulative enrollment.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {
                        "type": "string",
                        "description": "Protocol number of the trial, e.g. 'ONCO-2026-001'",
                    },
                },
                "required": ["trial_protocol"],
            },
        },
        {
            "name": "calculate_kit_demand",
            "description": "Calculate finished good kit demand for a trial at each site for each month, based on enrollment forecast and visit/dosing schedule.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {
                        "type": "string",
                        "description": "Protocol number of the trial",
                    },
                },
                "required": ["trial_protocol"],
            },
        },
        {
            "name": "apply_overage",
            "description": "Apply clinical supply overage percentage to base demand quantities. Returns adjusted quantities with overage.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {
                        "type": "string",
                        "description": "Protocol number of the trial",
                    },
                    "base_demand": {
                        "type": "integer",
                        "description": "Base demand quantity in kits",
                    },
                },
                "required": ["trial_protocol", "base_demand"],
            },
        },
        {
            "name": "compute_safety_stock",
            "description": "Calculate recommended safety stock level for a trial based on lead times, demand variability, and service level targets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {
                        "type": "string",
                        "description": "Protocol number of the trial",
                    },
                    "monthly_demand": {
                        "type": "integer",
                        "description": "Average monthly demand in kits",
                    },
                },
                "required": ["trial_protocol", "monthly_demand"],
            },
        },
        {
            "name": "aggregate_demand",
            "description": "Aggregate all site-level demands into a final DemandPlan. Call this after you have analyzed all trials. Returns the complete demand plan JSON.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "notes": {
                        "type": "string",
                        "description": "Any notes or assumptions to include in the plan",
                    },
                },
                "required": [],
            },
        },
    ]


def create_tool_handlers(state: SharedState) -> dict:
    """Create tool handler functions bound to the shared state."""

    def get_trial_summary() -> dict:
        trials = state.trials
        summaries = []
        for t in trials:
            sites_count = len(t.sites)
            fg = next(
                (fg for fg in state.materials.finished_goods
                 if any(dp.id == next((pp.drug_product_id for pp in state.materials.primary_packs if pp.id == fg.primary_pack_id), None)
                        for dp in state.materials.drug_products
                        if any(arm.drug_product_id == dp.id for arm in t.arms))),
                None,
            )
            summaries.append({
                "protocol": t.protocol_number,
                "name": t.name,
                "therapy_area": t.therapy_area.value,
                "phase": t.phase.value,
                "planned_enrollment": t.planned_enrollment,
                "sites": sites_count,
                "fsfv": str(t.fsfv),
                "lslv": str(t.lslv),
                "enrollment_months": t.enrollment_duration_months,
                "treatment_months": t.treatment_duration_months,
                "overage_pct": t.overage_pct,
                "arms": [{"name": a.name, "ratio": a.allocation_ratio} for a in t.arms],
                "countries": t.countries,
                "finished_good_id": str(fg.id) if fg else None,
            })
        return {"trials": summaries, "total_trials": len(summaries)}

    def forecast_enrollment(trial_protocol: str) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        forecasts = [
            f for f in state.enrollment_forecasts if f.trial_id == trial.id
        ]

        # Group by month
        monthly: dict[str, dict] = {}
        for f in forecasts:
            key = str(f.month)
            if key not in monthly:
                monthly[key] = {"month": key, "new_patients": 0, "cumulative": 0, "active": 0, "sites_enrolling": 0}
            monthly[key]["new_patients"] += f.forecasted_new_patients
            monthly[key]["cumulative"] = max(monthly[key]["cumulative"], f.cumulative_enrolled)
            monthly[key]["active"] = max(monthly[key]["active"], f.cumulative_active)
            monthly[key]["sites_enrolling"] += 1

        months = sorted(monthly.values(), key=lambda x: x["month"])
        return {
            "trial": trial_protocol,
            "planned_enrollment": trial.planned_enrollment,
            "monthly_forecast": months,
            "total_months": len(months),
        }

    def calculate_kit_demand(trial_protocol: str) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        # Find the finished good for this trial
        fg = _find_trial_fg(state, trial)
        if not fg:
            return {"error": f"No finished good found for {trial_protocol}"}

        # Calculate demand per site per month
        site_demands = []
        forecasts = [f for f in state.enrollment_forecasts if f.trial_id == trial.id]

        # Group forecasts by site
        site_groups: dict[str, list] = {}
        for f in forecasts:
            sid = str(f.site_id)
            if sid not in site_groups:
                site_groups[sid] = []
            site_groups[sid].append(f)

        # Calculate visits needing kits per month
        treatment_arm = trial.arms[0]  # primary treatment arm
        resupply_visits = [v for v in treatment_arm.visits if v.requires_resupply]
        kits_per_active_patient_per_month = max(1, len(resupply_visits) // max(1, trial.treatment_duration_months))

        for sid, site_forecasts in site_groups.items():
            for f in sorted(site_forecasts, key=lambda x: x.month):
                active = f.cumulative_active
                kits = active * kits_per_active_patient_per_month * fg.kits_per_patient_visit
                site_demands.append({
                    "site_id": sid,
                    "month": str(f.month),
                    "active_patients": active,
                    "kit_demand": kits,
                    "finished_good_id": str(fg.id),
                })

        total_kits = sum(d["kit_demand"] for d in site_demands)
        return {
            "trial": trial_protocol,
            "kits_per_patient_month": kits_per_active_patient_per_month,
            "site_demands": site_demands[:100],  # limit for context window
            "total_base_demand_kits": total_kits,
            "sites_with_demand": len(site_groups),
            "months_covered": len(set(d["month"] for d in site_demands)),
        }

    def apply_overage(trial_protocol: str, base_demand: int) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        overage_pct = trial.overage_pct
        with_overage = math.ceil(base_demand * (1 + overage_pct))
        return {
            "trial": trial_protocol,
            "base_demand": base_demand,
            "overage_pct": overage_pct,
            "demand_with_overage": with_overage,
            "overage_kits": with_overage - base_demand,
        }

    def compute_safety_stock(trial_protocol: str, monthly_demand: int) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        # Safety stock = 2 months of supply (configurable)
        safety_months = 2.0
        # Higher for Phase I (more uncertainty)
        if trial.phase.value == "phase_i":
            safety_months = 3.0
        elif trial.phase.value == "phase_iii":
            safety_months = 1.5

        safety_stock = math.ceil(monthly_demand * safety_months)
        return {
            "trial": trial_protocol,
            "monthly_demand": monthly_demand,
            "safety_months": safety_months,
            "safety_stock_kits": safety_stock,
            "phase": trial.phase.value,
        }

    def aggregate_demand(notes: str = "") -> dict:
        """Aggregate all enrollment data into a demand plan."""
        from datetime import datetime

        all_demands = []
        demand_by_trial: dict[str, int] = {}

        for trial in state.trials:
            fg = _find_trial_fg(state, trial)
            if not fg:
                continue

            forecasts = [f for f in state.enrollment_forecasts if f.trial_id == trial.id]
            treatment_arm = trial.arms[0]
            resupply_visits = [v for v in treatment_arm.visits if v.requires_resupply]
            kits_per_month = max(1, len(resupply_visits) // max(1, trial.treatment_duration_months))

            trial_total = 0
            for f in forecasts:
                base_kits = f.cumulative_active * kits_per_month * fg.kits_per_patient_visit
                with_overage = math.ceil(base_kits * (1 + trial.overage_pct))
                safety = math.ceil(base_kits * 0.15)  # simplified safety stock

                all_demands.append({
                    "trial_id": str(trial.id),
                    "trial_protocol": trial.protocol_number,
                    "site_id": str(f.site_id),
                    "month": str(f.month),
                    "finished_good_id": str(fg.id),
                    "quantity_kits": base_kits,
                    "quantity_with_overage": with_overage,
                    "safety_stock_kits": safety,
                })
                trial_total += with_overage

            demand_by_trial[trial.protocol_number] = trial_total

        # Determine date range
        all_months = [d["month"] for d in all_demands]
        horizon_start = min(all_months) if all_months else str(date.today())
        horizon_end = max(all_months) if all_months else str(date.today())

        return {
            "generated_at": datetime.now().isoformat(),
            "horizon_start": horizon_start,
            "horizon_end": horizon_end,
            "total_kit_demand": sum(d["quantity_with_overage"] for d in all_demands),
            "demand_by_trial": demand_by_trial,
            "total_site_demand_records": len(all_demands),
            "notes": notes,
        }

    return {
        "get_trial_summary": get_trial_summary,
        "forecast_enrollment": forecast_enrollment,
        "calculate_kit_demand": calculate_kit_demand,
        "apply_overage": apply_overage,
        "compute_safety_stock": compute_safety_stock,
        "aggregate_demand": aggregate_demand,
    }


def _find_trial_fg(state: SharedState, trial) -> object | None:
    """Find the finished good associated with a trial's primary treatment arm."""
    if not trial.arms:
        return None
    dp_id = trial.arms[0].drug_product_id
    # DP -> PP -> FG
    for pp in state.materials.primary_packs:
        if pp.drug_product_id == dp_id:
            for fg in state.materials.finished_goods:
                if fg.primary_pack_id == pp.id:
                    return fg
    return None
