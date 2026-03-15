"""DemandBDIAgent: translates enrollment forecasts into finished-good demand using BDI reasoning."""

from __future__ import annotations

import math
from datetime import date, datetime

from pydantic import BaseModel
from rich.console import Console

from csc.bdi.base_agent import BaseBDIAgent
from csc.bdi.desires import Desire
from csc.bdi.intentions import Plan
from csc.models import DemandPlan, SiteDemand, Urgency
from csc.models.material import FinishedGood, MaterialCatalog
from csc.models.trial import Trial

console = Console()


class DemandBDIAgent(BaseBDIAgent):

    @property
    def agent_name(self) -> str:
        return "demand_review"

    def get_input_keys(self) -> list[str]:
        return ["trials", "sites", "enrollment_forecasts", "patient_cohorts", "materials"]

    def get_output_key(self) -> str:
        return "demand_plan"

    # ── Belief Initialization ────────────────────────────────────────────────

    def initialize_beliefs(self) -> None:
        """Compute trial-to-FG mapping from materials catalog."""
        trials = self.belief_base.get("trials") or []
        materials: MaterialCatalog = self.belief_base.get("materials") or MaterialCatalog()

        trial_fg_map: dict[str, FinishedGood] = {}
        for trial in trials:
            fg = _find_trial_fg(trial, materials)
            if fg:
                trial_fg_map[trial.protocol_number] = fg

        self.belief_base.set("trial_fg_map", trial_fg_map)
        self.belief_base.set("trial_enrollment", {})
        self.belief_base.set("site_demands", [])
        self.belief_base.set("demand_plan_compiled", False)

        console.print(f"    Derived: {len(trial_fg_map)} trial-to-FG mappings")

    # ── Desires ──────────────────────────────────────────────────────────────

    def define_desires(self) -> None:
        self.goal_hierarchy.desires = [
            Desire(
                name="forecast_all_enrollment",
                priority=1,
                description="All trials have enrollment data grouped by trial",
                is_satisfied=lambda bb: bool(bb.get("trial_enrollment"))
                and len(bb.get("trial_enrollment")) == len(bb.get("trials") or []),
            ),
            Desire(
                name="compute_all_kit_demand",
                priority=2,
                description="Kit demands computed for all trial-site pairs",
                is_satisfied=lambda bb: bool(bb.get("site_demands")),
            ),
            Desire(
                name="compile_demand_plan",
                priority=3,
                description="Final DemandPlan aggregated",
                is_satisfied=lambda bb: bb.get("demand_plan_compiled") is True,
            ),
        ]

    # ── Plans ────────────────────────────────────────────────────────────────

    def define_plans(self) -> None:
        self.plan_library.plans = [
            Plan(
                name="plan_forecast_enrollment",
                goal_name="forecast_all_enrollment",
                context_condition=lambda bb: bool(bb.get("trials")) and bool(bb.get("enrollment_forecasts")),
                body=_plan_forecast_enrollment,
            ),
            Plan(
                name="plan_compute_kit_demand",
                goal_name="compute_all_kit_demand",
                context_condition=lambda bb: bool(bb.get("trial_enrollment")),
                body=_plan_compute_kit_demand,
            ),
            Plan(
                name="plan_aggregate",
                goal_name="compile_demand_plan",
                context_condition=lambda bb: bool(bb.get("site_demands")),
                body=_plan_aggregate,
            ),
        ]

    # ── Output Construction ──────────────────────────────────────────────────

    def build_output(self) -> BaseModel:
        site_demands: list[SiteDemand] = self.belief_base.get("site_demands") or []
        demand_by_trial: dict[str, int] = self.belief_base.get("demand_by_trial") or {}

        all_months = [sd.month for sd in site_demands]
        horizon_start = min(all_months) if all_months else date.today()
        horizon_end = max(all_months) if all_months else date.today()
        total_kit_demand = sum(sd.quantity_with_overage for sd in site_demands)

        console.print(f"      Result: {total_kit_demand:,} total kits across {len(site_demands)} site-demand records")

        return DemandPlan(
            generated_at=datetime.now(),
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            site_demands=site_demands,
            total_kit_demand=total_kit_demand,
            demand_by_trial=demand_by_trial,
            assumptions=[
                "BDI agent: deterministic demand computation",
                "Overage applied per trial configuration",
                "Safety stock: 3mo (Phase I), 2mo (Phase II), 1.5mo (Phase III)",
            ],
        )


# ── Plan Bodies (standalone functions) ────────────────────────────────────────


def _plan_forecast_enrollment(bb, agent: BaseBDIAgent) -> None:
    """Group enrollment forecasts by trial and compute monthly totals."""
    trials = bb.get("trials") or []
    forecasts = bb.get("enrollment_forecasts") or []

    trial_enrollment: dict[str, list[dict]] = {}
    for trial in trials:
        trial_forecasts = [f for f in forecasts if f.trial_id == trial.id]
        monthly: dict[str, dict] = {}
        for f in trial_forecasts:
            key = str(f.month)
            if key not in monthly:
                monthly[key] = {"month": key, "new_patients": 0, "cumulative": 0, "active": 0}
            monthly[key]["new_patients"] += f.forecasted_new_patients
            monthly[key]["cumulative"] = max(monthly[key]["cumulative"], f.cumulative_enrolled)
            monthly[key]["active"] = max(monthly[key]["active"], f.cumulative_active)

        trial_enrollment[trial.protocol_number] = sorted(monthly.values(), key=lambda x: x["month"])

    bb.set("trial_enrollment", trial_enrollment)
    console.print(f"      Result: {len(trial_enrollment)} trials forecasted")


def _plan_compute_kit_demand(bb, agent: BaseBDIAgent) -> None:
    """Compute kit demand per site per month for all trials, with overage and safety stock."""
    trials = bb.get("trials") or []
    forecasts = bb.get("enrollment_forecasts") or []
    trial_fg_map: dict[str, FinishedGood] = bb.get("trial_fg_map") or {}

    site_demands: list[SiteDemand] = []
    demand_by_trial: dict[str, int] = {}

    for trial in trials:
        fg = trial_fg_map.get(trial.protocol_number)
        if not fg:
            continue

        # Calculate kits per active patient per month
        treatment_arm = trial.arms[0] if trial.arms else None
        if not treatment_arm:
            continue
        resupply_visits = [v for v in treatment_arm.visits if v.requires_resupply]
        kits_per_month = max(1, len(resupply_visits) // max(1, trial.treatment_duration_months))

        # Safety stock multiplier based on phase
        safety_months = {"phase_i": 3.0, "phase_ii": 2.0, "phase_iii": 1.5}.get(trial.phase.value, 2.0)

        trial_total = 0
        trial_forecasts = [f for f in forecasts if f.trial_id == trial.id]

        for f in trial_forecasts:
            base_kits = f.cumulative_active * kits_per_month * fg.kits_per_patient_visit
            with_overage = math.ceil(base_kits * (1 + trial.overage_pct))
            safety = math.ceil(base_kits * safety_months * 0.05)  # 5% of demand × safety months

            site_demands.append(SiteDemand(
                trial_id=trial.id,
                site_id=f.site_id,
                month=f.month,
                finished_good_id=fg.id,
                quantity_kits=base_kits,
                quantity_with_overage=with_overage,
                safety_stock_kits=safety,
                urgency=Urgency.ROUTINE,
            ))
            trial_total += with_overage

        demand_by_trial[trial.protocol_number] = trial_total

    bb.set("site_demands", site_demands)
    bb.set("demand_by_trial", demand_by_trial)
    console.print(f"      Result: {len(site_demands)} site-demand records, {sum(demand_by_trial.values()):,} total kits")


def _plan_aggregate(bb, agent: BaseBDIAgent) -> None:
    """Mark the demand plan as compiled."""
    bb.set("demand_plan_compiled", True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_trial_fg(trial: Trial, materials: MaterialCatalog) -> FinishedGood | None:
    """Find the finished good associated with a trial's primary treatment arm."""
    if not trial.arms:
        return None
    dp_id = trial.arms[0].drug_product_id
    for pp in materials.primary_packs:
        if pp.drug_product_id == dp_id:
            for fg in materials.finished_goods:
                if fg.primary_pack_id == pp.id:
                    return fg
    return None
