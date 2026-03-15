"""PlantCapacityBDIAgent: validates manufacturing capacity using BDI reasoning."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel
from rich.console import Console

from csc.bdi.base_agent import BaseBDIAgent
from csc.bdi.desires import Desire
from csc.bdi.intentions import Plan
from csc.models import LineType, PlantCapacityPlan
from csc.models.capacity import CapacityCalendar

console = Console()


class PlantCapacityBDIAgent(BaseBDIAgent):

    @property
    def agent_name(self) -> str:
        return "plant_capacity"

    def get_input_keys(self) -> list[str]:
        return ["supply_plan", "plants", "equipment_lines", "changeover_rules", "materials"]

    def get_output_key(self) -> str:
        return "plant_capacity_plan"

    # ── Belief Initialization ────────────────────────────────────────────────

    def initialize_beliefs(self) -> None:
        self.belief_base.set("plant_load", None)
        self.belief_base.set("plant_capacity", None)
        self.belief_base.set("campaign_schedules", None)
        self.belief_base.set("bottlenecks", None)
        self.belief_base.set("plant_plan_compiled", False)

    # ── Desires ──────────────────────────────────────────────────────────────

    def define_desires(self) -> None:
        self.goal_hierarchy.desires = [
            Desire(
                name="assess_plant_load",
                priority=1,
                description="Workload per plant computed",
                is_satisfied=lambda bb: bb.get("plant_load") is not None,
            ),
            Desire(
                name="assess_plant_capacity",
                priority=2,
                description="Capacity per plant computed",
                is_satisfied=lambda bb: bb.get("plant_capacity") is not None,
            ),
            Desire(
                name="evaluate_campaigns",
                priority=3,
                description="Campaign feasibility evaluated",
                is_satisfied=lambda bb: bb.get("campaign_schedules") is not None,
            ),
            Desire(
                name="identify_bottlenecks",
                priority=4,
                description="Bottlenecks identified",
                is_satisfied=lambda bb: bb.get("bottlenecks") is not None,
            ),
            Desire(
                name="compile_plant_plan",
                priority=5,
                description="Final plant capacity plan built",
                is_satisfied=lambda bb: bb.get("plant_plan_compiled") is True,
            ),
        ]

    # ── Plans ────────────────────────────────────────────────────────────────

    def define_plans(self) -> None:
        self.plan_library.plans = [
            Plan(
                name="plan_plant_load",
                goal_name="assess_plant_load",
                context_condition=lambda bb: bb.has("plants"),
                body=_plan_plant_load,
            ),
            Plan(
                name="plan_plant_capacity",
                goal_name="assess_plant_capacity",
                context_condition=lambda bb: bb.has("plants") and bb.has("equipment_lines"),
                body=_plan_plant_capacity,
            ),
            Plan(
                name="plan_evaluate_campaigns",
                goal_name="evaluate_campaigns",
                context_condition=lambda bb: bb.get("plant_capacity") is not None,
                body=_plan_evaluate_campaigns,
            ),
            Plan(
                name="plan_find_bottlenecks",
                goal_name="identify_bottlenecks",
                context_condition=lambda bb: bb.get("campaign_schedules") is not None,
                body=_plan_find_bottlenecks,
            ),
            Plan(
                name="plan_compile_plant",
                goal_name="compile_plant_plan",
                context_condition=lambda bb: bb.get("bottlenecks") is not None,
                body=_plan_compile_plant,
            ),
        ]

    # ── Output Construction ──────────────────────────────────────────────────

    def build_output(self) -> BaseModel:
        calendars = self.belief_base.get("plant_calendars") or []
        feasible = self.belief_base.get("plant_feasible")
        adjustments = self.belief_base.get("plant_adjustments") or []
        bottlenecks = self.belief_base.get("bottlenecks") or []

        if feasible is None:
            feasible = True

        console.print(f"      Result: feasible={feasible}, {len(bottlenecks)} bottlenecks, {len(adjustments)} adjustments")

        return PlantCapacityPlan(
            generated_at=datetime.now().isoformat(),
            plant_calendars=calendars,
            feasible=feasible,
            adjustments=adjustments,
            reasoning=[
                "BDI agent: deterministic plant capacity assessment",
                f"Bottlenecks detected: {len(bottlenecks)}",
            ],
        )


# ── Plan Bodies ───────────────────────────────────────────────────────────────


def _plan_plant_load(bb, agent: BaseBDIAgent) -> None:
    """Count DS/DP batches per plant from supply plan."""
    supply_plan = bb.get("supply_plan")
    plants = bb.get("plants") or []
    materials = bb.get("materials")

    plant_load: dict[str, dict] = {}

    if supply_plan and supply_plan.batches:
        for batch in supply_plan.batches:
            loc = str(batch.location_id)
            plant = next((p for p in plants if str(p.id) == loc), None)
            if plant:
                if plant.name not in plant_load:
                    plant_load[plant.name] = {"ds_batches": 0, "dp_batches": 0, "materials": set()}
                if batch.stage.value == "ds":
                    plant_load[plant.name]["ds_batches"] += 1
                elif batch.stage.value == "dp":
                    plant_load[plant.name]["dp_batches"] += 1
                plant_load[plant.name]["materials"].add(batch.batch_number.split("-")[2] if "-" in batch.batch_number else "unknown")

        for name in plant_load:
            plant_load[name]["materials"] = list(plant_load[name]["materials"])

    # Fallback: estimate from material catalog
    if not plant_load and materials:
        for plant in plants:
            ds_at_plant = [ds for ds in materials.drug_substances if ds.plant_id == plant.id]
            dp_at_plant = [dp for dp in materials.drug_products if dp.plant_id == plant.id]
            plant_load[plant.name] = {
                "ds_products": len(ds_at_plant),
                "dp_products": len(dp_at_plant),
                "ds_names": [ds.name for ds in ds_at_plant],
                "dp_names": [dp.name for dp in dp_at_plant],
                "note": "Estimated from material catalog",
            }

    bb.set("plant_load", plant_load)
    console.print(f"      Result: workload assessed for {len(plant_load)} plants")


def _plan_plant_capacity(bb, agent: BaseBDIAgent) -> None:
    """Compute capacity per plant from equipment lines."""
    plants = bb.get("plants") or []
    equipment_lines = bb.get("equipment_lines") or []
    materials = bb.get("materials")

    capacity_info = []
    for plant in plants:
        lines = [ln for ln in equipment_lines if ln.location_id == plant.id]
        reactors = [ln for ln in lines if ln.line_type == LineType.REACTOR]
        formulation = [ln for ln in lines if ln.line_type == LineType.FORMULATION]

        monthly_reactor = sum(ln.capacity_per_day * ln.available_days_per_month for ln in reactors)
        monthly_form = sum(ln.capacity_per_day * ln.available_days_per_month for ln in formulation)

        ds_count = len([ds for ds in materials.drug_substances if ds.plant_id == plant.id]) if materials else 0
        dp_count = len([dp for dp in materials.drug_products if dp.plant_id == plant.id]) if materials else 0

        capacity_info.append({
            "name": plant.name,
            "region": plant.region.value,
            "reactor_lines": len(reactors),
            "formulation_lines": len(formulation),
            "monthly_reactor_capacity_kg": monthly_reactor,
            "monthly_formulation_capacity_units": monthly_form,
            "annual_capacity_kg": plant.annual_capacity_kg,
            "ds_products": ds_count,
            "dp_products": dp_count,
        })

    bb.set("plant_capacity", capacity_info)
    console.print(f"      Result: capacity computed for {len(capacity_info)} plants")


def _plan_evaluate_campaigns(bb, agent: BaseBDIAgent) -> None:
    """Evaluate campaign feasibility: total campaign days vs available line-days."""
    plants = bb.get("plants") or []
    equipment_lines = bb.get("equipment_lines") or []
    materials = bb.get("materials")

    schedules = []
    for plant in plants:
        ds_at_plant = [ds for ds in materials.drug_substances if ds.plant_id == plant.id] if materials else []
        dp_at_plant = [dp for dp in materials.drug_products if dp.plant_id == plant.id] if materials else []

        lines = [ln for ln in equipment_lines if ln.location_id == plant.id]
        reactors = [ln for ln in lines if ln.line_type == LineType.REACTOR]

        total_campaign_days = 0
        campaigns = []

        for ds in ds_at_plant:
            days = ds.manufacturing_lead_time_days + ds.qc_release_time_days + 3  # +3 changeover
            total_campaign_days += days
            campaigns.append({
                "material": ds.name,
                "type": "DS",
                "campaign_days": days,
                "yield_rate": ds.yield_rate,
            })

        for dp in dp_at_plant:
            days = dp.manufacturing_lead_time_days + dp.qc_release_time_days + 3
            total_campaign_days += days
            campaigns.append({
                "material": dp.name,
                "type": "DP",
                "campaign_days": days,
                "yield_rate": dp.yield_rate,
            })

        available_days = len(reactors) * 22 * 12  # per year across all lines
        utilization = (total_campaign_days / available_days * 100) if available_days > 0 else 0

        schedules.append({
            "plant": plant.name,
            "total_campaign_days_needed": total_campaign_days,
            "available_line_days_per_year": available_days,
            "estimated_utilization_pct": round(utilization, 1),
            "campaigns": campaigns,
            "feasible": utilization < 85,
        })

    bb.set("campaign_schedules", schedules)
    console.print(f"      Result: {len(schedules)} plants evaluated for campaign feasibility")


def _plan_find_bottlenecks(bb, agent: BaseBDIAgent) -> None:
    """Identify manufacturing capacity bottlenecks."""
    plants = bb.get("plants") or []
    equipment_lines = bb.get("equipment_lines") or []
    materials = bb.get("materials")

    bottlenecks = []
    for plant in plants:
        lines = [ln for ln in equipment_lines if ln.location_id == plant.id]
        reactors = [ln for ln in lines if ln.line_type == LineType.REACTOR]
        formulation = [ln for ln in lines if ln.line_type == LineType.FORMULATION]

        ds_count = len([ds for ds in materials.drug_substances if ds.plant_id == plant.id]) if materials else 0
        dp_count = len([dp for dp in materials.drug_products if dp.plant_id == plant.id]) if materials else 0

        if reactors:
            products_per_reactor = ds_count / len(reactors)
            if products_per_reactor > 3:
                bottlenecks.append({
                    "plant": plant.name,
                    "resource": "reactors",
                    "severity": "high" if products_per_reactor > 5 else "medium",
                    "detail": f"{ds_count} DS products on {len(reactors)} reactors ({products_per_reactor:.1f} per reactor)",
                })

        if formulation:
            products_per_line = dp_count / len(formulation)
            if products_per_line > 3:
                bottlenecks.append({
                    "plant": plant.name,
                    "resource": "formulation_lines",
                    "severity": "high" if products_per_line > 5 else "medium",
                    "detail": f"{dp_count} DP products on {len(formulation)} formulation lines",
                })

    bb.set("bottlenecks", bottlenecks)
    console.print(f"      Result: {len(bottlenecks)} bottlenecks identified")


def _plan_compile_plant(bb, agent: BaseBDIAgent) -> None:
    """Build capacity calendars and determine feasibility."""
    plant_capacity = bb.get("plant_capacity") or []
    campaign_schedules = bb.get("campaign_schedules") or []
    bottlenecks = bb.get("bottlenecks") or []

    calendars = []
    for info in plant_capacity:
        sched = next((s for s in campaign_schedules if s["plant"] == info["name"]), {})
        calendars.append(CapacityCalendar(
            location_id=next((p.id for p in (bb.get("plants") or []) if p.name == info["name"]), None) or __import__("uuid").uuid4(),
            location_name=info["name"],
            month=date.today().replace(day=1),
            utilization_pct=sched.get("estimated_utilization_pct", 0),
        ))

    feasible = all(s.get("feasible", True) for s in campaign_schedules)

    adjustments = []
    for bn in bottlenecks:
        if bn["severity"] == "high":
            adjustments.append(f"Capacity constraint at {bn['plant']} - {bn['resource']}: {bn['detail']}")

    bb.set("plant_calendars", calendars)
    bb.set("plant_feasible", feasible)
    bb.set("plant_adjustments", adjustments)
    bb.set("plant_plan_compiled", True)
    console.print(f"      Result: feasible={feasible}")
