"""DepotCapacityBDIAgent: validates packaging/labeling capacity using BDI reasoning."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel
from rich.console import Console

from csc.bdi.base_agent import BaseBDIAgent
from csc.bdi.desires import Desire
from csc.bdi.intentions import Plan
from csc.models import DepotCapacityPlan, LineType
from csc.models.capacity import CapacityCalendar

console = Console()


class DepotCapacityBDIAgent(BaseBDIAgent):

    @property
    def agent_name(self) -> str:
        return "depot_capacity"

    def get_input_keys(self) -> list[str]:
        return ["supply_plan", "depots", "equipment_lines", "changeover_rules", "materials"]

    def get_output_key(self) -> str:
        return "depot_capacity_plan"

    # ── Belief Initialization ────────────────────────────────────────────────

    def initialize_beliefs(self) -> None:
        self.belief_base.set("depot_load", None)
        self.belief_base.set("depot_capacity", None)
        self.belief_base.set("labeling_requirements", None)
        self.belief_base.set("bottlenecks", None)
        self.belief_base.set("depot_plan_compiled", False)

    # ── Desires ──────────────────────────────────────────────────────────────

    def define_desires(self) -> None:
        self.goal_hierarchy.desires = [
            Desire(
                name="assess_depot_load",
                priority=1,
                description="Workload per depot computed",
                is_satisfied=lambda bb: bb.get("depot_load") is not None,
            ),
            Desire(
                name="assess_depot_capacity",
                priority=2,
                description="Capacity per depot computed",
                is_satisfied=lambda bb: bb.get("depot_capacity") is not None,
            ),
            Desire(
                name="assess_labeling",
                priority=3,
                description="Labeling complexity checked",
                is_satisfied=lambda bb: bb.get("labeling_requirements") is not None,
            ),
            Desire(
                name="identify_bottlenecks",
                priority=4,
                description="Bottlenecks found",
                is_satisfied=lambda bb: bb.get("bottlenecks") is not None,
            ),
            Desire(
                name="compile_depot_plan",
                priority=5,
                description="Final depot capacity plan built",
                is_satisfied=lambda bb: bb.get("depot_plan_compiled") is True,
            ),
        ]

    # ── Plans ────────────────────────────────────────────────────────────────

    def define_plans(self) -> None:
        self.plan_library.plans = [
            Plan(
                name="plan_depot_load",
                goal_name="assess_depot_load",
                context_condition=lambda bb: bb.has("depots"),
                body=_plan_depot_load,
            ),
            Plan(
                name="plan_depot_capacity",
                goal_name="assess_depot_capacity",
                context_condition=lambda bb: bb.has("depots") and bb.has("equipment_lines"),
                body=_plan_depot_capacity,
            ),
            Plan(
                name="plan_labeling",
                goal_name="assess_labeling",
                context_condition=lambda bb: bb.has("materials"),
                body=_plan_labeling,
            ),
            Plan(
                name="plan_find_bottlenecks",
                goal_name="identify_bottlenecks",
                context_condition=lambda bb: bb.get("depot_capacity") is not None,
                body=_plan_find_bottlenecks,
            ),
            Plan(
                name="plan_compile_depot",
                goal_name="compile_depot_plan",
                context_condition=lambda bb: bb.get("bottlenecks") is not None,
                body=_plan_compile_depot,
            ),
        ]

    # ── Output Construction ──────────────────────────────────────────────────

    def build_output(self) -> BaseModel:
        calendars = self.belief_base.get("depot_calendars") or []
        feasible = self.belief_base.get("depot_feasible")
        adjustments = self.belief_base.get("depot_adjustments") or []
        bottlenecks = self.belief_base.get("bottlenecks") or []

        if feasible is None:
            feasible = True

        console.print(f"      Result: feasible={feasible}, {len(bottlenecks)} bottlenecks, {len(adjustments)} adjustments")

        return DepotCapacityPlan(
            generated_at=datetime.now().isoformat(),
            depot_calendars=calendars,
            feasible=feasible,
            adjustments=adjustments,
            reasoning=[
                "BDI agent: deterministic depot capacity assessment",
                f"Bottlenecks detected: {len(bottlenecks)}",
            ],
        )


# ── Plan Bodies ───────────────────────────────────────────────────────────────


def _plan_depot_load(bb, agent: BaseBDIAgent) -> None:
    """Count PP/FG batches per depot from supply plan."""
    supply_plan = bb.get("supply_plan")
    depots = bb.get("depots") or []
    demand_plan = agent.state.demand_plan

    depot_load: dict[str, dict] = {}

    if supply_plan and supply_plan.batches:
        for batch in supply_plan.batches:
            loc = str(batch.location_id)
            depot = next((d for d in depots if str(d.id) == loc), None)
            if depot:
                if depot.name not in depot_load:
                    depot_load[depot.name] = {"packaging_batches": 0, "labeling_batches": 0, "total_quantity": 0}
                if batch.stage.value == "pp":
                    depot_load[depot.name]["packaging_batches"] += 1
                elif batch.stage.value == "fg":
                    depot_load[depot.name]["labeling_batches"] += 1
                depot_load[depot.name]["total_quantity"] += batch.quantity

    # Fallback: estimate from demand if no batches at depots
    if not depot_load and demand_plan:
        sites = agent.state.sites
        for depot in depots:
            region_trials = [
                t for t in agent.state.trials
                if any(s.region == depot.region for s in sites if s.id in t.sites)
            ]
            kit_est = sum(demand_plan.demand_by_trial.get(t.protocol_number, 0) for t in region_trials)
            region_depot_count = max(1, len([d for d in depots if d.region == depot.region]))
            depot_load[depot.name] = {
                "packaging_batches": 0,
                "labeling_batches": 0,
                "estimated_kits": kit_est // region_depot_count,
                "note": "Estimated from demand plan",
            }

    bb.set("depot_load", depot_load)
    console.print(f"      Result: workload assessed for {len(depot_load)} depots")


def _plan_depot_capacity(bb, agent: BaseBDIAgent) -> None:
    """Compute capacity per depot from equipment lines."""
    depots = bb.get("depots") or []
    equipment_lines = bb.get("equipment_lines") or []

    capacity_info = []
    for depot in depots:
        lines = [ln for ln in equipment_lines if ln.location_id == depot.id]
        pkg_lines = [ln for ln in lines if ln.line_type == LineType.PACKAGING]
        lbl_lines = [ln for ln in lines if ln.line_type == LineType.LABELING]

        monthly_pkg = sum(ln.capacity_per_day * ln.available_days_per_month for ln in pkg_lines)
        monthly_lbl = sum(ln.capacity_per_day * ln.available_days_per_month for ln in lbl_lines)

        capacity_info.append({
            "name": depot.name,
            "region": depot.region.value,
            "type": depot.depot_type,
            "packaging_lines": len(pkg_lines),
            "labeling_lines": len(lbl_lines),
            "monthly_packaging_capacity": monthly_pkg,
            "monthly_labeling_capacity": monthly_lbl,
            "storage_pallets": depot.storage_capacity_pallets,
        })

    bb.set("depot_capacity", capacity_info)
    console.print(f"      Result: capacity computed for {len(capacity_info)} depots")


def _plan_labeling(bb, agent: BaseBDIAgent) -> None:
    """Assess labeling complexity for finished goods."""
    materials = bb.get("materials")
    depots = bb.get("depots") or []

    requirements = []
    if materials:
        for fg in materials.finished_goods:
            depot = next((d for d in depots if d.id == fg.depot_id), None)
            requirements.append({
                "finished_good": fg.name,
                "languages": fg.label_languages,
                "language_count": len(fg.label_languages),
                "country_specific": fg.country_specific,
                "depot": depot.name if depot else "unknown",
                "complexity": "high" if len(fg.label_languages) > 3 else "medium" if len(fg.label_languages) > 1 else "low",
            })

    bb.set("labeling_requirements", requirements)
    console.print(f"      Result: {len(requirements)} FGs assessed for labeling complexity")


def _plan_find_bottlenecks(bb, agent: BaseBDIAgent) -> None:
    """Identify capacity bottlenecks across all depots."""
    depots = bb.get("depots") or []
    equipment_lines = bb.get("equipment_lines") or []
    materials = bb.get("materials")

    bottlenecks = []
    for depot in depots:
        lines = [ln for ln in equipment_lines if ln.location_id == depot.id]
        pkg_lines = [ln for ln in lines if ln.line_type == LineType.PACKAGING]
        lbl_lines = [ln for ln in lines if ln.line_type == LineType.LABELING]

        fgs_at_depot = [fg for fg in materials.finished_goods if fg.depot_id == depot.id] if materials else []
        pps_at_depot = [pp for pp in materials.primary_packs if pp.depot_id == depot.id] if materials else []

        if pkg_lines:
            products_per_line = len(pps_at_depot) / len(pkg_lines)
            if products_per_line > 3:
                bottlenecks.append({
                    "depot": depot.name,
                    "resource": "packaging_lines",
                    "severity": "high" if products_per_line > 5 else "medium",
                    "detail": f"{len(pps_at_depot)} products on {len(pkg_lines)} packaging lines ({products_per_line:.1f} per line)",
                })

        if lbl_lines:
            total_languages = sum(len(fg.label_languages) for fg in fgs_at_depot)
            if total_languages > len(lbl_lines) * 4:
                bottlenecks.append({
                    "depot": depot.name,
                    "resource": "labeling_lines",
                    "severity": "high",
                    "detail": f"{total_languages} language variants across {len(lbl_lines)} labeling lines",
                })

    bb.set("bottlenecks", bottlenecks)
    console.print(f"      Result: {len(bottlenecks)} bottlenecks identified")


def _plan_compile_depot(bb, agent: BaseBDIAgent) -> None:
    """Build capacity calendars and determine feasibility."""
    depot_capacity = bb.get("depot_capacity") or []
    bottlenecks = bb.get("bottlenecks") or []

    calendars = []
    for info in depot_capacity:
        calendars.append(CapacityCalendar(
            location_id=next((d.id for d in (bb.get("depots") or []) if d.name == info["name"]), None) or __import__("uuid").uuid4(),
            location_name=info["name"],
            month=date.today().replace(day=1),
            utilization_pct=65.0,  # estimated
        ))

    feasible = all(b["severity"] != "high" for b in bottlenecks)

    adjustments = []
    for bn in bottlenecks:
        if bn["severity"] == "high":
            adjustments.append(f"Consider adding capacity at {bn['depot']} for {bn['resource']}: {bn['detail']}")

    bb.set("depot_calendars", calendars)
    bb.set("depot_feasible", feasible)
    bb.set("depot_adjustments", adjustments)
    bb.set("depot_plan_compiled", True)
    console.print(f"      Result: feasible={feasible}")
