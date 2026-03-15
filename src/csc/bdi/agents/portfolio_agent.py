"""PortfolioBDIAgent: cross-trial prioritization and conflict detection using BDI reasoning."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from pydantic import BaseModel
from rich.console import Console

from csc.bdi.base_agent import BaseBDIAgent
from csc.bdi.desires import Desire
from csc.bdi.intentions import Plan
from csc.orchestrator.state import PortfolioPlan

console = Console()


class PortfolioBDIAgent(BaseBDIAgent):

    @property
    def agent_name(self) -> str:
        return "portfolio_review"

    def get_input_keys(self) -> list[str]:
        return ["trials", "demand_plan", "materials", "plants", "depots"]

    def get_output_key(self) -> str:
        return "portfolio_plan"

    # ── Belief Initialization ────────────────────────────────────────────────

    def initialize_beliefs(self) -> None:
        self.belief_base.set("ranked_trials", None)
        self.belief_base.set("conflicts", None)
        self.belief_base.set("synergies", None)
        self.belief_base.set("resource_allocations", None)
        self.belief_base.set("portfolio_complete", False)

    # ── Desires ──────────────────────────────────────────────────────────────

    def define_desires(self) -> None:
        self.goal_hierarchy.desires = [
            Desire(
                name="rank_all_trials",
                priority=1,
                description="All trials scored and ranked by priority",
                is_satisfied=lambda bb: bb.get("ranked_trials") is not None,
            ),
            Desire(
                name="detect_all_conflicts",
                priority=2,
                description="Resource conflicts identified",
                is_satisfied=lambda bb: bb.get("conflicts") is not None,
            ),
            Desire(
                name="find_all_synergies",
                priority=3,
                description="Material synergies found",
                is_satisfied=lambda bb: bb.get("synergies") is not None,
            ),
            Desire(
                name="compile_portfolio_plan",
                priority=4,
                description="Final portfolio plan built with resource allocations",
                is_satisfied=lambda bb: bb.get("portfolio_complete") is True,
            ),
        ]

    # ── Plans ────────────────────────────────────────────────────────────────

    def define_plans(self) -> None:
        self.plan_library.plans = [
            Plan(
                name="plan_rank_trials",
                goal_name="rank_all_trials",
                context_condition=lambda bb: bool(bb.get("trials")),
                body=_plan_rank_trials,
            ),
            Plan(
                name="plan_detect_conflicts",
                goal_name="detect_all_conflicts",
                context_condition=lambda bb: bb.get("ranked_trials") is not None,
                body=_plan_detect_conflicts,
            ),
            Plan(
                name="plan_find_synergies",
                goal_name="find_all_synergies",
                context_condition=lambda bb: bb.get("ranked_trials") is not None,
                body=_plan_find_synergies,
            ),
            Plan(
                name="plan_compile_portfolio",
                goal_name="compile_portfolio_plan",
                context_condition=lambda bb: (
                    bb.get("ranked_trials") is not None
                    and bb.get("conflicts") is not None
                    and bb.get("synergies") is not None
                ),
                body=_plan_compile_portfolio,
            ),
        ]

    # ── Output Construction ──────────────────────────────────────────────────

    def build_output(self) -> BaseModel:
        ranked = self.belief_base.get("ranked_trials") or []
        conflicts = self.belief_base.get("conflicts") or []
        synergies = self.belief_base.get("synergies") or []
        allocations = self.belief_base.get("resource_allocations") or []

        console.print(f"      Result: {len(ranked)} trials ranked, {len(conflicts)} conflicts, {len(synergies)} synergies")

        return PortfolioPlan(
            generated_at=datetime.now(),
            ranked_trials=ranked,
            conflicts=conflicts,
            synergies=synergies,
            resource_allocations=allocations,
            reasoning=[
                "BDI agent: deterministic priority scoring",
                "Weights: phase=0.35, therapy=0.25, timeline=0.20, enrollment=0.20",
                "Conflict threshold: >3 trials per plant (medium), >5 (high)",
            ],
        )


# ── Plan Bodies ───────────────────────────────────────────────────────────────


def _plan_rank_trials(bb, agent: BaseBDIAgent) -> None:
    """Rank trials using weighted scoring."""
    trials = bb.get("trials") or []
    demand_plan = bb.get("demand_plan")

    phase_scores = {"phase_iii": 1.0, "phase_ii": 0.6, "phase_i": 0.3}
    therapy_scores = {"oncology": 1.0, "rare_disease": 0.8, "immunology": 0.7, "neuroscience": 0.6}

    phase_weight, therapy_weight, timeline_weight, enrollment_weight = 0.35, 0.25, 0.20, 0.20

    # Compute normalizers
    max_enrollment = max((t.planned_enrollment for t in trials), default=1) or 1
    if len(trials) > 1:
        min_fsfv = min(t.fsfv for t in trials)
        max_days = max((t.fsfv - min_fsfv).days for t in trials) or 1
    else:
        min_fsfv = trials[0].fsfv if trials else None
        max_days = 1

    ranked = []
    for trial in trials:
        phase_score = phase_scores.get(trial.phase.value, 0.5)
        therapy_score = therapy_scores.get(trial.therapy_area.value, 0.5)
        timeline_score = 1.0 - ((trial.fsfv - min_fsfv).days / max_days) if min_fsfv else 0.5
        enrollment_score = trial.planned_enrollment / max_enrollment

        total = (
            phase_weight * phase_score
            + therapy_weight * therapy_score
            + timeline_weight * timeline_score
            + enrollment_weight * enrollment_score
        )

        demand_total = 0
        if demand_plan:
            demand_total = demand_plan.demand_by_trial.get(trial.protocol_number, 0)

        ranked.append({
            "protocol": trial.protocol_number,
            "therapy_area": trial.therapy_area.value,
            "phase": trial.phase.value,
            "planned_enrollment": trial.planned_enrollment,
            "total_demand_kits": demand_total,
            "priority_score": round(total, 3),
            "scores": {
                "phase": round(phase_score, 2),
                "therapy": round(therapy_score, 2),
                "timeline": round(timeline_score, 2),
                "enrollment": round(enrollment_score, 2),
            },
        })

    ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    bb.set("ranked_trials", ranked)
    console.print(f"      Result: {len(ranked)} trials ranked (top: {ranked[0]['protocol'] if ranked else 'none'})")


def _plan_detect_conflicts(bb, agent: BaseBDIAgent) -> None:
    """Detect resource conflicts: trials sharing plants or depot regions."""
    trials = bb.get("trials") or []
    materials = bb.get("materials")
    plants = bb.get("plants") or []
    sites = agent.state.sites

    conflicts = []

    # Group trials by plant
    plant_trials: dict[str, list[str]] = {}
    for trial in trials:
        dp_id = trial.arms[0].drug_product_id if trial.arms else None
        dp = materials.get_dp(dp_id) if dp_id and materials else None
        ds = materials.get_ds(dp.drug_substance_id) if dp else None
        if ds:
            pid = str(ds.plant_id)
            plant_trials.setdefault(pid, []).append(trial.protocol_number)

    for pid, protocols in plant_trials.items():
        if len(protocols) > 3:
            plant = next((p for p in plants if str(p.id) == pid), None)
            conflicts.append({
                "resource_type": "plant",
                "resource_name": plant.name if plant else pid,
                "trials_affected": protocols,
                "severity": "high" if len(protocols) > 5 else "medium",
                "description": f"{len(protocols)} trials competing for capacity at {plant.name if plant else 'plant'}",
            })

    # Group trials by depot region
    region_trials: dict[str, set[str]] = {}
    for trial in trials:
        for site_id in trial.sites:
            site = next((s for s in sites if s.id == site_id), None)
            if site:
                region_trials.setdefault(site.region.value, set()).add(trial.protocol_number)

    for region, protocols in region_trials.items():
        if len(protocols) > 4:
            conflicts.append({
                "resource_type": "depot_region",
                "resource_name": f"{region.upper()} depot network",
                "trials_affected": list(protocols),
                "severity": "high" if len(protocols) > 8 else "medium",
                "description": f"{len(protocols)} trials need packaging/labeling in {region.upper()} region",
            })

    bb.set("conflicts", conflicts)
    console.print(f"      Result: {len(conflicts)} conflicts detected")


def _plan_find_synergies(bb, agent: BaseBDIAgent) -> None:
    """Find material synergies: shared plants, shared formulation types."""
    materials = bb.get("materials")
    plants = bb.get("plants") or []
    synergies = []

    if not materials:
        bb.set("synergies", synergies)
        return

    # Group DS by plant
    plant_materials: dict[str, list[str]] = {}
    for ds in materials.drug_substances:
        pid = str(ds.plant_id)
        plant_materials.setdefault(pid, []).append(ds.name)

    for pid, mat_names in plant_materials.items():
        if len(mat_names) > 1:
            plant = next((p for p in plants if str(p.id) == pid), None)
            synergies.append({
                "type": "shared_plant",
                "resource": plant.name if plant else pid,
                "materials": mat_names,
                "benefit": "Campaign scheduling optimization — batch similar products together to minimize changeover",
            })

    # Shared formulation types
    form_types = Counter(dp.formulation_type.value for dp in materials.drug_products)
    for form_type, count in form_types.items():
        if count > 2:
            synergies.append({
                "type": "shared_formulation",
                "formulation": form_type,
                "count": count,
                "benefit": f"{count} products use {form_type} formulation — potential for shared packaging equipment",
            })

    bb.set("synergies", synergies)
    console.print(f"      Result: {len(synergies)} synergies found")


def _plan_compile_portfolio(bb, agent: BaseBDIAgent) -> None:
    """Build resource allocations based on ranking and conflicts, then finalize."""
    ranked = bb.get("ranked_trials") or []
    conflicts = bb.get("conflicts") or []

    rank_map = {r["protocol"]: r["rank"] for r in ranked}

    allocations = []
    for conflict in conflicts:
        affected = conflict["trials_affected"]
        prioritized = sorted(affected, key=lambda p: rank_map.get(p, 999))
        allocations.append({
            "conflict": conflict["description"],
            "resolution": f"Prioritize: {', '.join(prioritized[:3])}. Defer lower-priority trials if capacity is insufficient.",
            "priority_order": prioritized,
        })

    bb.set("resource_allocations", allocations)
    bb.set("portfolio_complete", True)
    console.print(f"      Result: {len(allocations)} resource allocation suggestions")
