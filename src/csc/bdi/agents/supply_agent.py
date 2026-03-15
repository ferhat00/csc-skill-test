"""SupplyBDIAgent: backward-plans supply across DS->DP->PP->FG stages using BDI reasoning."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel
from rich.console import Console

from csc.bdi.base_agent import BaseBDIAgent
from csc.bdi.desires import Desire
from csc.bdi.intentions import Plan
from csc.models import SupplyPlan
from csc.models.common import BatchStatus, SupplyChainStage, UnitOfMeasure
from csc.models.material import MaterialCatalog
from csc.models.supply import Batch
from csc.models.trial import Trial

console = Console()


class SupplyBDIAgent(BaseBDIAgent):

    @property
    def agent_name(self) -> str:
        return "supply_review"

    def get_input_keys(self) -> list[str]:
        return ["demand_plan", "portfolio_plan", "materials", "inventory_positions", "plants", "depots", "transport_lanes"]

    def get_output_key(self) -> str:
        return "supply_plan"

    # ── Belief Initialization ────────────────────────────────────────────────

    def initialize_beliefs(self) -> None:
        self.belief_base.set("trial_bom", {})
        self.belief_base.set("trial_schedule", {})
        self.belief_base.set("inventory_gaps", {})
        self.belief_base.set("planned_batches", [])
        self.belief_base.set("shortfall_alerts", [])
        self.belief_base.set("supply_plan_compiled", False)

        # Load capacity constraints from replanning (if any)
        constraints = getattr(self.state, "_bdi_capacity_constraints", None)
        if constraints:
            self.belief_base.set("capacity_constraints", constraints, source="upstream_agent")
            console.print(f"    Replanning: {len(constraints)} capacity constraints loaded")

    # ── Desires ──────────────────────────────────────────────────────────────

    def define_desires(self) -> None:
        trials_with_demand = self._get_trials_with_demand()
        num_trials = len(trials_with_demand)

        self.goal_hierarchy.desires = [
            Desire(
                name="explode_all_boms",
                priority=1,
                description="BOM computed for every trial with demand",
                is_satisfied=lambda bb, n=num_trials: len(bb.get("trial_bom") or {}) >= n,
            ),
            Desire(
                name="schedule_all_backward",
                priority=2,
                description="Backward schedules computed for all trials",
                is_satisfied=lambda bb, n=num_trials: len(bb.get("trial_schedule") or {}) >= n,
            ),
            Desire(
                name="check_inventory_gaps",
                priority=3,
                description="Inventory checked and gaps identified",
                is_satisfied=lambda bb, n=num_trials: len(bb.get("inventory_gaps") or {}) >= n,
            ),
            Desire(
                name="schedule_all_batches",
                priority=4,
                description="Production batches created",
                is_satisfied=lambda bb: bool(bb.get("planned_batches")),
            ),
            Desire(
                name="compile_supply_plan",
                priority=5,
                description="Final SupplyPlan assembled",
                is_satisfied=lambda bb: bb.get("supply_plan_compiled") is True,
            ),
        ]

    # ── Plans ────────────────────────────────────────────────────────────────

    def define_plans(self) -> None:
        self.plan_library.plans = [
            Plan(
                name="plan_explode_boms",
                goal_name="explode_all_boms",
                context_condition=lambda bb: bb.get("demand_plan") is not None and bb.has("materials"),
                body=_plan_explode_boms,
            ),
            Plan(
                name="plan_backward_schedule",
                goal_name="schedule_all_backward",
                context_condition=lambda bb: bool(bb.get("trial_bom")),
                body=_plan_backward_schedule,
            ),
            Plan(
                name="plan_check_inventory",
                goal_name="check_inventory_gaps",
                context_condition=lambda bb: bool(bb.get("trial_bom")),
                body=_plan_check_inventory,
            ),
            Plan(
                name="plan_schedule_batches",
                goal_name="schedule_all_batches",
                context_condition=lambda bb: bool(bb.get("trial_schedule")) and bool(bb.get("inventory_gaps")),
                body=_plan_schedule_batches,
            ),
            Plan(
                name="plan_compile_supply",
                goal_name="compile_supply_plan",
                context_condition=lambda bb: bool(bb.get("planned_batches")),
                body=_plan_compile_supply,
            ),
        ]

    # ── Output Construction ──────────────────────────────────────────────────

    def build_output(self) -> BaseModel:
        batches: list[Batch] = self.belief_base.get("planned_batches") or []
        shortfalls: list[str] = self.belief_base.get("shortfall_alerts") or []

        all_dates = []
        for b in batches:
            all_dates.extend([b.planned_start, b.planned_end])

        horizon_start = min(all_dates) if all_dates else date.today()
        horizon_end = max(all_dates) if all_dates else date.today()

        console.print(f"      Result: {len(batches)} batches, {len(shortfalls)} shortfalls")

        reasoning = ["BDI agent: deterministic backward planning with BOM explosion"]
        constraints = self.belief_base.get("capacity_constraints")
        if constraints:
            reasoning.append(f"Replanning with {len(constraints)} capacity constraints")

        return SupplyPlan(
            generated_at=datetime.now(),
            horizon_start=horizon_start,
            horizon_end=horizon_end,
            batches=batches,
            orders=[],
            inventory_projections=[],
            shortfall_alerts=shortfalls,
            reasoning=reasoning,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_trials_with_demand(self) -> list[str]:
        demand_plan = self.belief_base.get("demand_plan")
        if not demand_plan:
            return []
        return [p for p, d in demand_plan.demand_by_trial.items() if d > 0]


# ── Plan Bodies ───────────────────────────────────────────────────────────────


def _plan_explode_boms(bb, agent: BaseBDIAgent) -> None:
    """BOM explosion: FG -> PP -> DP -> DS with yield losses."""
    demand_plan = bb.get("demand_plan")
    materials: MaterialCatalog = bb.get("materials")
    trials = agent.state.trials

    trial_bom: dict[str, dict] = {}

    for protocol, total_kits in demand_plan.demand_by_trial.items():
        if total_kits <= 0:
            continue

        trial = next((t for t in trials if t.protocol_number == protocol), None)
        if not trial:
            continue

        chain = _get_material_chain(trial, materials)
        if not chain:
            continue

        ds, dp, pp, fg = chain

        fg_qty = total_kits
        pp_qty = math.ceil(fg_qty * 1.05)  # 5% packaging loss
        dp_qty = math.ceil(pp_qty * pp.pack_size / dp.yield_rate)
        ds_qty_kg = math.ceil(dp.ds_quantity_per_batch_kg * math.ceil(dp_qty / dp.batch_size_units) * 10) / 10

        ds_batches = math.ceil(ds_qty_kg / ds.batch_size_kg)
        dp_batches = math.ceil(dp_qty / dp.batch_size_units)

        trial_bom[protocol] = {
            "ds": {"material": ds, "qty_kg": ds_qty_kg, "batches": ds_batches},
            "dp": {"material": dp, "qty_units": dp_qty, "batches": dp_batches},
            "pp": {"material": pp, "qty_packs": pp_qty, "batches": 1},
            "fg": {"material": fg, "qty_kits": fg_qty, "batches": 1},
        }

    bb.set("trial_bom", trial_bom)
    console.print(f"      Result: BOM exploded for {len(trial_bom)} trials")


def _plan_backward_schedule(bb, agent: BaseBDIAgent) -> None:
    """Compute backward schedules from demand dates through all stages."""
    demand_plan = bb.get("demand_plan")
    trial_bom = bb.get("trial_bom") or {}
    trials = agent.state.trials

    trial_schedule: dict[str, dict] = {}

    for protocol, bom in trial_bom.items():
        trial = next((t for t in trials if t.protocol_number == protocol), None)
        if not trial:
            continue

        ds = bom["ds"]["material"]
        dp = bom["dp"]["material"]
        pp = bom["pp"]["material"]
        fg = bom["fg"]["material"]

        # Find earliest demand month for this trial
        trial_demands = [sd for sd in demand_plan.site_demands if str(sd.trial_id) == str(trial.id)]
        if not trial_demands:
            continue
        earliest_demand = min(sd.month for sd in trial_demands)
        req = earliest_demand

        # Work backwards
        transport_depot_to_site = 3
        fg_start = req - timedelta(days=transport_depot_to_site + fg.labeling_lead_time_days)
        pp_start = fg_start - timedelta(days=pp.packaging_lead_time_days)
        transport_plant_to_depot = 7
        dp_start = pp_start - timedelta(days=transport_plant_to_depot + dp.manufacturing_lead_time_days + dp.qc_release_time_days)
        ds_start = dp_start - timedelta(days=ds.manufacturing_lead_time_days + ds.qc_release_time_days)

        trial_schedule[protocol] = {
            "required_date": req,
            "ds_start": ds_start,
            "dp_start": dp_start,
            "pp_start": pp_start,
            "fg_start": fg_start,
            "total_lead_days": (req - ds_start).days,
        }

    bb.set("trial_schedule", trial_schedule)
    console.print(f"      Result: backward schedules for {len(trial_schedule)} trials")


def _plan_check_inventory(bb, agent: BaseBDIAgent) -> None:
    """Check current inventory against demand, identify gaps."""
    trial_bom = bb.get("trial_bom") or {}
    inventory_positions = bb.get("inventory_positions") or []

    inventory_gaps: dict[str, dict] = {}

    for protocol, bom in trial_bom.items():
        gaps = {}
        for stage_key in ("ds", "dp", "pp", "fg"):
            material = bom[stage_key]["material"]
            pos = next((ip for ip in inventory_positions if ip.material_id == material.id), None)
            on_hand = pos.on_hand if pos else 0
            available = pos.available if pos else 0

            # Determine demand quantity
            if stage_key == "ds":
                needed = bom[stage_key]["qty_kg"]
            elif stage_key == "dp":
                needed = bom[stage_key]["qty_units"]
            elif stage_key == "pp":
                needed = bom[stage_key]["qty_packs"]
            else:
                needed = bom[stage_key]["qty_kits"]

            gap = max(0, needed - available)
            gaps[stage_key] = {"on_hand": on_hand, "available": available, "needed": needed, "gap": gap}

        inventory_gaps[protocol] = gaps

    bb.set("inventory_gaps", inventory_gaps)
    total_gaps = sum(1 for g in inventory_gaps.values() if any(v["gap"] > 0 for v in g.values()))
    console.print(f"      Result: {total_gaps}/{len(inventory_gaps)} trials have inventory gaps")


def _plan_schedule_batches(bb, agent: BaseBDIAgent) -> None:
    """Create production batches based on BOM, schedule, and inventory gaps."""
    trial_bom = bb.get("trial_bom") or {}
    trial_schedule = bb.get("trial_schedule") or {}
    inventory_gaps = bb.get("inventory_gaps") or {}
    capacity_constraints = bb.get("capacity_constraints")

    planned_batches: list[Batch] = []
    shortfall_alerts: list[str] = []

    for protocol, bom in trial_bom.items():
        schedule = trial_schedule.get(protocol)
        gaps = inventory_gaps.get(protocol, {})
        if not schedule:
            shortfall_alerts.append(f"No schedule for {protocol}")
            continue

        for stage_key, stage_enum, date_key in [
            ("ds", SupplyChainStage.DRUG_SUBSTANCE, "ds_start"),
            ("dp", SupplyChainStage.DRUG_PRODUCT, "dp_start"),
            ("pp", SupplyChainStage.PRIMARY_PACK, "pp_start"),
            ("fg", SupplyChainStage.FINISHED_GOOD, "fg_start"),
        ]:
            material = bom[stage_key]["material"]
            num_batches = bom[stage_key]["batches"]
            gap = gaps.get(stage_key, {}).get("gap", 0)

            # Only schedule batches needed to cover the gap
            if gap <= 0 and stage_key in ("ds", "dp"):
                continue  # inventory covers demand

            start_date = schedule[date_key]
            lead_days = 30  # default batch duration

            if hasattr(material, "manufacturing_lead_time_days"):
                lead_days = material.manufacturing_lead_time_days
            elif hasattr(material, "packaging_lead_time_days"):
                lead_days = material.packaging_lead_time_days
            elif hasattr(material, "labeling_lead_time_days"):
                lead_days = material.labeling_lead_time_days

            # Determine quantity per batch
            if stage_key == "ds":
                qty = material.batch_size_kg
                unit = UnitOfMeasure.KG
            elif stage_key == "dp":
                qty = material.batch_size_units
                unit = material.unit
            elif stage_key == "pp":
                qty = bom[stage_key]["qty_packs"]
                unit = UnitOfMeasure.UNITS
            else:
                qty = bom[stage_key]["qty_kits"]
                unit = UnitOfMeasure.KITS

            location_id = material.plant_id if hasattr(material, "plant_id") else material.depot_id

            for i in range(num_batches):
                batch_start = start_date + timedelta(days=i * 3)  # stagger by 3 days
                batch_end = batch_start + timedelta(days=lead_days)
                expiry = batch_end + timedelta(days=material.shelf_life_months * 30)

                planned_batches.append(Batch(
                    id=uuid4(),
                    material_id=material.id,
                    stage=stage_enum,
                    batch_number=f"B-{stage_key.upper()}-{protocol}-{i+1:02d}",
                    quantity=float(qty),
                    unit=unit,
                    status=BatchStatus.PLANNED,
                    planned_start=batch_start,
                    planned_end=batch_end,
                    expiry_date=expiry,
                    location_id=location_id,
                ))

    if not planned_batches:
        shortfall_alerts.append("No batches scheduled — supply plan may be incomplete")

    bb.set("planned_batches", planned_batches)
    bb.set("shortfall_alerts", shortfall_alerts)
    console.print(f"      Result: {len(planned_batches)} batches scheduled")


def _plan_compile_supply(bb, agent: BaseBDIAgent) -> None:
    """Mark supply plan as compiled."""
    bb.set("supply_plan_compiled", True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_material_chain(trial: Trial, materials: MaterialCatalog):
    """Get the full DS->DP->PP->FG chain for a trial."""
    if not trial.arms:
        return None
    dp_id = trial.arms[0].drug_product_id
    dp = materials.get_dp(dp_id)
    if not dp:
        return None
    ds = materials.get_ds(dp.drug_substance_id)
    pp = next((p for p in materials.primary_packs if p.drug_product_id == dp.id), None)
    if not pp:
        return None
    fg = next((f for f in materials.finished_goods if f.primary_pack_id == pp.id), None)
    if not fg:
        return None
    return (ds, dp, pp, fg)
