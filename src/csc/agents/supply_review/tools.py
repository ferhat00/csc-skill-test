"""Tool functions for the Supply Review Agent."""

from __future__ import annotations

import math
from datetime import date, timedelta
from uuid import uuid4

from csc.orchestrator.state import SharedState


def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "get_demand_summary",
            "description": "Get a summary of the demand plan: total kits needed per trial, timeline, and top demand sites.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "explode_bom",
            "description": "Perform a bill of materials explosion for a finished good. Shows the full chain: FG → PP → DP → DS with quantities needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {"type": "string", "description": "Protocol number of the trial"},
                    "quantity_kits": {"type": "integer", "description": "Number of finished good kits needed"},
                },
                "required": ["trial_protocol", "quantity_kits"],
            },
        },
        {
            "name": "plan_backwards",
            "description": "Calculate backward schedule from a required delivery date through all supply chain stages. Returns the dates each stage must start.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {"type": "string", "description": "Protocol number"},
                    "required_date": {"type": "string", "description": "Date kits needed at site (YYYY-MM-DD)"},
                },
                "required": ["trial_protocol", "required_date"],
            },
        },
        {
            "name": "check_inventory",
            "description": "Check current inventory positions for all materials of a given trial at their respective locations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {"type": "string", "description": "Protocol number"},
                },
                "required": ["trial_protocol"],
            },
        },
        {
            "name": "schedule_batch",
            "description": "Create a planned production batch for a material at a specific location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trial_protocol": {"type": "string", "description": "Protocol number"},
                    "stage": {"type": "string", "enum": ["ds", "dp", "pp", "fg"], "description": "Supply chain stage"},
                    "num_batches": {"type": "integer", "description": "Number of batches to schedule"},
                    "target_date": {"type": "string", "description": "Target completion date (YYYY-MM-DD)"},
                },
                "required": ["trial_protocol", "stage", "num_batches", "target_date"],
            },
        },
        {
            "name": "build_supply_plan",
            "description": "Compile all planned batches and orders into the final supply plan. Call this when you are done planning.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "notes": {"type": "string", "description": "Key decisions and trade-offs"},
                },
                "required": [],
            },
        },
    ]


def create_tool_handlers(state: SharedState) -> dict:
    planned_batches: list[dict] = []
    planned_orders: list[dict] = []

    # Reset any cached plan from a previous run of this agent.
    state._supply_plan_raw = None

    def get_demand_summary() -> dict:
        if not state.demand_plan:
            return {"error": "No demand plan available. Run demand review first."}

        dp = state.demand_plan
        return {
            "total_kit_demand": dp.total_kit_demand,
            "horizon": f"{dp.horizon_start} to {dp.horizon_end}",
            "demand_by_trial": dp.demand_by_trial,
            "assumptions": dp.assumptions,
            "total_site_records": len(dp.site_demands),
        }

    def explode_bom(trial_protocol: str, quantity_kits: int) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        chain = _get_material_chain(state, trial)
        if not chain:
            return {"error": f"Could not find material chain for {trial_protocol}"}

        ds, dp, pp, fg = chain

        # Calculate quantities needed at each stage (working backwards)
        fg_qty = quantity_kits
        pp_qty = math.ceil(fg_qty * 1.05)  # 5% packaging loss
        dp_qty = math.ceil(pp_qty * pp.pack_size / dp.yield_rate)
        ds_qty_kg = math.ceil(dp.ds_quantity_per_batch_kg * math.ceil(dp_qty / dp.batch_size_units) * 10) / 10

        # Batches needed
        ds_batches = math.ceil(ds_qty_kg / ds.batch_size_kg)
        dp_batches = math.ceil(dp_qty / dp.batch_size_units)

        return {
            "trial": trial_protocol,
            "bom": [
                {"stage": "FG", "material": fg.name, "quantity": fg_qty, "unit": "kits", "location": str(fg.depot_id)},
                {"stage": "PP", "material": pp.name, "quantity": pp_qty, "unit": "packs", "location": str(pp.depot_id)},
                {"stage": "DP", "material": dp.name, "quantity": dp_qty, "unit": dp.unit.value, "batches": dp_batches, "location": str(dp.plant_id)},
                {"stage": "DS", "material": ds.name, "quantity_kg": ds_qty_kg, "batches": ds_batches, "location": str(ds.plant_id)},
            ],
        }

    def plan_backwards(trial_protocol: str, required_date: str) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        chain = _get_material_chain(state, trial)
        if not chain:
            return {"error": f"No material chain for {trial_protocol}"}

        ds, dp, pp, fg = chain
        req = date.fromisoformat(required_date)

        # Work backwards
        transport_depot_to_site = 3  # days
        fg_start = req - timedelta(days=transport_depot_to_site + fg.labeling_lead_time_days)
        pp_start = fg_start - timedelta(days=pp.packaging_lead_time_days)

        transport_plant_to_depot = 7  # avg
        dp_start = pp_start - timedelta(days=transport_plant_to_depot + dp.manufacturing_lead_time_days + dp.qc_release_time_days)
        ds_start = dp_start - timedelta(days=ds.manufacturing_lead_time_days + ds.qc_release_time_days)

        total_lead_days = (req - ds_start).days

        return {
            "trial": trial_protocol,
            "required_at_site": required_date,
            "schedule": [
                {"stage": "DS Manufacturing Start", "date": str(ds_start), "lead_days": ds.manufacturing_lead_time_days + ds.qc_release_time_days},
                {"stage": "DP Manufacturing Start", "date": str(dp_start), "lead_days": dp.manufacturing_lead_time_days + dp.qc_release_time_days},
                {"stage": "Transport Plant→Depot", "date": str(pp_start - timedelta(days=transport_plant_to_depot))},
                {"stage": "PP Packaging Start", "date": str(pp_start), "lead_days": pp.packaging_lead_time_days},
                {"stage": "FG Labeling Start", "date": str(fg_start), "lead_days": fg.labeling_lead_time_days},
                {"stage": "Transport Depot→Site", "date": str(req - timedelta(days=transport_depot_to_site))},
                {"stage": "Available at Site", "date": required_date},
            ],
            "total_lead_time_days": total_lead_days,
        }

    def check_inventory(trial_protocol: str) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        chain = _get_material_chain(state, trial)
        if not chain:
            return {"error": f"No material chain for {trial_protocol}"}

        ds, dp, pp, fg = chain
        inventory = []
        for material in [ds, dp, pp, fg]:
            pos = next(
                (ip for ip in state.inventory_positions if ip.material_id == material.id),
                None,
            )
            if pos:
                inventory.append({
                    "material": material.name,
                    "stage": material.stage.value,
                    "on_hand": pos.on_hand,
                    "in_transit": pos.in_transit,
                    "available": pos.available,
                    "days_of_supply": round(pos.days_of_supply, 1),
                })
            else:
                inventory.append({"material": material.name, "stage": material.stage.value, "on_hand": 0, "available": 0})

        return {"trial": trial_protocol, "inventory": inventory}

    def schedule_batch(trial_protocol: str, stage: str, num_batches: int, target_date: str) -> dict:
        trial = next((t for t in state.trials if t.protocol_number == trial_protocol), None)
        if not trial:
            return {"error": f"Trial {trial_protocol} not found"}

        chain = _get_material_chain(state, trial)
        if not chain:
            return {"error": f"No material chain"}

        stage_map = {"ds": chain[0], "dp": chain[1], "pp": chain[2], "fg": chain[3]}
        material = stage_map.get(stage)
        if not material:
            return {"error": f"Invalid stage: {stage}"}

        target = date.fromisoformat(target_date)
        batches = []
        for i in range(num_batches):
            batch_id = str(uuid4())
            batches.append({
                "batch_id": batch_id,
                "material": material.name,
                "material_id": str(material.id),
                "stage": stage,
                "batch_number": f"B-{stage.upper()}-{trial_protocol}-{i+1:02d}",
                "planned_start": str(target - timedelta(days=30)),
                "planned_end": str(target),
                "location_id": str(material.plant_id if hasattr(material, "plant_id") else material.depot_id),
            })
            planned_batches.append(batches[-1])

        return {"scheduled_batches": batches, "trial": trial_protocol, "stage": stage}

    def build_supply_plan(notes: str = "") -> dict:
        from datetime import datetime

        all_dates = []
        for b in planned_batches:
            all_dates.extend([b.get("planned_start", ""), b.get("planned_end", "")])

        valid_dates = [d for d in all_dates if d]
        horizon_start = min(valid_dates) if valid_dates else str(date.today())
        horizon_end = max(valid_dates) if valid_dates else str(date.today())

        shortfalls = []
        if not planned_batches:
            shortfalls.append("No batches were scheduled — supply plan may be incomplete")

        plan = {
            "generated_at": datetime.now().isoformat(),
            "horizon_start": horizon_start,
            "horizon_end": horizon_end,
            "batches": planned_batches,
            "orders": planned_orders,
            "total_batches": len(planned_batches),
            "shortfall_alerts": shortfalls,
            "notes": notes,
        }

        # Persist the compiled plan on shared state so parse_output can use it
        # directly, avoiding the need for the LLM to reproduce all batch data.
        state._supply_plan_raw = plan

        return {
            "status": "supply_plan_compiled",
            "total_batches": len(planned_batches),
            "horizon_start": horizon_start,
            "horizon_end": horizon_end,
            "shortfall_alerts": shortfalls,
            "message": (
                "Supply plan compiled and stored. Output a brief JSON with "
                "horizon_start, horizon_end, shortfall_alerts, and reasoning only."
            ),
        }

    return {
        "get_demand_summary": get_demand_summary,
        "explode_bom": explode_bom,
        "plan_backwards": plan_backwards,
        "check_inventory": check_inventory,
        "schedule_batch": schedule_batch,
        "build_supply_plan": build_supply_plan,
    }


def _get_material_chain(state: SharedState, trial):
    """Get the full DS→DP→PP→FG chain for a trial."""
    if not trial.arms:
        return None
    dp_id = trial.arms[0].drug_product_id
    dp = state.materials.get_dp(dp_id)
    if not dp:
        return None
    ds = state.materials.get_ds(dp.drug_substance_id)
    pp = next((p for p in state.materials.primary_packs if p.drug_product_id == dp.id), None)
    if not pp:
        return None
    fg = next((f for f in state.materials.finished_goods if f.primary_pack_id == pp.id), None)
    if not fg:
        return None
    return (ds, dp, pp, fg)
