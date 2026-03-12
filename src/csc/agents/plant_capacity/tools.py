"""Tool functions for the Plant Capacity Agent."""

from __future__ import annotations

from csc.models import LineType
from csc.orchestrator.state import SharedState


def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "get_supply_plan_plant_load",
            "description": "Get the manufacturing workload at each plant based on the supply plan: DS and DP batches needed, timelines, and materials.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_plant_capacity",
            "description": "Get capacity details for plants: reactor lines, formulation lines, throughput rates, and availability.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "plant_name": {"type": "string", "description": "Name of plant (optional, returns all if omitted)"},
                },
                "required": [],
            },
        },
        {
            "name": "check_campaign_schedule",
            "description": "Evaluate whether planned batches can fit into plant capacity, accounting for changeover times.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "plant_name": {"type": "string", "description": "Name of plant to check"},
                },
                "required": [],
            },
        },
        {
            "name": "find_bottlenecks",
            "description": "Identify manufacturing capacity bottlenecks across all plants.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "build_capacity_plan",
            "description": "Compile the plant capacity assessment into the final plan.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "notes": {"type": "string", "description": "Key findings and recommendations"},
                },
                "required": [],
            },
        },
    ]


def create_tool_handlers(state: SharedState) -> dict:

    def get_supply_plan_plant_load() -> dict:
        if not state.supply_plan:
            return {"error": "No supply plan available"}

        plant_load: dict[str, dict] = {}
        for batch in state.supply_plan.batches:
            loc = str(batch.location_id)
            plant = next((p for p in state.plants if str(p.id) == loc), None)
            if plant:
                if plant.name not in plant_load:
                    plant_load[plant.name] = {"ds_batches": 0, "dp_batches": 0, "materials": set()}
                if batch.stage.value == "ds":
                    plant_load[plant.name]["ds_batches"] += 1
                elif batch.stage.value == "dp":
                    plant_load[plant.name]["dp_batches"] += 1
                plant_load[plant.name]["materials"].add(batch.batch_number.split("-")[2] if "-" in batch.batch_number else "unknown")

        # Convert sets to lists for JSON
        for name in plant_load:
            plant_load[name]["materials"] = list(plant_load[name]["materials"])

        # If no batches at plants, estimate from materials
        if not plant_load:
            for plant in state.plants:
                ds_at_plant = [ds for ds in state.materials.drug_substances if ds.plant_id == plant.id]
                dp_at_plant = [dp for dp in state.materials.drug_products if dp.plant_id == plant.id]
                plant_load[plant.name] = {
                    "ds_products": len(ds_at_plant),
                    "dp_products": len(dp_at_plant),
                    "ds_names": [ds.name for ds in ds_at_plant],
                    "dp_names": [dp.name for dp in dp_at_plant],
                    "note": "Estimated from material catalog (no batches in supply plan for this plant)",
                }

        return {"plant_workload": plant_load}

    def get_plant_capacity(plant_name: str = "") -> dict:
        plants_to_check = state.plants
        if plant_name:
            plants_to_check = [p for p in state.plants if p.name == plant_name]

        result = []
        for plant in plants_to_check:
            lines = [l for l in state.equipment_lines if l.location_id == plant.id]
            reactors = [l for l in lines if l.line_type == LineType.REACTOR]
            formulation = [l for l in lines if l.line_type == LineType.FORMULATION]

            monthly_reactor_cap = sum(l.capacity_per_day * l.available_days_per_month for l in reactors)
            monthly_form_cap = sum(l.capacity_per_day * l.available_days_per_month for l in formulation)

            # Count how many products this plant serves
            ds_count = len([ds for ds in state.materials.drug_substances if ds.plant_id == plant.id])
            dp_count = len([dp for dp in state.materials.drug_products if dp.plant_id == plant.id])

            result.append({
                "name": plant.name,
                "region": plant.region.value,
                "reactor_lines": len(reactors),
                "formulation_lines": len(formulation),
                "monthly_reactor_capacity_kg": monthly_reactor_cap,
                "monthly_formulation_capacity_units": monthly_form_cap,
                "annual_capacity_kg": plant.annual_capacity_kg,
                "ds_products": ds_count,
                "dp_products": dp_count,
            })

        return {"plants": result}

    def check_campaign_schedule(plant_name: str = "") -> dict:
        plants_to_check = state.plants
        if plant_name:
            plants_to_check = [p for p in state.plants if p.name == plant_name]

        schedules = []
        for plant in plants_to_check:
            ds_at_plant = [ds for ds in state.materials.drug_substances if ds.plant_id == plant.id]
            dp_at_plant = [dp for dp in state.materials.drug_products if dp.plant_id == plant.id]

            lines = [l for l in state.equipment_lines if l.location_id == plant.id]
            reactors = [l for l in lines if l.line_type == LineType.REACTOR]

            # Estimate campaign days needed per product
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

        return {"campaign_schedules": schedules}

    def find_bottlenecks() -> dict:
        bottlenecks = []
        for plant in state.plants:
            lines = [l for l in state.equipment_lines if l.location_id == plant.id]
            reactors = [l for l in lines if l.line_type == LineType.REACTOR]
            formulation = [l for l in lines if l.line_type == LineType.FORMULATION]

            ds_count = len([ds for ds in state.materials.drug_substances if ds.plant_id == plant.id])
            dp_count = len([dp for dp in state.materials.drug_products if dp.plant_id == plant.id])

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

        return {"bottlenecks": bottlenecks, "total_bottlenecks": len(bottlenecks)}

    def build_capacity_plan(notes: str = "") -> dict:
        from datetime import datetime

        plant_caps = get_plant_capacity()
        bottleneck_result = find_bottlenecks()
        schedule_result = check_campaign_schedule()

        calendars = []
        for plant_info in plant_caps["plants"]:
            sched = next((s for s in schedule_result["campaign_schedules"] if s["plant"] == plant_info["name"]), {})
            calendars.append({
                "location_name": plant_info["name"],
                "region": plant_info["region"],
                "reactor_capacity_kg_month": plant_info["monthly_reactor_capacity_kg"],
                "formulation_capacity_units_month": plant_info["monthly_formulation_capacity_units"],
                "utilization_pct": sched.get("estimated_utilization_pct", 0),
                "feasible": sched.get("feasible", True),
            })

        feasible = all(s.get("feasible", True) for s in schedule_result["campaign_schedules"])

        adjustments = []
        for bn in bottleneck_result["bottlenecks"]:
            if bn["severity"] == "high":
                adjustments.append(f"Capacity constraint at {bn['plant']} - {bn['resource']}: {bn['detail']}")

        return {
            "generated_at": datetime.now().isoformat(),
            "plant_calendars": calendars,
            "feasible": feasible,
            "adjustments": adjustments,
            "bottlenecks": bottleneck_result["bottlenecks"],
            "notes": notes,
        }

    return {
        "get_supply_plan_plant_load": get_supply_plan_plant_load,
        "get_plant_capacity": get_plant_capacity,
        "check_campaign_schedule": check_campaign_schedule,
        "find_bottlenecks": find_bottlenecks,
        "build_capacity_plan": build_capacity_plan,
    }
