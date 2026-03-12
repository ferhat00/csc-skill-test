"""Tool functions for the Depot Capacity Agent."""

from __future__ import annotations

from csc.models import LineType
from csc.orchestrator.state import SharedState


def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "get_supply_plan_depot_load",
            "description": "Get the packaging and labeling workload at each depot based on the supply plan. Shows what needs to be packed/labeled, quantities, and timelines.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_depot_capacity",
            "description": "Get capacity details for a specific depot or all depots: packaging lines, labeling lines, throughput rates, and current utilization.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "depot_name": {"type": "string", "description": "Name of depot (optional, returns all if omitted)"},
                },
                "required": [],
            },
        },
        {
            "name": "check_labeling_requirements",
            "description": "Check labeling complexity for trials served by each depot: languages needed, country-specific labels, and impact on throughput.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "find_bottlenecks",
            "description": "Identify capacity bottlenecks across all depots. Returns constrained lines and periods.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "build_capacity_plan",
            "description": "Compile the depot capacity assessment into the final plan. Call this when done analyzing.",
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

    def get_supply_plan_depot_load() -> dict:
        if not state.supply_plan:
            return {"error": "No supply plan available"}

        # Count batches per depot
        depot_load: dict[str, dict] = {}
        for batch in state.supply_plan.batches:
            loc = str(batch.location_id)
            depot = next((d for d in state.depots if str(d.id) == loc), None)
            if depot:
                if depot.name not in depot_load:
                    depot_load[depot.name] = {"packaging_batches": 0, "labeling_batches": 0, "total_quantity": 0}
                if batch.stage.value in ("pp",):
                    depot_load[depot.name]["packaging_batches"] += 1
                elif batch.stage.value in ("fg",):
                    depot_load[depot.name]["labeling_batches"] += 1
                depot_load[depot.name]["total_quantity"] += batch.quantity

        # If no batches at depots, estimate from demand
        if not depot_load and state.demand_plan:
            for depot in state.depots:
                depot_load[depot.name] = {
                    "packaging_batches": 0,
                    "labeling_batches": 0,
                    "estimated_kits": 0,
                    "note": "Estimated from demand plan",
                }
                # Count trials using this depot's region
                region_trials = [
                    t for t in state.trials
                    if any(s.region == depot.region for s in state.sites if s.id in t.sites)
                ]
                kit_est = sum(state.demand_plan.demand_by_trial.get(t.protocol_number, 0) for t in region_trials)
                depot_load[depot.name]["estimated_kits"] = kit_est // max(1, len([d for d in state.depots if d.region == depot.region]))

        return {"depot_workload": depot_load}

    def get_depot_capacity(depot_name: str = "") -> dict:
        depots_to_check = state.depots
        if depot_name:
            depots_to_check = [d for d in state.depots if d.name == depot_name]

        result = []
        for depot in depots_to_check:
            lines = [l for l in state.equipment_lines if l.location_id == depot.id]
            pkg_lines = [l for l in lines if l.line_type == LineType.PACKAGING]
            lbl_lines = [l for l in lines if l.line_type == LineType.LABELING]

            monthly_pkg_capacity = sum(l.capacity_per_day * l.available_days_per_month for l in pkg_lines)
            monthly_lbl_capacity = sum(l.capacity_per_day * l.available_days_per_month for l in lbl_lines)

            result.append({
                "name": depot.name,
                "region": depot.region.value,
                "type": depot.depot_type,
                "packaging_lines": len(pkg_lines),
                "labeling_lines": len(lbl_lines),
                "monthly_packaging_capacity_units": monthly_pkg_capacity,
                "monthly_labeling_capacity_kits": monthly_lbl_capacity,
                "supported_languages": depot.supported_languages,
                "storage_pallets": depot.storage_capacity_pallets,
            })

        return {"depots": result}

    def check_labeling_requirements() -> dict:
        requirements = []
        for fg in state.materials.finished_goods:
            depot = next((d for d in state.depots if d.id == fg.depot_id), None)
            requirements.append({
                "finished_good": fg.name,
                "languages": fg.label_languages,
                "language_count": len(fg.label_languages),
                "country_specific": fg.country_specific,
                "target_countries": fg.target_countries,
                "depot": depot.name if depot else "unknown",
                "labeling_days": fg.labeling_lead_time_days,
                "complexity": "high" if len(fg.label_languages) > 3 else "medium" if len(fg.label_languages) > 1 else "low",
            })

        return {"labeling_requirements": requirements}

    def find_bottlenecks() -> dict:
        bottlenecks = []
        for depot in state.depots:
            lines = [l for l in state.equipment_lines if l.location_id == depot.id]
            pkg_lines = [l for l in lines if l.line_type == LineType.PACKAGING]
            lbl_lines = [l for l in lines if l.line_type == LineType.LABELING]

            # Count how many FGs this depot serves
            fgs_at_depot = [fg for fg in state.materials.finished_goods if fg.depot_id == depot.id]
            pps_at_depot = [pp for pp in state.materials.primary_packs if pp.depot_id == depot.id]

            # Simple utilization estimate
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
                products_per_line = len(fgs_at_depot) / len(lbl_lines)
                total_languages = sum(len(fg.label_languages) for fg in fgs_at_depot)
                if total_languages > len(lbl_lines) * 4:
                    bottlenecks.append({
                        "depot": depot.name,
                        "resource": "labeling_lines",
                        "severity": "high",
                        "detail": f"{total_languages} language variants across {len(lbl_lines)} labeling lines",
                    })

        return {"bottlenecks": bottlenecks, "total_bottlenecks": len(bottlenecks)}

    def build_capacity_plan(notes: str = "") -> dict:
        from datetime import datetime

        depot_caps = get_depot_capacity()
        bottleneck_result = find_bottlenecks()

        calendars = []
        for depot_info in depot_caps["depots"]:
            calendars.append({
                "location_name": depot_info["name"],
                "region": depot_info["region"],
                "packaging_capacity": depot_info["monthly_packaging_capacity_units"],
                "labeling_capacity": depot_info["monthly_labeling_capacity_kits"],
                "utilization_pct": 65.0,  # estimated
            })

        feasible = all(b["severity"] != "high" for b in bottleneck_result["bottlenecks"])

        adjustments = []
        for bn in bottleneck_result["bottlenecks"]:
            if bn["severity"] == "high":
                adjustments.append(f"Consider adding capacity at {bn['depot']} for {bn['resource']}: {bn['detail']}")

        return {
            "generated_at": datetime.now().isoformat(),
            "depot_calendars": calendars,
            "feasible": feasible,
            "adjustments": adjustments,
            "bottlenecks": bottleneck_result["bottlenecks"],
            "notes": notes,
        }

    return {
        "get_supply_plan_depot_load": get_supply_plan_depot_load,
        "get_depot_capacity": get_depot_capacity,
        "check_labeling_requirements": check_labeling_requirements,
        "find_bottlenecks": find_bottlenecks,
        "build_capacity_plan": build_capacity_plan,
    }
