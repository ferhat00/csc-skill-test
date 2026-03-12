"""Tool functions for the Portfolio Review Agent."""

from __future__ import annotations

from csc.orchestrator.state import SharedState


def get_tool_definitions() -> list[dict]:
    return [
        {
            "name": "get_portfolio_overview",
            "description": "Get a comprehensive overview of the entire portfolio: trials, their resource assignments (plants, depots), shared materials, and demand summaries.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "rank_trials_by_priority",
            "description": "Rank all trials by priority using configurable criteria weights. Returns scored and ranked list.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "phase_weight": {"type": "number", "description": "Weight for trial phase (0-1). Phase III = highest."},
                    "therapy_weight": {"type": "number", "description": "Weight for therapy area priority (0-1). Oncology = highest."},
                    "timeline_weight": {"type": "number", "description": "Weight for timeline urgency (0-1). Sooner FSFV = higher."},
                    "enrollment_weight": {"type": "number", "description": "Weight for patient count (0-1). More patients = higher."},
                },
                "required": [],
            },
        },
        {
            "name": "detect_resource_conflicts",
            "description": "Find resource conflicts where multiple trials compete for the same plant, depot, or equipment capacity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource_type": {
                        "type": "string",
                        "enum": ["plant", "depot", "all"],
                        "description": "Type of resource to check. Default: all",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "find_material_synergies",
            "description": "Identify trials that share drug substances or drug products, enabling batch consolidation opportunities.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "suggest_prioritization",
            "description": "Given identified conflicts, suggest priority-based resolution strategies.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "conflict_count": {
                        "type": "integer",
                        "description": "Number of conflicts to resolve (resolves top N by severity)",
                    },
                },
                "required": [],
            },
        },
    ]


def create_tool_handlers(state: SharedState) -> dict:

    def get_portfolio_overview() -> dict:
        overview = []
        for trial in state.trials:
            # Find material chain
            dp_id = trial.arms[0].drug_product_id if trial.arms else None
            dp = state.materials.get_dp(dp_id) if dp_id else None
            ds = state.materials.get_ds(dp.drug_substance_id) if dp else None
            plant_name = None
            if ds:
                plant = next((p for p in state.plants if p.id == ds.plant_id), None)
                plant_name = plant.name if plant else None

            # Find demand from demand plan
            demand_total = 0
            if state.demand_plan:
                demand_total = state.demand_plan.demand_by_trial.get(trial.protocol_number, 0)

            overview.append({
                "protocol": trial.protocol_number,
                "therapy_area": trial.therapy_area.value,
                "phase": trial.phase.value,
                "enrollment": trial.planned_enrollment,
                "sites": len(trial.sites),
                "fsfv": str(trial.fsfv),
                "lslv": str(trial.lslv),
                "plant": plant_name,
                "ds_name": ds.name if ds else None,
                "dp_name": dp.name if dp else None,
                "total_demand_kits": demand_total,
            })

        return {
            "portfolio": overview,
            "total_trials": len(overview),
            "therapy_areas": list(set(t["therapy_area"] for t in overview)),
            "plants_used": list(set(t["plant"] for t in overview if t["plant"])),
        }

    def rank_trials_by_priority(
        phase_weight: float = 0.35,
        therapy_weight: float = 0.25,
        timeline_weight: float = 0.20,
        enrollment_weight: float = 0.20,
    ) -> dict:
        phase_scores = {"phase_iii": 1.0, "phase_ii": 0.6, "phase_i": 0.3}
        therapy_scores = {"oncology": 1.0, "rare_disease": 0.8, "immunology": 0.7, "neuroscience": 0.6}

        ranked = []
        for trial in state.trials:
            phase_score = phase_scores.get(trial.phase.value, 0.5)
            therapy_score = therapy_scores.get(trial.therapy_area.value, 0.5)

            # Timeline urgency: closer FSFV = more urgent
            days_to_fsfv = (trial.fsfv - state.trials[0].fsfv).days if state.trials else 0
            max_days = max((t.fsfv - state.trials[0].fsfv).days for t in state.trials) or 1
            timeline_score = 1.0 - (days_to_fsfv / max_days)

            # Enrollment size (normalized)
            max_enrollment = max(t.planned_enrollment for t in state.trials) or 1
            enrollment_score = trial.planned_enrollment / max_enrollment

            total = (
                phase_weight * phase_score
                + therapy_weight * therapy_score
                + timeline_weight * timeline_score
                + enrollment_weight * enrollment_score
            )

            ranked.append({
                "protocol": trial.protocol_number,
                "therapy_area": trial.therapy_area.value,
                "phase": trial.phase.value,
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

        return {"ranked_trials": ranked}

    def detect_resource_conflicts(resource_type: str = "all") -> dict:
        conflicts = []

        # Group trials by plant
        if resource_type in ("plant", "all"):
            plant_trials: dict[str, list] = {}
            for trial in state.trials:
                dp_id = trial.arms[0].drug_product_id if trial.arms else None
                dp = state.materials.get_dp(dp_id) if dp_id else None
                ds = state.materials.get_ds(dp.drug_substance_id) if dp else None
                if ds:
                    pid = str(ds.plant_id)
                    if pid not in plant_trials:
                        plant_trials[pid] = []
                    plant_trials[pid].append(trial.protocol_number)

            for pid, protocols in plant_trials.items():
                if len(protocols) > 3:  # Conflict if >3 trials share a plant
                    plant = next((p for p in state.plants if str(p.id) == pid), None)
                    conflicts.append({
                        "resource_type": "plant",
                        "resource_name": plant.name if plant else pid,
                        "trials_affected": protocols,
                        "severity": "high" if len(protocols) > 5 else "medium",
                        "description": f"{len(protocols)} trials competing for capacity at {plant.name if plant else 'plant'}",
                    })

        # Group trials by depot region
        if resource_type in ("depot", "all"):
            region_trials: dict[str, list] = {}
            for trial in state.trials:
                for site_id in trial.sites:
                    site = next((s for s in state.sites if s.id == site_id), None)
                    if site:
                        r = site.region.value
                        if r not in region_trials:
                            region_trials[r] = set()
                        region_trials[r].add(trial.protocol_number)

            for region, protocols in region_trials.items():
                if len(protocols) > 4:
                    conflicts.append({
                        "resource_type": "depot_region",
                        "resource_name": f"{region.upper()} depot network",
                        "trials_affected": list(protocols),
                        "severity": "high" if len(protocols) > 8 else "medium",
                        "description": f"{len(protocols)} trials need packaging/labeling in {region.upper()} region",
                    })

        return {"conflicts": conflicts, "total_conflicts": len(conflicts)}

    def find_material_synergies() -> dict:
        # In this synthetic data, each trial has unique materials
        # But we can identify same-plant synergies for campaign scheduling
        synergies = []

        # Group by plant
        plant_materials: dict[str, list] = {}
        for ds in state.materials.drug_substances:
            pid = str(ds.plant_id)
            if pid not in plant_materials:
                plant_materials[pid] = []
            plant_materials[pid].append(ds.name)

        for pid, materials in plant_materials.items():
            if len(materials) > 1:
                plant = next((p for p in state.plants if str(p.id) == pid), None)
                synergies.append({
                    "type": "shared_plant",
                    "resource": plant.name if plant else pid,
                    "materials": materials,
                    "benefit": "Campaign scheduling optimization — batch similar products together to minimize changeover",
                })

        # Check for same-formulation-type products that could share packaging
        from collections import Counter
        form_types = Counter(dp.formulation_type.value for dp in state.materials.drug_products)
        for form_type, count in form_types.items():
            if count > 2:
                synergies.append({
                    "type": "shared_formulation",
                    "formulation": form_type,
                    "count": count,
                    "benefit": f"{count} products use {form_type} formulation — potential for shared packaging equipment",
                })

        return {"synergies": synergies, "total_synergies": len(synergies)}

    def suggest_prioritization(conflict_count: int = 5) -> dict:
        # Get conflicts and rankings
        conflicts_result = detect_resource_conflicts()
        ranking_result = rank_trials_by_priority()

        rank_map = {r["protocol"]: r["rank"] for r in ranking_result["ranked_trials"]}

        suggestions = []
        for conflict in conflicts_result["conflicts"][:conflict_count]:
            affected = conflict["trials_affected"]
            prioritized = sorted(affected, key=lambda p: rank_map.get(p, 999))
            suggestions.append({
                "conflict": conflict["description"],
                "resolution": f"Prioritize in order: {', '.join(prioritized[:3])}. Defer lower-priority trials if capacity is insufficient.",
                "priority_order": prioritized,
            })

        return {"suggestions": suggestions}

    return {
        "get_portfolio_overview": get_portfolio_overview,
        "rank_trials_by_priority": rank_trials_by_priority,
        "detect_resource_conflicts": detect_resource_conflicts,
        "find_material_synergies": find_material_synergies,
        "suggest_prioritization": suggest_prioritization,
    }
