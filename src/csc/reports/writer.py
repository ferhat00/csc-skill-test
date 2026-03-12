"""Report writer: generates CSV and JSON reports from pipeline state."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from csc.orchestrator.state import SharedState


def write_reports(state: SharedState, output_dir: Path, fmt: str = "both") -> list[Path]:
    """Write all reports to the output directory. Returns list of created files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    if fmt in ("json", "both"):
        # Full pipeline snapshot
        path = output_dir / "pipeline_snapshot.json"
        path.write_text(json.dumps(state.to_snapshot(), indent=2, default=str))
        created.append(path)

        # Demand plan
        if state.demand_plan:
            path = output_dir / "demand_plan.json"
            path.write_text(state.demand_plan.model_dump_json(indent=2))
            created.append(path)

        # Portfolio plan
        if state.portfolio_plan:
            path = output_dir / "portfolio_plan.json"
            path.write_text(state.portfolio_plan.model_dump_json(indent=2))
            created.append(path)

        # Supply plan
        if state.supply_plan:
            path = output_dir / "supply_plan.json"
            path.write_text(state.supply_plan.model_dump_json(indent=2))
            created.append(path)

        # Capacity plans
        if state.depot_capacity_plan:
            path = output_dir / "depot_capacity_plan.json"
            path.write_text(state.depot_capacity_plan.model_dump_json(indent=2))
            created.append(path)

        if state.plant_capacity_plan:
            path = output_dir / "plant_capacity_plan.json"
            path.write_text(state.plant_capacity_plan.model_dump_json(indent=2))
            created.append(path)

    if fmt in ("csv", "both"):
        # Demand forecast CSV
        if state.demand_plan and state.demand_plan.site_demands:
            path = output_dir / "demand_forecast.csv"
            _write_csv(
                path,
                ["trial_id", "site_id", "month", "finished_good_id", "quantity_kits", "quantity_with_overage", "safety_stock_kits", "urgency"],
                [
                    {
                        "trial_id": str(sd.trial_id),
                        "site_id": str(sd.site_id),
                        "month": str(sd.month),
                        "finished_good_id": str(sd.finished_good_id),
                        "quantity_kits": sd.quantity_kits,
                        "quantity_with_overage": sd.quantity_with_overage,
                        "safety_stock_kits": sd.safety_stock_kits,
                        "urgency": sd.urgency.value,
                    }
                    for sd in state.demand_plan.site_demands
                ],
            )
            created.append(path)

        # Demand by trial CSV
        if state.demand_plan and state.demand_plan.demand_by_trial:
            path = output_dir / "demand_by_trial.csv"
            _write_csv(
                path,
                ["protocol", "total_kits"],
                [{"protocol": k, "total_kits": v} for k, v in state.demand_plan.demand_by_trial.items()],
            )
            created.append(path)

        # Supply plan batches CSV
        if state.supply_plan and state.supply_plan.batches:
            path = output_dir / "supply_batches.csv"
            _write_csv(
                path,
                ["batch_number", "stage", "material_id", "quantity", "unit", "status", "planned_start", "planned_end", "location_id"],
                [
                    {
                        "batch_number": b.batch_number,
                        "stage": b.stage.value,
                        "material_id": str(b.material_id),
                        "quantity": b.quantity,
                        "unit": b.unit.value,
                        "status": b.status.value,
                        "planned_start": str(b.planned_start),
                        "planned_end": str(b.planned_end),
                        "location_id": str(b.location_id),
                    }
                    for b in state.supply_plan.batches
                ],
            )
            created.append(path)

        # Portfolio priorities CSV
        if state.portfolio_plan and state.portfolio_plan.ranked_trials:
            path = output_dir / "portfolio_priorities.csv"
            _write_csv(
                path,
                ["rank", "protocol", "priority_score", "therapy_area", "phase"],
                [
                    {
                        "rank": t.get("rank", ""),
                        "protocol": t.get("protocol", ""),
                        "priority_score": t.get("priority_score", ""),
                        "therapy_area": t.get("therapy_area", ""),
                        "phase": t.get("phase", ""),
                    }
                    for t in state.portfolio_plan.ranked_trials
                ],
            )
            created.append(path)

        # Events log CSV
        if state.events:
            path = output_dir / "pipeline_events.csv"
            _write_csv(
                path,
                ["timestamp", "agent", "event_type", "message"],
                [
                    {
                        "timestamp": str(e.timestamp),
                        "agent": e.agent_name,
                        "event_type": e.event_type,
                        "message": e.message,
                    }
                    for e in state.events
                ],
            )
            created.append(path)

    return created


def _write_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
