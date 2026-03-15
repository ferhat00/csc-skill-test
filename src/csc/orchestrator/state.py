"""SharedState blackboard: the central data store agents read from and write to."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from csc.models import (
    ChangeoverRule,
    ClinicalSite,
    DemandPlan,
    DepotCapacityPlan,
    EnrollmentForecast,
    EquipmentLine,
    InventoryPosition,
    MaterialCatalog,
    PackingDepot,
    PatientCohort,
    PilotPlant,
    PlantCapacityPlan,
    SupplyPlan,
    TransportLane,
    Trial,
)
from csc.models.trial import Arm


class PortfolioPlan(BaseModel):
    """Output of the Portfolio Review Agent."""

    generated_at: datetime
    ranked_trials: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    synergies: list[dict] = Field(default_factory=list)
    resource_allocations: list[dict] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)


class PipelineEvent(BaseModel):
    """An event in the pipeline execution log."""

    timestamp: datetime
    agent_name: str
    event_type: str  # "started", "tool_call", "completed", "error"
    message: str
    data: dict = Field(default_factory=dict)


class SharedState:
    """Central blackboard that all agents read from and write to.

    Reference data is loaded from synthetic data files.
    Agent outputs are populated as the pipeline runs.
    """

    def __init__(self) -> None:
        # Reference data (loaded from synthetic data)
        self.trials: list[Trial] = []
        self.sites: list[ClinicalSite] = []
        self.depots: list[PackingDepot] = []
        self.plants: list[PilotPlant] = []
        self.transport_lanes: list[TransportLane] = []
        self.materials: MaterialCatalog = MaterialCatalog()
        self.equipment_lines: list[EquipmentLine] = []
        self.changeover_rules: list[ChangeoverRule] = []
        self.enrollment_forecasts: list[EnrollmentForecast] = []
        self.patient_cohorts: list[PatientCohort] = []
        self.inventory_positions: list[InventoryPosition] = []

        # Agent outputs (populated as pipeline runs)
        self.demand_plan: DemandPlan | None = None
        self.portfolio_plan: PortfolioPlan | None = None
        self.supply_plan: SupplyPlan | None = None
        self.depot_capacity_plan: DepotCapacityPlan | None = None
        self.plant_capacity_plan: PlantCapacityPlan | None = None

        # Event log
        self.events: list[PipelineEvent] = []

        # Internal scratch-pads written by tool handlers, read by parse_output
        self._demand_plan_raw: dict | None = None
        self._supply_plan_raw: dict | None = None

        # BDI replanning: capacity constraints injected by pipeline
        self._bdi_capacity_constraints: list[str] | None = None

    def log_event(self, agent_name: str, event_type: str, message: str, data: dict | None = None) -> None:
        self.events.append(
            PipelineEvent(
                timestamp=datetime.now(),
                agent_name=agent_name,
                event_type=event_type,
                message=message,
                data=data or {},
            )
        )

    def get(self, key: str) -> Any:
        return getattr(self, key, None)

    def set(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def load_from_dir(self, data_dir: Path) -> None:
        """Load all reference data from JSON files in the given directory."""
        self.plants = _load_list(data_dir / "plants.json", PilotPlant)
        self.depots = _load_list(data_dir / "depots.json", PackingDepot)
        self.sites = _load_list(data_dir / "sites.json", ClinicalSite)
        self.transport_lanes = _load_list(data_dir / "transport_lanes.json", TransportLane)
        self.equipment_lines = _load_list(data_dir / "equipment_lines.json", EquipmentLine)
        self.changeover_rules = _load_list(data_dir / "changeover_rules.json", ChangeoverRule)
        self.enrollment_forecasts = _load_list(data_dir / "enrollment_forecasts.json", EnrollmentForecast)
        self.patient_cohorts = _load_list(data_dir / "patient_cohorts.json", PatientCohort)
        self.inventory_positions = _load_list(data_dir / "inventory_positions.json", InventoryPosition)
        self.trials = _load_list(data_dir / "trials.json", Trial)

        # Materials are nested, not a list
        materials_path = data_dir / "materials.json"
        if materials_path.exists():
            raw = json.loads(materials_path.read_text())
            self.materials = MaterialCatalog.model_validate(raw)

    def to_snapshot(self) -> dict:
        """Serialize the full state to a JSON-compatible dict."""

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, BaseModel):
                return obj.model_dump(mode="json")
            if isinstance(obj, list):
                return [_serialize(item) for item in obj]
            if isinstance(obj, UUID):
                return str(obj)
            return obj

        return {
            "trials": _serialize(self.trials),
            "sites": _serialize(self.sites),
            "depots": _serialize(self.depots),
            "plants": _serialize(self.plants),
            "materials": _serialize(self.materials),
            "demand_plan": _serialize(self.demand_plan) if self.demand_plan else None,
            "portfolio_plan": _serialize(self.portfolio_plan) if self.portfolio_plan else None,
            "supply_plan": _serialize(self.supply_plan) if self.supply_plan else None,
            "depot_capacity_plan": _serialize(self.depot_capacity_plan) if self.depot_capacity_plan else None,
            "plant_capacity_plan": _serialize(self.plant_capacity_plan) if self.plant_capacity_plan else None,
            "events": _serialize(self.events),
        }


def _load_list(path: Path, model_class: type[BaseModel]) -> list:
    """Load a JSON array file into a list of Pydantic model instances."""
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [model_class.model_validate(item) for item in raw]
