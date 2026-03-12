"""Network topology models: plants, depots, clinical sites, and transport lanes."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from .common import (
    CSCBaseModel,
    Identifier,
    Region,
    SupplyChainStage,
)


class PilotPlant(Identifier):
    """Manufacturing facility for drug substance and/or drug product."""

    region: Region
    capabilities: list[SupplyChainStage] = Field(
        description="Which stages this plant handles (DS, DP)"
    )
    equipment_lines: int
    annual_capacity_kg: float
    country: str


class PackingDepot(Identifier):
    """Facility for primary packaging, secondary packaging, and labeling."""

    region: Region
    depot_type: Literal["regional", "local"]
    packaging_lines: int
    labeling_lines: int
    supported_languages: list[str]
    storage_capacity_pallets: int
    country: str


class ClinicalSite(Identifier):
    """A clinical trial site where patients are enrolled and treated."""

    region: Region
    country: str
    city: str
    assigned_depot_id: UUID = Field(description="Local depot that serves this site")
    max_patients: int


class TransportLane(CSCBaseModel):
    """A shipping route between two locations in the network."""

    origin_id: UUID
    destination_id: UUID
    lane_type: Literal["plant_to_depot", "depot_to_depot", "depot_to_site"]
    lead_time_days: int
    cost_per_unit: float
    temperature_controlled: bool = True
