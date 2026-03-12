"""Capacity models: equipment lines, campaigns, changeover rules, capacity calendars."""

from __future__ import annotations

from datetime import date
from uuid import UUID, uuid4

from pydantic import Field

from .common import (
    CSCBaseModel,
    CampaignStatus,
    LineType,
    UnitOfMeasure,
)


class EquipmentLine(CSCBaseModel):
    """A production or packaging line at a plant or depot."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    location_id: UUID  # plant or depot
    line_type: LineType
    capacity_per_day: float
    unit: UnitOfMeasure
    available_days_per_month: int = 22  # accounting for maintenance


class Campaign(CSCBaseModel):
    """A scheduled production campaign on an equipment line."""

    id: UUID = Field(default_factory=uuid4)
    line_id: UUID
    material_id: UUID
    start_date: date
    end_date: date
    num_batches: int
    changeover_days_before: int = 0
    status: CampaignStatus = CampaignStatus.TENTATIVE


class ChangeoverRule(CSCBaseModel):
    """Rules for changeover between materials on shared equipment."""

    from_material_id: UUID | None = None  # None = any material
    to_material_id: UUID | None = None
    line_id: UUID
    changeover_days: int
    cleaning_validation_required: bool = False


class CapacityCalendar(CSCBaseModel):
    """Monthly capacity view for a location (plant or depot)."""

    location_id: UUID
    location_name: str
    month: date = Field(description="First day of the month")
    lines: list[EquipmentLine] = Field(default_factory=list)
    campaigns: list[Campaign] = Field(default_factory=list)
    utilization_pct: float = 0.0
    available_slots: int = 0
    bottleneck_lines: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)


class DepotCapacityPlan(CSCBaseModel):
    """Output of the Depot Capacity Agent."""

    generated_at: str  # ISO datetime
    depot_calendars: list[CapacityCalendar] = Field(default_factory=list)
    feasible: bool = True
    adjustments: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)


class PlantCapacityPlan(CSCBaseModel):
    """Output of the Plant Capacity Agent."""

    generated_at: str  # ISO datetime
    plant_calendars: list[CapacityCalendar] = Field(default_factory=list)
    feasible: bool = True
    adjustments: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
