"""Demand forecasting models: enrollment, site demand, demand plans."""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import Field

from .common import CSCBaseModel, Urgency


class EnrollmentForecast(CSCBaseModel):
    """Monthly enrollment forecast for a trial at a specific site."""

    trial_id: UUID
    site_id: UUID
    month: date = Field(description="First day of the month")
    forecasted_new_patients: int
    actual_new_patients: int | None = None  # None if future month
    cumulative_enrolled: int
    cumulative_active: int


class SiteDemand(CSCBaseModel):
    """Monthly demand for a finished good at a specific site."""

    trial_id: UUID
    site_id: UUID
    month: date
    finished_good_id: UUID
    quantity_kits: int  # base demand in kit units
    quantity_with_overage: int  # demand including clinical overage
    safety_stock_kits: int
    urgency: Urgency = Urgency.ROUTINE


class DemandPlan(CSCBaseModel):
    """Output of the Demand Review Agent: aggregated demand across time and sites."""

    generated_at: datetime
    horizon_start: date
    horizon_end: date
    site_demands: list[SiteDemand]
    total_kit_demand: int
    demand_by_trial: dict[str, int] = Field(
        default_factory=dict,
        description="Trial protocol number -> total kits demanded",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="LLM-generated reasoning and assumptions",
    )
