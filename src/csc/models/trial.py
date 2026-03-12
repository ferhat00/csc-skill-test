"""Clinical trial domain models: trials, arms, visit schedules, patients."""

from __future__ import annotations

from datetime import date
from uuid import UUID, uuid4

from pydantic import Field

from .common import (
    CSCBaseModel,
    Identifier,
    TherapyArea,
    TrialPhase,
    UnitOfMeasure,
)


class DosingRegimen(CSCBaseModel):
    """Dosing parameters for a treatment arm."""

    dose_mg: float
    doses_per_visit: int
    unit_form: UnitOfMeasure  # e.g., tablets, vials


class VisitSchedule(CSCBaseModel):
    """A single visit in a treatment arm's schedule."""

    visit_number: int
    day_from_enrollment: int  # day relative to patient enrollment
    dosing: DosingRegimen
    requires_resupply: bool = True  # whether site needs resupply after this visit


class Arm(CSCBaseModel):
    """A treatment arm within a clinical trial."""

    id: UUID = Field(default_factory=uuid4)
    name: str  # e.g., "Treatment", "Placebo", "Active Comparator"
    allocation_ratio: float = Field(
        description="Randomization ratio, e.g., 2.0 for 2:1"
    )
    visits: list[VisitSchedule]
    drug_product_id: UUID  # which drug product this arm uses


class Trial(Identifier):
    """A clinical trial with its full configuration."""

    protocol_number: str  # e.g., "ONCO-2026-001"
    therapy_area: TherapyArea
    phase: TrialPhase
    arms: list[Arm]
    planned_enrollment: int
    sites: list[UUID]  # ClinicalSite IDs
    fsfv: date  # First Subject First Visit
    lslv: date  # Last Subject Last Visit
    enrollment_duration_months: int
    treatment_duration_months: int
    countries: list[str]
    overage_pct: float = Field(
        default=0.20, description="Clinical supply overage percentage"
    )


class PatientCohort(CSCBaseModel):
    """Aggregated patient data per site per trial per arm (not individual patients)."""

    trial_id: UUID
    site_id: UUID
    arm_id: UUID
    enrolled_count: int
    enrollment_date_first: date
    enrollment_date_last: date | None = None
    active_count: int
    completed_count: int = 0
    discontinued_count: int = 0
