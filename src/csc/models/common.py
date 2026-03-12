"""Shared enums, base classes, and common types used across all models."""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────


class TherapyArea(str, Enum):
    ONCOLOGY = "oncology"
    IMMUNOLOGY = "immunology"
    NEUROSCIENCE = "neuroscience"
    RARE_DISEASE = "rare_disease"


class TrialPhase(str, Enum):
    PHASE_I = "phase_i"
    PHASE_II = "phase_ii"
    PHASE_III = "phase_iii"


class Region(str, Enum):
    US = "us"
    EU = "eu"
    APAC = "apac"


class SupplyChainStage(str, Enum):
    DRUG_SUBSTANCE = "ds"
    DRUG_PRODUCT = "dp"
    PRIMARY_PACK = "pp"
    FINISHED_GOOD = "fg"
    DISTRIBUTION = "distribution"


class UnitOfMeasure(str, Enum):
    KG = "kg"
    L = "liters"
    UNITS = "units"
    VIALS = "vials"
    TABLETS = "tablets"
    KITS = "kits"


class BatchStatus(str, Enum):
    PLANNED = "planned"
    IN_PRODUCTION = "in_production"
    QC_HOLD = "qc_hold"
    RELEASED = "released"
    EXPIRED = "expired"


class FormulationType(str, Enum):
    LYOPHILIZED = "lyophilized"
    LIQUID = "liquid"
    TABLET = "tablet"
    CAPSULE = "capsule"
    INJECTABLE = "injectable"


class Urgency(str, Enum):
    ROUTINE = "routine"
    EXPEDITE = "expedite"
    CRITICAL = "critical"


class OrderStatus(str, Enum):
    PLANNED = "planned"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"


class CampaignStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class LineType(str, Enum):
    REACTOR = "reactor"
    FORMULATION = "formulation"
    PACKAGING = "packaging"
    LABELING = "labeling"


# ── Base Models ──────────────────────────────────────────────────────────────


class CSCBaseModel(BaseModel):
    """Base model with common configuration for all domain models."""

    model_config = {"extra": "forbid"}


class Identifier(CSCBaseModel):
    """Base for all identifiable entities."""

    id: UUID = Field(default_factory=uuid4)
    name: str
