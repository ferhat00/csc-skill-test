"""Supply planning models: batches, inventory, orders, supply plans."""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID, uuid4

from pydantic import Field

from .common import (
    BatchStatus,
    CSCBaseModel,
    OrderStatus,
    SupplyChainStage,
    UnitOfMeasure,
)


class Batch(CSCBaseModel):
    """A production batch at a plant or depot."""

    id: UUID = Field(default_factory=uuid4)
    material_id: UUID
    stage: SupplyChainStage
    batch_number: str
    quantity: float
    unit: UnitOfMeasure
    status: BatchStatus = BatchStatus.PLANNED
    planned_start: date
    planned_end: date
    expiry_date: date
    location_id: UUID  # plant or depot UUID


class InventoryPosition(CSCBaseModel):
    """Current inventory snapshot for a material at a location."""

    material_id: UUID
    location_id: UUID
    on_hand: float
    in_transit: float = 0.0
    on_order: float = 0.0  # planned batches not yet started
    allocated: float = 0.0  # committed to specific demand
    available: float = 0.0  # on_hand - allocated
    days_of_supply: float = 0.0


class SupplyOrder(CSCBaseModel):
    """A supply order to move or produce material."""

    id: UUID = Field(default_factory=uuid4)
    material_id: UUID
    stage: SupplyChainStage
    quantity: float
    unit: UnitOfMeasure
    origin_id: UUID
    destination_id: UUID
    required_date: date
    ship_date: date
    status: OrderStatus = OrderStatus.PLANNED


class SupplyPlan(CSCBaseModel):
    """Output of the Supply Review Agent: full supply plan across all stages."""

    generated_at: datetime
    horizon_start: date
    horizon_end: date
    batches: list[Batch] = Field(default_factory=list)
    orders: list[SupplyOrder] = Field(default_factory=list)
    inventory_projections: list[InventoryPosition] = Field(default_factory=list)
    shortfall_alerts: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(
        default_factory=list,
        description="LLM-generated supply reasoning",
    )
