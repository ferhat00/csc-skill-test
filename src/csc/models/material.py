"""Material hierarchy: Drug Substance → Drug Product → Primary Pack → Finished Good."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from .common import (
    CSCBaseModel,
    FormulationType,
    Identifier,
    SupplyChainStage,
    UnitOfMeasure,
)


class MaterialBase(Identifier):
    """Base class for all materials in the supply chain."""

    stage: SupplyChainStage
    unit: UnitOfMeasure
    shelf_life_months: int
    storage_condition: str  # e.g., "2-8C", "15-25C", "-20C"


class DrugSubstance(MaterialBase):
    """Active pharmaceutical ingredient (API) produced at a pilot plant."""

    stage: Literal[SupplyChainStage.DRUG_SUBSTANCE] = SupplyChainStage.DRUG_SUBSTANCE
    molecular_weight: float
    batch_size_kg: float
    manufacturing_lead_time_days: int
    qc_release_time_days: int
    yield_rate: float  # e.g., 0.85 = 85%
    plant_id: UUID


class DrugProduct(MaterialBase):
    """Formulated drug product (e.g., tablets, vials) produced at a pilot plant."""

    stage: Literal[SupplyChainStage.DRUG_PRODUCT] = SupplyChainStage.DRUG_PRODUCT
    drug_substance_id: UUID
    formulation_type: FormulationType
    ds_quantity_per_batch_kg: float  # how much DS needed per DP batch
    batch_size_units: int
    manufacturing_lead_time_days: int
    qc_release_time_days: int
    yield_rate: float
    plant_id: UUID


class PrimaryPack(MaterialBase):
    """Primary packaging of drug product (e.g., blistered tablets, labeled vials)."""

    stage: Literal[SupplyChainStage.PRIMARY_PACK] = SupplyChainStage.PRIMARY_PACK
    drug_product_id: UUID
    pack_size: int  # units per primary pack
    packaging_lead_time_days: int
    depot_id: UUID


class FinishedGood(MaterialBase):
    """Secondary-packed and labeled product ready for distribution to sites."""

    stage: Literal[SupplyChainStage.FINISHED_GOOD] = SupplyChainStage.FINISHED_GOOD
    primary_pack_id: UUID
    label_languages: list[str]
    country_specific: bool
    target_countries: list[str]
    labeling_lead_time_days: int
    depot_id: UUID
    kits_per_patient_visit: int = 1


class MaterialCatalog(CSCBaseModel):
    """Consolidated catalog of all materials across the supply chain."""

    drug_substances: list[DrugSubstance] = []
    drug_products: list[DrugProduct] = []
    primary_packs: list[PrimaryPack] = []
    finished_goods: list[FinishedGood] = []

    def get_ds(self, ds_id: UUID) -> DrugSubstance | None:
        return next((ds for ds in self.drug_substances if ds.id == ds_id), None)

    def get_dp(self, dp_id: UUID) -> DrugProduct | None:
        return next((dp for dp in self.drug_products if dp.id == dp_id), None)

    def get_pp(self, pp_id: UUID) -> PrimaryPack | None:
        return next((pp for pp in self.primary_packs if pp.id == pp_id), None)

    def get_fg(self, fg_id: UUID) -> FinishedGood | None:
        return next((fg for fg in self.finished_goods if fg.id == fg_id), None)
