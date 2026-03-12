"""Hardcoded domain-realistic seed constants for synthetic data generation."""

from csc.models.common import (
    FormulationType,
    LineType,
    Region,
    TherapyArea,
    TrialPhase,
    UnitOfMeasure,
)

# ── Countries per region ─────────────────────────────────────────────────────

COUNTRIES_BY_REGION: dict[Region, list[str]] = {
    Region.US: ["US", "CA"],
    Region.EU: ["UK", "DE", "FR", "ES", "IT", "NL"],
    Region.APAC: ["AU", "JP", "KR", "SG"],
}

ALL_COUNTRIES = [c for cs in COUNTRIES_BY_REGION.values() for c in cs]

# ── Cities per country (for clinical sites) ──────────────────────────────────

CITIES_BY_COUNTRY: dict[str, list[str]] = {
    "US": ["Boston", "Houston", "San Francisco", "Chicago", "Philadelphia"],
    "CA": ["Toronto", "Montreal"],
    "UK": ["London", "Manchester"],
    "DE": ["Berlin", "Munich"],
    "FR": ["Paris", "Lyon"],
    "ES": ["Madrid", "Barcelona"],
    "IT": ["Milan", "Rome"],
    "NL": ["Amsterdam"],
    "AU": ["Sydney", "Melbourne"],
    "JP": ["Tokyo", "Osaka"],
    "KR": ["Seoul"],
    "SG": ["Singapore"],
}

# ── Trial configuration templates ────────────────────────────────────────────

TRIAL_TEMPLATES: dict[TherapyArea, dict] = {
    TherapyArea.ONCOLOGY: {
        "prefix": "ONCO",
        "formulation": FormulationType.INJECTABLE,
        "unit_form": UnitOfMeasure.VIALS,
        "dose_mg_range": (100.0, 500.0),
        "doses_per_visit": 1,
        "visits_per_month": 2,
        "treatment_months_range": {
            TrialPhase.PHASE_I: (3, 6),
            TrialPhase.PHASE_II: (6, 12),
            TrialPhase.PHASE_III: (12, 24),
        },
        "enrollment_range": {
            TrialPhase.PHASE_I: (20, 50),
            TrialPhase.PHASE_II: (80, 200),
            TrialPhase.PHASE_III: (300, 800),
        },
        "storage_condition": "2-8C",
    },
    TherapyArea.IMMUNOLOGY: {
        "prefix": "IMMU",
        "formulation": FormulationType.INJECTABLE,
        "unit_form": UnitOfMeasure.VIALS,
        "dose_mg_range": (50.0, 300.0),
        "doses_per_visit": 1,
        "visits_per_month": 1,
        "treatment_months_range": {
            TrialPhase.PHASE_I: (3, 6),
            TrialPhase.PHASE_II: (6, 12),
            TrialPhase.PHASE_III: (12, 18),
        },
        "enrollment_range": {
            TrialPhase.PHASE_I: (15, 40),
            TrialPhase.PHASE_II: (60, 150),
            TrialPhase.PHASE_III: (200, 600),
        },
        "storage_condition": "2-8C",
    },
    TherapyArea.NEUROSCIENCE: {
        "prefix": "NEUR",
        "formulation": FormulationType.TABLET,
        "unit_form": UnitOfMeasure.TABLETS,
        "dose_mg_range": (5.0, 50.0),
        "doses_per_visit": 30,  # monthly supply of daily tablets
        "visits_per_month": 1,
        "treatment_months_range": {
            TrialPhase.PHASE_I: (1, 3),
            TrialPhase.PHASE_II: (3, 6),
            TrialPhase.PHASE_III: (6, 12),
        },
        "enrollment_range": {
            TrialPhase.PHASE_I: (20, 60),
            TrialPhase.PHASE_II: (100, 250),
            TrialPhase.PHASE_III: (400, 1000),
        },
        "storage_condition": "15-25C",
    },
    TherapyArea.RARE_DISEASE: {
        "prefix": "RARE",
        "formulation": FormulationType.LYOPHILIZED,
        "unit_form": UnitOfMeasure.VIALS,
        "dose_mg_range": (10.0, 100.0),
        "doses_per_visit": 1,
        "visits_per_month": 1,
        "treatment_months_range": {
            TrialPhase.PHASE_I: (3, 6),
            TrialPhase.PHASE_II: (6, 12),
            TrialPhase.PHASE_III: (12, 24),
        },
        "enrollment_range": {
            TrialPhase.PHASE_I: (10, 30),
            TrialPhase.PHASE_II: (30, 80),
            TrialPhase.PHASE_III: (60, 200),
        },
        "storage_condition": "-20C",
    },
}

# ── Trial distribution: (therapy_area, phase) ───────────────────────────────

TRIAL_MIX: list[tuple[TherapyArea, TrialPhase]] = [
    # 4 Oncology
    (TherapyArea.ONCOLOGY, TrialPhase.PHASE_I),
    (TherapyArea.ONCOLOGY, TrialPhase.PHASE_II),
    (TherapyArea.ONCOLOGY, TrialPhase.PHASE_III),
    (TherapyArea.ONCOLOGY, TrialPhase.PHASE_II),
    # 4 Immunology
    (TherapyArea.IMMUNOLOGY, TrialPhase.PHASE_I),
    (TherapyArea.IMMUNOLOGY, TrialPhase.PHASE_II),
    (TherapyArea.IMMUNOLOGY, TrialPhase.PHASE_III),
    (TherapyArea.IMMUNOLOGY, TrialPhase.PHASE_III),
    # 4 Neuroscience
    (TherapyArea.NEUROSCIENCE, TrialPhase.PHASE_I),
    (TherapyArea.NEUROSCIENCE, TrialPhase.PHASE_II),
    (TherapyArea.NEUROSCIENCE, TrialPhase.PHASE_III),
    (TherapyArea.NEUROSCIENCE, TrialPhase.PHASE_II),
    # 3 Rare Disease
    (TherapyArea.RARE_DISEASE, TrialPhase.PHASE_I),
    (TherapyArea.RARE_DISEASE, TrialPhase.PHASE_II),
    (TherapyArea.RARE_DISEASE, TrialPhase.PHASE_III),
]

# ── Manufacturing parameters ─────────────────────────────────────────────────

PLANT_CONFIGS = [
    {
        "name": "Springfield Pilot Plant",
        "region": Region.US,
        "country": "US",
        "equipment_lines": 4,
        "annual_capacity_kg": 5000.0,
    },
    {
        "name": "Basel Pilot Plant",
        "region": Region.EU,
        "country": "DE",
        "equipment_lines": 3,
        "annual_capacity_kg": 4000.0,
    },
]

DEPOT_CONFIGS = [
    # Regional depots
    {
        "name": "US Regional Depot",
        "region": Region.US,
        "depot_type": "regional",
        "country": "US",
        "packaging_lines": 3,
        "labeling_lines": 2,
        "supported_languages": ["en"],
        "storage_capacity_pallets": 500,
    },
    {
        "name": "EU Regional Depot",
        "region": Region.EU,
        "depot_type": "regional",
        "country": "NL",
        "packaging_lines": 4,
        "labeling_lines": 3,
        "supported_languages": ["en", "de", "fr", "es", "it", "nl"],
        "storage_capacity_pallets": 600,
    },
    {
        "name": "APAC Regional Depot",
        "region": Region.APAC,
        "depot_type": "regional",
        "country": "SG",
        "packaging_lines": 2,
        "labeling_lines": 2,
        "supported_languages": ["en", "ja", "ko"],
        "storage_capacity_pallets": 400,
    },
    # Local depots
    {
        "name": "US East Local Depot",
        "region": Region.US,
        "depot_type": "local",
        "country": "US",
        "packaging_lines": 1,
        "labeling_lines": 1,
        "supported_languages": ["en"],
        "storage_capacity_pallets": 200,
    },
    {
        "name": "EU Central Local Depot",
        "region": Region.EU,
        "depot_type": "local",
        "country": "DE",
        "packaging_lines": 2,
        "labeling_lines": 2,
        "supported_languages": ["en", "de", "fr"],
        "storage_capacity_pallets": 250,
    },
    {
        "name": "APAC East Local Depot",
        "region": Region.APAC,
        "depot_type": "local",
        "country": "JP",
        "packaging_lines": 1,
        "labeling_lines": 1,
        "supported_languages": ["en", "ja"],
        "storage_capacity_pallets": 150,
    },
]

# ── Supply chain stage parameters ────────────────────────────────────────────

DS_PARAMS = {
    "batch_size_kg_range": (5.0, 50.0),
    "manufacturing_lead_time_days": 45,
    "qc_release_time_days": 21,
    "yield_rate_range": (0.75, 0.92),
    "shelf_life_months": 36,
}

DP_PARAMS = {
    "batch_size_units_range": (500, 5000),
    "ds_per_batch_kg_range": (1.0, 10.0),
    "manufacturing_lead_time_days": 30,
    "qc_release_time_days": 14,
    "yield_rate_range": (0.85, 0.97),
    "shelf_life_months": 24,
}

PP_PARAMS = {
    "pack_size_range": (1, 30),
    "packaging_lead_time_days": 7,
    "shelf_life_months": 24,
}

FG_PARAMS = {
    "labeling_lead_time_days": 5,
    "shelf_life_months": 18,
}

# ── Transport lead times (days) ──────────────────────────────────────────────

TRANSPORT_LEAD_TIMES = {
    "plant_to_depot_same_region": 3,
    "plant_to_depot_cross_region": 10,
    "depot_to_depot": 5,
    "depot_to_site_same_country": 2,
    "depot_to_site_cross_country": 5,
}

# ── Equipment line templates ─────────────────────────────────────────────────

PLANT_LINE_TEMPLATES = [
    {"suffix": "Reactor", "line_type": LineType.REACTOR, "capacity_per_day": 10.0, "unit": UnitOfMeasure.KG},
    {"suffix": "Formulation", "line_type": LineType.FORMULATION, "capacity_per_day": 500.0, "unit": UnitOfMeasure.UNITS},
]

DEPOT_LINE_TEMPLATES = [
    {"suffix": "Packaging", "line_type": LineType.PACKAGING, "capacity_per_day": 1000.0, "unit": UnitOfMeasure.UNITS},
    {"suffix": "Labeling", "line_type": LineType.LABELING, "capacity_per_day": 2000.0, "unit": UnitOfMeasure.KITS},
]

# ── Enrollment rate (patients/site/month) by phase ──────────────────────────

ENROLLMENT_RATES: dict[TrialPhase, tuple[float, float]] = {
    TrialPhase.PHASE_I: (0.5, 1.5),
    TrialPhase.PHASE_II: (1.0, 3.0),
    TrialPhase.PHASE_III: (2.0, 5.0),
}

# ── Overage and safety stock ─────────────────────────────────────────────────

OVERAGE_PCT_BY_PHASE: dict[TrialPhase, float] = {
    TrialPhase.PHASE_I: 0.25,
    TrialPhase.PHASE_II: 0.20,
    TrialPhase.PHASE_III: 0.15,
}

SAFETY_STOCK_MONTHS = 2.0  # months of safety stock at each site

# ── Changeover parameters ────────────────────────────────────────────────────

CHANGEOVER_DAYS_DEFAULT = 3
CHANGEOVER_DAYS_WITH_VALIDATION = 7
