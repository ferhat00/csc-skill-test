"""Master synthetic data generator.

Creates a coherent, self-referencing dataset for the clinical supply chain:
network → materials → trials → enrollment → inventory.
"""

from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

from csc.models import (
    Arm,
    ClinicalSite,
    DosingRegimen,
    DrugProduct,
    DrugSubstance,
    EnrollmentForecast,
    EquipmentLine,
    FinishedGood,
    InventoryPosition,
    MaterialCatalog,
    PackingDepot,
    PatientCohort,
    PilotPlant,
    PrimaryPack,
    Region,
    SupplyChainStage,
    TransportLane,
    Trial,
    TrialPhase,
    UnitOfMeasure,
    VisitSchedule,
    ChangeoverRule,
)
from csc.data.seed import (
    CHANGEOVER_DAYS_DEFAULT,
    CHANGEOVER_DAYS_WITH_VALIDATION,
    CITIES_BY_COUNTRY,
    COUNTRIES_BY_REGION,
    DEPOT_CONFIGS,
    DEPOT_LINE_TEMPLATES,
    DP_PARAMS,
    DS_PARAMS,
    ENROLLMENT_RATES,
    FG_PARAMS,
    OVERAGE_PCT_BY_PHASE,
    PLANT_CONFIGS,
    PLANT_LINE_TEMPLATES,
    PP_PARAMS,
    TRANSPORT_LEAD_TIMES,
    TRIAL_MIX,
    TRIAL_TEMPLATES,
)


class MasterGenerator:
    """Generates a full, coherent synthetic dataset."""

    def __init__(self, seed: int = 42, num_sites: int = 30, base_date: date | None = None):
        self.rng = random.Random(seed)
        self.num_sites = num_sites
        self.base_date = base_date or date(2026, 1, 1)

        # Generated entities (populated by generate())
        self.plants: list[PilotPlant] = []
        self.depots: list[PackingDepot] = []
        self.sites: list[ClinicalSite] = []
        self.transport_lanes: list[TransportLane] = []
        self.equipment_lines: list[EquipmentLine] = []
        self.changeover_rules: list[ChangeoverRule] = []
        self.materials = MaterialCatalog()
        self.trials: list[Trial] = []
        self.enrollment_forecasts: list[EnrollmentForecast] = []
        self.patient_cohorts: list[PatientCohort] = []
        self.inventory_positions: list[InventoryPosition] = []

    def generate(self) -> None:
        """Generate all synthetic data in dependency order."""
        self._generate_plants()
        self._generate_depots()
        self._generate_sites()
        self._generate_transport_lanes()
        self._generate_equipment_lines()
        self._generate_trials_and_materials()
        self._generate_changeover_rules()
        self._generate_enrollment_forecasts()
        self._generate_patient_cohorts()
        self._generate_inventory()

    def save(self, output_dir: Path) -> None:
        """Save all generated data as JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = {
            "plants": [p.model_dump(mode="json") for p in self.plants],
            "depots": [d.model_dump(mode="json") for d in self.depots],
            "sites": [s.model_dump(mode="json") for s in self.sites],
            "transport_lanes": [t.model_dump(mode="json") for t in self.transport_lanes],
            "equipment_lines": [e.model_dump(mode="json") for e in self.equipment_lines],
            "changeover_rules": [c.model_dump(mode="json") for c in self.changeover_rules],
            "materials": self.materials.model_dump(mode="json"),
            "trials": [t.model_dump(mode="json") for t in self.trials],
            "enrollment_forecasts": [e.model_dump(mode="json") for e in self.enrollment_forecasts],
            "patient_cohorts": [p.model_dump(mode="json") for p in self.patient_cohorts],
            "inventory_positions": [i.model_dump(mode="json") for i in self.inventory_positions],
        }

        for name, data in datasets.items():
            path = output_dir / f"{name}.json"
            path.write_text(json.dumps(data, indent=2, default=str))

    # ── Network Generation ───────────────────────────────────────────────

    def _generate_plants(self) -> None:
        for cfg in PLANT_CONFIGS:
            self.plants.append(
                PilotPlant(
                    id=uuid4(),
                    name=cfg["name"],
                    region=cfg["region"],
                    capabilities=[SupplyChainStage.DRUG_SUBSTANCE, SupplyChainStage.DRUG_PRODUCT],
                    equipment_lines=cfg["equipment_lines"],
                    annual_capacity_kg=cfg["annual_capacity_kg"],
                    country=cfg["country"],
                )
            )

    def _generate_depots(self) -> None:
        for cfg in DEPOT_CONFIGS:
            self.depots.append(
                PackingDepot(
                    id=uuid4(),
                    name=cfg["name"],
                    region=cfg["region"],
                    depot_type=cfg["depot_type"],
                    packaging_lines=cfg["packaging_lines"],
                    labeling_lines=cfg["labeling_lines"],
                    supported_languages=cfg["supported_languages"],
                    storage_capacity_pallets=cfg["storage_capacity_pallets"],
                    country=cfg["country"],
                )
            )

    def _generate_sites(self) -> None:
        # Distribute sites across regions roughly equally
        local_depots = [d for d in self.depots if d.depot_type == "local"]
        sites_per_region = self.num_sites // len(local_depots)
        remainder = self.num_sites % len(local_depots)

        for i, depot in enumerate(local_depots):
            n = sites_per_region + (1 if i < remainder else 0)
            region_countries = COUNTRIES_BY_REGION[depot.region]

            for j in range(n):
                country = region_countries[j % len(region_countries)]
                cities = CITIES_BY_COUNTRY.get(country, ["City"])
                city = cities[j % len(cities)]

                self.sites.append(
                    ClinicalSite(
                        id=uuid4(),
                        name=f"{city} Clinical Site {j + 1}",
                        region=depot.region,
                        country=country,
                        city=city,
                        assigned_depot_id=depot.id,
                        max_patients=self.rng.randint(30, 100),
                    )
                )

    def _generate_transport_lanes(self) -> None:
        regional_depots = [d for d in self.depots if d.depot_type == "regional"]
        local_depots = [d for d in self.depots if d.depot_type == "local"]

        # Plant → Regional Depot lanes
        for plant in self.plants:
            for depot in regional_depots:
                same_region = plant.region == depot.region
                lead_time = (
                    TRANSPORT_LEAD_TIMES["plant_to_depot_same_region"]
                    if same_region
                    else TRANSPORT_LEAD_TIMES["plant_to_depot_cross_region"]
                )
                self.transport_lanes.append(
                    TransportLane(
                        origin_id=plant.id,
                        destination_id=depot.id,
                        lane_type="plant_to_depot",
                        lead_time_days=lead_time,
                        cost_per_unit=self.rng.uniform(5.0, 25.0),
                        temperature_controlled=True,
                    )
                )

        # Regional Depot → Local Depot lanes
        for rd in regional_depots:
            for ld in local_depots:
                if rd.region == ld.region:
                    self.transport_lanes.append(
                        TransportLane(
                            origin_id=rd.id,
                            destination_id=ld.id,
                            lane_type="depot_to_depot",
                            lead_time_days=TRANSPORT_LEAD_TIMES["depot_to_depot"],
                            cost_per_unit=self.rng.uniform(3.0, 15.0),
                            temperature_controlled=True,
                        )
                    )

        # Local Depot → Site lanes
        for site in self.sites:
            depot = next(d for d in local_depots if d.id == site.assigned_depot_id)
            same_country = depot.country == site.country
            lead_time = (
                TRANSPORT_LEAD_TIMES["depot_to_site_same_country"]
                if same_country
                else TRANSPORT_LEAD_TIMES["depot_to_site_cross_country"]
            )
            self.transport_lanes.append(
                TransportLane(
                    origin_id=depot.id,
                    destination_id=site.id,
                    lane_type="depot_to_site",
                    lead_time_days=lead_time,
                    cost_per_unit=self.rng.uniform(2.0, 10.0),
                    temperature_controlled=True,
                )
            )

    def _generate_equipment_lines(self) -> None:
        for plant in self.plants:
            for i in range(plant.equipment_lines):
                tmpl = PLANT_LINE_TEMPLATES[i % len(PLANT_LINE_TEMPLATES)]
                self.equipment_lines.append(
                    EquipmentLine(
                        id=uuid4(),
                        name=f"{plant.name} - {tmpl['suffix']} {i + 1}",
                        location_id=plant.id,
                        line_type=tmpl["line_type"],
                        capacity_per_day=tmpl["capacity_per_day"],
                        unit=tmpl["unit"],
                    )
                )

        for depot in self.depots:
            for i in range(depot.packaging_lines):
                tmpl = DEPOT_LINE_TEMPLATES[0]  # packaging
                self.equipment_lines.append(
                    EquipmentLine(
                        id=uuid4(),
                        name=f"{depot.name} - {tmpl['suffix']} {i + 1}",
                        location_id=depot.id,
                        line_type=tmpl["line_type"],
                        capacity_per_day=tmpl["capacity_per_day"],
                        unit=tmpl["unit"],
                    )
                )
            for i in range(depot.labeling_lines):
                tmpl = DEPOT_LINE_TEMPLATES[1]  # labeling
                self.equipment_lines.append(
                    EquipmentLine(
                        id=uuid4(),
                        name=f"{depot.name} - {tmpl['suffix']} {i + 1}",
                        location_id=depot.id,
                        line_type=tmpl["line_type"],
                        capacity_per_day=tmpl["capacity_per_day"],
                        unit=tmpl["unit"],
                    )
                )

    # ── Trials and Materials ─────────────────────────────────────────────

    def _generate_trials_and_materials(self) -> None:
        for idx, (therapy_area, phase) in enumerate(TRIAL_MIX):
            tmpl = TRIAL_TEMPLATES[therapy_area]
            trial_num = idx + 1

            # Pick a plant (alternate between US and EU)
            plant = self.plants[idx % len(self.plants)]

            # Pick depots: regional depot in same region as plant, plus matching local
            regional_depot = next(
                d for d in self.depots if d.depot_type == "regional" and d.region == plant.region
            )
            local_depot = next(
                d for d in self.depots if d.depot_type == "local" and d.region == plant.region
            )

            # Select sites for this trial (2-4 sites per region, up to 3 regions)
            regions_for_trial = self._pick_trial_regions(phase)
            trial_sites = self._pick_trial_sites(regions_for_trial, phase)

            # Generate material chain: DS → DP → PP → FG
            ds, dp, pp, fg = self._generate_material_chain(
                trial_num, therapy_area, phase, tmpl, plant, regional_depot, local_depot, trial_sites
            )

            # Determine enrollment and timeline
            enrollment_range = tmpl["enrollment_range"][phase]
            planned_enrollment = self.rng.randint(*enrollment_range)
            treatment_range = tmpl["treatment_months_range"][phase]
            treatment_months = self.rng.randint(*treatment_range)

            # Calculate enrollment duration based on sites and rates
            rate_range = ENROLLMENT_RATES[phase]
            avg_rate = (rate_range[0] + rate_range[1]) / 2
            enrollment_months = max(3, math.ceil(planned_enrollment / (len(trial_sites) * avg_rate)))

            # Timeline
            fsfv = self.base_date + timedelta(days=self.rng.randint(0, 180))
            lslv = fsfv + timedelta(days=enrollment_months * 30 + treatment_months * 30)

            # Build visit schedule
            visits = self._build_visit_schedule(tmpl, treatment_months)

            # Build arms (treatment + placebo for Phase II/III)
            dp_id = dp.id
            arms = [
                Arm(
                    id=uuid4(),
                    name="Treatment",
                    allocation_ratio=2.0 if phase != TrialPhase.PHASE_I else 1.0,
                    visits=visits,
                    drug_product_id=dp_id,
                )
            ]
            if phase in (TrialPhase.PHASE_II, TrialPhase.PHASE_III):
                arms.append(
                    Arm(
                        id=uuid4(),
                        name="Placebo",
                        allocation_ratio=1.0,
                        visits=visits,
                        drug_product_id=dp_id,  # placebo uses same packaging
                    )
                )

            protocol = f"{tmpl['prefix']}-2026-{trial_num:03d}"
            trial_countries = list({s.country for s in self.sites if s.id in [ts for ts in [site.id for site in trial_sites]]})

            trial = Trial(
                id=uuid4(),
                name=f"{therapy_area.value.title()} {phase.value.replace('_', ' ').title()} Study {trial_num}",
                protocol_number=protocol,
                therapy_area=therapy_area,
                phase=phase,
                arms=arms,
                planned_enrollment=planned_enrollment,
                sites=[s.id for s in trial_sites],
                fsfv=fsfv,
                lslv=lslv,
                enrollment_duration_months=enrollment_months,
                treatment_duration_months=treatment_months,
                countries=trial_countries,
                overage_pct=OVERAGE_PCT_BY_PHASE[phase],
            )
            self.trials.append(trial)

    def _pick_trial_regions(self, phase) -> list[Region]:
        if phase == TrialPhase.PHASE_I:
            return [self.rng.choice(list(Region))]
        elif phase == TrialPhase.PHASE_II:
            regions = list(Region)
            self.rng.shuffle(regions)
            return regions[:2]
        else:
            return list(Region)

    def _pick_trial_sites(self, regions: list[Region], phase) -> list[ClinicalSite]:
        sites_per_region = {
            TrialPhase.PHASE_I: (1, 2),
            TrialPhase.PHASE_II: (2, 4),
            TrialPhase.PHASE_III: (3, 5),
        }[phase]

        selected = []
        for region in regions:
            region_sites = [s for s in self.sites if s.region == region]
            n = min(self.rng.randint(*sites_per_region), len(region_sites))
            selected.extend(self.rng.sample(region_sites, n))
        return selected

    def _generate_material_chain(self, trial_num, therapy_area, phase, tmpl, plant, regional_depot, local_depot, trial_sites):
        dose_mg = round(self.rng.uniform(*tmpl["dose_mg_range"]), 1)

        # Drug Substance
        ds = DrugSubstance(
            id=uuid4(),
            name=f"DS-{tmpl['prefix']}-{trial_num:03d}",
            unit=UnitOfMeasure.KG,
            shelf_life_months=DS_PARAMS["shelf_life_months"],
            storage_condition=tmpl["storage_condition"],
            molecular_weight=round(self.rng.uniform(200, 800), 1),
            batch_size_kg=round(self.rng.uniform(*DS_PARAMS["batch_size_kg_range"]), 1),
            manufacturing_lead_time_days=DS_PARAMS["manufacturing_lead_time_days"],
            qc_release_time_days=DS_PARAMS["qc_release_time_days"],
            yield_rate=round(self.rng.uniform(*DS_PARAMS["yield_rate_range"]), 2),
            plant_id=plant.id,
        )

        # Drug Product
        dp = DrugProduct(
            id=uuid4(),
            name=f"DP-{tmpl['prefix']}-{trial_num:03d}",
            unit=tmpl["unit_form"],
            shelf_life_months=DP_PARAMS["shelf_life_months"],
            storage_condition=tmpl["storage_condition"],
            drug_substance_id=ds.id,
            formulation_type=tmpl["formulation"],
            ds_quantity_per_batch_kg=round(self.rng.uniform(*DP_PARAMS["ds_per_batch_kg_range"]), 1),
            batch_size_units=self.rng.randint(*DP_PARAMS["batch_size_units_range"]),
            manufacturing_lead_time_days=DP_PARAMS["manufacturing_lead_time_days"],
            qc_release_time_days=DP_PARAMS["qc_release_time_days"],
            yield_rate=round(self.rng.uniform(*DP_PARAMS["yield_rate_range"]), 2),
            plant_id=plant.id,
        )

        # Primary Pack
        pack_size = self.rng.randint(*PP_PARAMS["pack_size_range"])
        pp = PrimaryPack(
            id=uuid4(),
            name=f"PP-{tmpl['prefix']}-{trial_num:03d}",
            unit=UnitOfMeasure.UNITS,
            shelf_life_months=PP_PARAMS["shelf_life_months"],
            storage_condition=tmpl["storage_condition"],
            drug_product_id=dp.id,
            pack_size=pack_size,
            packaging_lead_time_days=PP_PARAMS["packaging_lead_time_days"],
            depot_id=regional_depot.id,
        )

        # Finished Good — determine target countries from trial sites
        target_countries = list({s.country for s in trial_sites})
        label_languages = self._languages_for_countries(target_countries)

        fg = FinishedGood(
            id=uuid4(),
            name=f"FG-{tmpl['prefix']}-{trial_num:03d}",
            unit=UnitOfMeasure.KITS,
            shelf_life_months=FG_PARAMS["shelf_life_months"],
            storage_condition=tmpl["storage_condition"],
            primary_pack_id=pp.id,
            label_languages=label_languages,
            country_specific=len(target_countries) > 1,
            target_countries=target_countries,
            labeling_lead_time_days=FG_PARAMS["labeling_lead_time_days"],
            depot_id=regional_depot.id,
            kits_per_patient_visit=1,
        )

        self.materials.drug_substances.append(ds)
        self.materials.drug_products.append(dp)
        self.materials.primary_packs.append(pp)
        self.materials.finished_goods.append(fg)

        return ds, dp, pp, fg

    def _languages_for_countries(self, countries: list[str]) -> list[str]:
        lang_map = {
            "US": "en", "CA": "en", "UK": "en", "AU": "en", "SG": "en",
            "DE": "de", "FR": "fr", "ES": "es", "IT": "it", "NL": "nl",
            "JP": "ja", "KR": "ko",
        }
        return sorted(set(lang_map.get(c, "en") for c in countries))

    def _build_visit_schedule(self, tmpl: dict, treatment_months: int) -> list[VisitSchedule]:
        visits_per_month = tmpl["visits_per_month"]
        total_visits = treatment_months * visits_per_month
        days_between = 30 // visits_per_month

        visits = []
        for v in range(total_visits):
            visits.append(
                VisitSchedule(
                    visit_number=v + 1,
                    day_from_enrollment=v * days_between,
                    dosing=DosingRegimen(
                        dose_mg=self.rng.uniform(*tmpl["dose_mg_range"]),
                        doses_per_visit=tmpl["doses_per_visit"],
                        unit_form=tmpl["unit_form"],
                    ),
                    requires_resupply=(v % visits_per_month == 0),  # resupply monthly
                )
            )
        return visits

    def _generate_changeover_rules(self) -> None:
        # For each equipment line, create default changeover rules
        for line in self.equipment_lines:
            self.changeover_rules.append(
                ChangeoverRule(
                    from_material_id=None,
                    to_material_id=None,
                    line_id=line.id,
                    changeover_days=CHANGEOVER_DAYS_DEFAULT,
                    cleaning_validation_required=False,
                )
            )
        # Add stricter rules for cross-therapy-area changeovers on plant reactors
        reactor_lines = [l for l in self.equipment_lines if l.line_type.value == "reactor"]
        for line in reactor_lines:
            self.changeover_rules.append(
                ChangeoverRule(
                    from_material_id=None,
                    to_material_id=None,
                    line_id=line.id,
                    changeover_days=CHANGEOVER_DAYS_WITH_VALIDATION,
                    cleaning_validation_required=True,
                )
            )

    # ── Enrollment and Patients ──────────────────────────────────────────

    def _generate_enrollment_forecasts(self) -> None:
        for trial in self.trials:
            rate_range = ENROLLMENT_RATES[trial.phase]

            for site_id in trial.sites:
                site_rate = self.rng.uniform(*rate_range)
                cumulative = 0
                month_date = date(trial.fsfv.year, trial.fsfv.month, 1)
                end_date = date(trial.lslv.year, trial.lslv.month, 1)

                while month_date <= end_date and cumulative < trial.planned_enrollment:
                    # S-curve enrollment: slower at start and end
                    progress = cumulative / max(trial.planned_enrollment, 1)
                    s_factor = 4 * progress * (1 - progress) + 0.2  # peaks at 50% enrollment
                    new_patients = max(0, round(site_rate * s_factor + self.rng.gauss(0, 0.5)))

                    # Cap at planned enrollment
                    new_patients = min(new_patients, trial.planned_enrollment - cumulative)
                    cumulative += new_patients

                    # Decide if this month is in the past (actual) or future (forecast)
                    is_past = month_date < self.base_date
                    active = max(0, cumulative - self.rng.randint(0, max(1, cumulative // 10)))

                    self.enrollment_forecasts.append(
                        EnrollmentForecast(
                            trial_id=trial.id,
                            site_id=site_id,
                            month=month_date,
                            forecasted_new_patients=new_patients,
                            actual_new_patients=new_patients if is_past else None,
                            cumulative_enrolled=cumulative,
                            cumulative_active=active,
                        )
                    )

                    # Advance month
                    if month_date.month == 12:
                        month_date = date(month_date.year + 1, 1, 1)
                    else:
                        month_date = date(month_date.year, month_date.month + 1, 1)

    def _generate_patient_cohorts(self) -> None:
        for trial in self.trials:
            for arm in trial.arms:
                for site_id in trial.sites:
                    # Get enrollment data for this site
                    site_forecasts = [
                        f for f in self.enrollment_forecasts
                        if f.trial_id == trial.id and f.site_id == site_id
                    ]
                    if not site_forecasts:
                        continue

                    total_enrolled = site_forecasts[-1].cumulative_enrolled
                    # Split by arm allocation ratio
                    total_ratio = sum(a.allocation_ratio for a in trial.arms)
                    arm_enrolled = round(total_enrolled * arm.allocation_ratio / total_ratio)

                    discontinued = self.rng.randint(0, max(1, arm_enrolled // 8))
                    completed = self.rng.randint(0, max(1, arm_enrolled - discontinued))
                    active = arm_enrolled - completed - discontinued

                    self.patient_cohorts.append(
                        PatientCohort(
                            trial_id=trial.id,
                            site_id=site_id,
                            arm_id=arm.id,
                            enrolled_count=arm_enrolled,
                            enrollment_date_first=trial.fsfv,
                            enrollment_date_last=site_forecasts[-1].month if len(site_forecasts) > 1 else None,
                            active_count=max(0, active),
                            completed_count=max(0, completed),
                            discontinued_count=discontinued,
                        )
                    )

    # ── Inventory ────────────────────────────────────────────────────────

    def _generate_inventory(self) -> None:
        # Generate starting inventory at plants and depots
        for ds in self.materials.drug_substances:
            self.inventory_positions.append(
                InventoryPosition(
                    material_id=ds.id,
                    location_id=ds.plant_id,
                    on_hand=round(ds.batch_size_kg * self.rng.uniform(0.5, 2.0), 1),
                    available=round(ds.batch_size_kg * self.rng.uniform(0.3, 1.5), 1),
                    days_of_supply=self.rng.uniform(30, 90),
                )
            )

        for dp in self.materials.drug_products:
            self.inventory_positions.append(
                InventoryPosition(
                    material_id=dp.id,
                    location_id=dp.plant_id,
                    on_hand=round(dp.batch_size_units * self.rng.uniform(0.3, 1.5)),
                    available=round(dp.batch_size_units * self.rng.uniform(0.2, 1.0)),
                    days_of_supply=self.rng.uniform(20, 60),
                )
            )

        for pp in self.materials.primary_packs:
            self.inventory_positions.append(
                InventoryPosition(
                    material_id=pp.id,
                    location_id=pp.depot_id,
                    on_hand=round(self.rng.uniform(100, 500)),
                    available=round(self.rng.uniform(50, 300)),
                    days_of_supply=self.rng.uniform(15, 45),
                )
            )

        for fg in self.materials.finished_goods:
            # Inventory at depot
            self.inventory_positions.append(
                InventoryPosition(
                    material_id=fg.id,
                    location_id=fg.depot_id,
                    on_hand=round(self.rng.uniform(50, 300)),
                    in_transit=round(self.rng.uniform(0, 100)),
                    available=round(self.rng.uniform(30, 200)),
                    days_of_supply=self.rng.uniform(15, 45),
                )
            )
