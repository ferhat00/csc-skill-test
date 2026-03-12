"""Supply Review Agent: converts demand into backward-planned supply across all stages."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from csc.agents.base import BaseAgent
from csc.agents.demand_review.agent import _extract_json
from csc.agents.supply_review.prompts import SYSTEM_PROMPT
from csc.agents.supply_review.tools import create_tool_handlers, get_tool_definitions
from csc.models import Batch, BatchStatus, SupplyChainStage, SupplyPlan, UnitOfMeasure


class SupplyReviewAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "supply_review"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tools(self) -> list[dict]:
        return get_tool_definitions()

    def get_tool_handlers(self) -> dict[str, Any]:
        return create_tool_handlers(self.state)

    def get_input_keys(self) -> list[str]:
        return ["demand_plan", "portfolio_plan", "materials", "inventory_positions", "plants", "depots", "transport_lanes"]

    def get_output_key(self) -> str:
        return "supply_plan"

    def parse_output(self, raw: str) -> BaseModel:
        json_str = _extract_json(raw)
        data = json.loads(json_str)

        batches = []
        for b in data.get("batches", []):
            batches.append(Batch(
                material_id=b.get("material_id", "00000000-0000-0000-0000-000000000000"),
                stage=SupplyChainStage(b.get("stage", "ds")),
                batch_number=b.get("batch_number", "UNKNOWN"),
                quantity=b.get("quantity", 0),
                unit=UnitOfMeasure(b.get("unit", "units")),
                status=BatchStatus.PLANNED,
                planned_start=date.fromisoformat(b.get("planned_start", str(date.today()))),
                planned_end=date.fromisoformat(b.get("planned_end", str(date.today()))),
                expiry_date=date.fromisoformat(b.get("expiry_date", str(date.today().replace(year=date.today().year + 2)))),
                location_id=b.get("location_id", "00000000-0000-0000-0000-000000000000"),
            ))

        return SupplyPlan(
            generated_at=datetime.now(),
            horizon_start=date.fromisoformat(data.get("horizon_start", str(date.today()))),
            horizon_end=date.fromisoformat(data.get("horizon_end", str(date.today()))),
            batches=batches,
            shortfall_alerts=data.get("shortfall_alerts", []),
            reasoning=data.get("reasoning", data.get("notes", "").split(". ") if isinstance(data.get("notes"), str) else []),
        )
