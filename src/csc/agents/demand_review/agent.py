"""Demand Review Agent: translates clinical operations into finished good demand."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from csc.agents.base import BaseAgent
from csc.agents.demand_review.prompts import SYSTEM_PROMPT
from csc.agents.demand_review.tools import create_tool_handlers, get_tool_definitions
from csc.models import DemandPlan, SiteDemand, Urgency


class DemandReviewAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "demand_review"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tools(self) -> list[dict]:
        return get_tool_definitions()

    def get_tool_handlers(self) -> dict[str, Any]:
        return create_tool_handlers(self.state)

    def get_input_keys(self) -> list[str]:
        return ["trials", "sites", "enrollment_forecasts", "patient_cohorts", "materials"]

    def get_output_key(self) -> str:
        return "demand_plan"

    def parse_output(self, raw: str) -> BaseModel:
        # Extract JSON from the response
        json_str = _extract_json(raw)
        data = json.loads(json_str)

        # Build DemandPlan from the aggregated data
        site_demands = []
        if "site_demands" in data:
            for sd in data["site_demands"]:
                site_demands.append(SiteDemand(
                    trial_id=sd["trial_id"],
                    site_id=sd["site_id"],
                    month=date.fromisoformat(sd["month"]),
                    finished_good_id=sd["finished_good_id"],
                    quantity_kits=sd.get("quantity_kits", 0),
                    quantity_with_overage=sd.get("quantity_with_overage", 0),
                    safety_stock_kits=sd.get("safety_stock_kits", 0),
                    urgency=Urgency(sd.get("urgency", "routine")),
                ))

        return DemandPlan(
            generated_at=datetime.now(),
            horizon_start=date.fromisoformat(data.get("horizon_start", str(date.today()))),
            horizon_end=date.fromisoformat(data.get("horizon_end", str(date.today()))),
            site_demands=site_demands,
            total_kit_demand=data.get("total_kit_demand", sum(sd.quantity_with_overage for sd in site_demands)),
            demand_by_trial=data.get("demand_by_trial", {}),
            assumptions=data.get("assumptions", []),
        )


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown code fences."""
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    # Try to find raw JSON
    for i, ch in enumerate(text):
        if ch == "{":
            # Find matching closing brace
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
    return text
