"""Depot Capacity Agent: validates supply plan against packing depot capacity."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from csc.agents.base import BaseAgent
from csc.agents.demand_review.agent import _extract_json
from csc.agents.depot_capacity.prompts import SYSTEM_PROMPT
from csc.agents.depot_capacity.tools import create_tool_handlers, get_tool_definitions
from csc.models import DepotCapacityPlan


class DepotCapacityAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "depot_capacity"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tools(self) -> list[dict]:
        return get_tool_definitions()

    def get_tool_handlers(self) -> dict[str, Any]:
        return create_tool_handlers(self.state)

    def get_input_keys(self) -> list[str]:
        return ["supply_plan", "depots", "equipment_lines", "changeover_rules", "materials"]

    def get_output_key(self) -> str:
        return "depot_capacity_plan"

    def parse_output(self, raw: str) -> BaseModel:
        json_str = _extract_json(raw)
        data = json.loads(json_str)

        return DepotCapacityPlan(
            generated_at=data.get("generated_at", ""),
            feasible=data.get("feasible", True),
            adjustments=data.get("adjustments", []),
            reasoning=data.get("reasoning", []),
        )
