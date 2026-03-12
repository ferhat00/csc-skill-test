"""Portfolio Review Agent: cross-trial prioritization and conflict detection."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from csc.agents.base import BaseAgent
from csc.agents.demand_review.agent import _extract_json
from csc.agents.portfolio_review.prompts import SYSTEM_PROMPT
from csc.agents.portfolio_review.tools import create_tool_handlers, get_tool_definitions
from csc.orchestrator.state import PortfolioPlan


class PortfolioReviewAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "portfolio_review"

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def get_tools(self) -> list[dict]:
        return get_tool_definitions()

    def get_tool_handlers(self) -> dict[str, Any]:
        return create_tool_handlers(self.state)

    def get_input_keys(self) -> list[str]:
        return ["trials", "demand_plan", "materials", "plants", "depots"]

    def get_output_key(self) -> str:
        return "portfolio_plan"

    def parse_output(self, raw: str) -> BaseModel:
        json_str = _extract_json(raw)
        data = json.loads(json_str)

        return PortfolioPlan(
            generated_at=datetime.now(),
            ranked_trials=data.get("ranked_trials", []),
            conflicts=data.get("conflicts", []),
            synergies=data.get("synergies", []),
            resource_allocations=data.get("resource_allocations", []),
            reasoning=data.get("reasoning", []),
        )
