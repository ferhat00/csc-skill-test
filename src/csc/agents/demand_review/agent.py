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

    def get_compile_tool_name(self) -> str:
        return "aggregate_demand"

    def parse_output(self, raw: str) -> BaseModel:
        # Prefer the plan stored by aggregate_demand tool — it contains all
        # site_demands without relying on the LLM to reproduce them verbatim.
        cached = self.state._demand_plan_raw

        if cached is not None:
            src = cached
        else:
            json_str = _extract_json(raw)
            src = json.loads(json_str)

        # Overlay assumptions/demand summaries from LLM text when available.
        try:
            llm_json_str = _extract_json(raw)
            llm_data = json.loads(llm_json_str)
        except Exception:
            llm_data = {}

        assumptions = (
            llm_data.get("assumptions")
            or src.get("assumptions")
            or (src.get("notes", "").split(". ") if isinstance(src.get("notes"), str) else [])
        )

        demand_by_trial = (
            llm_data.get("demand_by_trial")
            or src.get("demand_by_trial")
            or {}
        )

        site_demands = []
        for sd in src.get("site_demands", []):
            site_demands.append(SiteDemand(
                trial_id=sd["trial_id"],
                site_id=sd["site_id"],
                month=date.fromisoformat(sd["month"]),
                finished_good_id=sd["finished_good_id"],
                quantity_kits=sd.get("quantity_kits", int(sd.get("kit_demand", 0))),
                quantity_with_overage=sd.get("quantity_with_overage", int(sd.get("kit_demand", 0))),
                safety_stock_kits=sd.get("safety_stock_kits", 0),
                urgency=Urgency(sd.get("urgency", "routine")),
            ))

        total_kit_demand = (
            src.get("total_kit_demand")
            or llm_data.get("total_kit_demand")
            or sum(sd.quantity_with_overage for sd in site_demands)
        )

        return DemandPlan(
            generated_at=datetime.now(),
            horizon_start=date.fromisoformat(src.get("horizon_start", str(date.today()))),
            horizon_end=date.fromisoformat(src.get("horizon_end", str(date.today()))),
            site_demands=site_demands,
            total_kit_demand=total_kit_demand,
            demand_by_trial=demand_by_trial,
            assumptions=assumptions,
        )


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown code fences.

    Falls back to a best-effort repair for truncated responses.
    """
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.find("```", start)
        candidate = text[start:end].strip() if end != -1 else text[start:].strip()
        if _is_valid_json(candidate):
            return candidate
        return _repair_json(candidate)
    if "```" in text:
        start = text.index("```") + 3
        end = text.find("```", start)
        candidate = text[start:end].strip() if end != -1 else text[start:].strip()
        if _is_valid_json(candidate):
            return candidate
        return _repair_json(candidate)
    # Try to find raw JSON using the stdlib decoder, which correctly handles
    # braces/brackets that appear inside string values.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                _, end = decoder.raw_decode(text, i)
                return text[i:end]
            except json.JSONDecodeError:
                # Might be truncated — try to repair from this offset
                repaired = _repair_json(text[i:])
                if _is_valid_json(repaired):
                    return repaired
                continue
    return text


def _is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _repair_json(text: str) -> str:
    """Attempt to close truncated JSON by tracking bracket/string state."""
    import re as _re

    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            stack.append(ch)
        elif ch in ("}", "]"):
            if stack:
                stack.pop()

    # Close any unterminated string first
    if in_string:
        text += '"'

    # Strip trailing whitespace for the checks below
    text = text.rstrip()

    # Remove trailing comma before closing (common LLM mistake)
    if text.endswith(","):
        text = text[:-1].rstrip()

    # If last meaningful token is a colon, the value is missing — inject null.
    # Also handles partial keywords: "key": tru / fals / nul
    if text.endswith(":"):
        text += " null"
    elif _re.search(r':\s*(?:tru|fals|nul)$', text):
        text = _re.sub(r'(:\s*)(?:tru|fals|nul)$', r'\1null', text)

    # Close all open containers in reverse order
    closing = {"{": "}", "[": "]"}
    for bracket in reversed(stack):
        text += closing[bracket]

    return text
