"""Belief representation for BDI agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from csc.orchestrator.state import SharedState


@dataclass
class Belief:
    """A single named belief with a typed value and provenance."""

    name: str
    value: Any
    source: str = "computed"  # "shared_state" | "computed" | "upstream_agent"
    confidence: float = 1.0


@dataclass
class BeliefBase:
    """An agent's world model — a keyed collection of beliefs."""

    beliefs: dict[str, Belief] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        b = self.beliefs.get(name)
        return b.value if b else None

    def set(self, name: str, value: Any, source: str = "computed") -> None:
        self.beliefs[name] = Belief(name=name, value=value, source=source)

    def has(self, name: str) -> bool:
        return name in self.beliefs and self.beliefs[name].value is not None

    def update_from_state(self, state: SharedState, keys: list[str]) -> None:
        """Bulk-load beliefs from SharedState reference data."""
        for key in keys:
            val = getattr(state, key, None)
            if val is not None:
                self.set(key, val, source="shared_state")
