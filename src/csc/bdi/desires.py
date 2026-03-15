"""Desire / goal representation for BDI agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class GoalStatus(str, Enum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    FAILED = "failed"
    DROPPED = "dropped"


@dataclass
class Desire:
    """A named objective with priority and satisfaction condition."""

    name: str
    priority: int  # lower = higher priority (1 = top)
    description: str
    is_satisfied: Callable[[Any], bool]  # takes BeliefBase, returns bool
    status: GoalStatus = GoalStatus.ACTIVE


@dataclass
class GoalHierarchy:
    """Ordered collection of desires for an agent."""

    desires: list[Desire] = field(default_factory=list)

    def active_desires(self) -> list[Desire]:
        """Return unsatisfied desires sorted by priority."""
        return sorted(
            [d for d in self.desires if d.status == GoalStatus.ACTIVE],
            key=lambda d: d.priority,
        )

    def update_statuses(self, belief_base: Any) -> None:
        """Re-evaluate which desires are satisfied."""
        for d in self.desires:
            if d.status == GoalStatus.ACTIVE and d.is_satisfied(belief_base):
                d.status = GoalStatus.ACHIEVED
