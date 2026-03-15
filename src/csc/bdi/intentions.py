"""Plan library and intention stack for BDI agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Plan:
    """A plan template in the plan library."""

    name: str
    goal_name: str  # which desire this plan addresses
    context_condition: Callable[[Any], bool]  # takes BeliefBase -> bool
    body: Callable[[Any, Any], None]  # takes (BeliefBase, agent) -> None
    priority: int = 0  # for choosing among applicable plans (lower = preferred)


@dataclass
class PlanLibrary:
    """Collection of plan templates indexed by goal name."""

    plans: list[Plan] = field(default_factory=list)

    def applicable_plans(self, goal_name: str, belief_base: Any) -> list[Plan]:
        """Return plans for a goal whose context conditions are met."""
        return sorted(
            [
                p
                for p in self.plans
                if p.goal_name == goal_name and p.context_condition(belief_base)
            ],
            key=lambda p: p.priority,
        )


@dataclass
class IntentionStack:
    """Currently committed plans being executed."""

    stack: list[Plan] = field(default_factory=list)

    def push(self, plan: Plan) -> None:
        self.stack.append(plan)

    def pop(self) -> Plan | None:
        return self.stack.pop() if self.stack else None

    def is_empty(self) -> bool:
        return len(self.stack) == 0

    def current(self) -> Plan | None:
        return self.stack[-1] if self.stack else None
