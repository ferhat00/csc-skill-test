"""BaseBDIAgent: abstract base class with the BDI reasoning cycle."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel
from rich.console import Console

from csc.bdi.beliefs import BeliefBase
from csc.bdi.desires import GoalHierarchy, GoalStatus
from csc.bdi.intentions import IntentionStack, PlanLibrary
from csc.orchestrator.state import SharedState

console = Console()


class BaseBDIAgent(ABC):
    """Abstract base for all BDI supply chain agents.

    Each BDI agent:
    1. Reads from SharedState into a BeliefBase
    2. Defines a GoalHierarchy (desires with priorities)
    3. Provides a PlanLibrary (plan templates with context conditions)
    4. Runs the BDI reasoning cycle
    5. Builds a Pydantic output model from final beliefs
    6. Writes the output to SharedState
    """

    def __init__(self, state: SharedState) -> None:
        self.state = state
        self.belief_base = BeliefBase()
        self.goal_hierarchy = GoalHierarchy()
        self.plan_library = PlanLibrary()
        self.intention_stack = IntentionStack()

    @property
    @abstractmethod
    def agent_name(self) -> str: ...

    @abstractmethod
    def get_input_keys(self) -> list[str]: ...

    @abstractmethod
    def get_output_key(self) -> str: ...

    @abstractmethod
    def initialize_beliefs(self) -> None:
        """Compute derived beliefs from SharedState data."""
        ...

    @abstractmethod
    def define_desires(self) -> None:
        """Populate self.goal_hierarchy with agent-specific goals."""
        ...

    @abstractmethod
    def define_plans(self) -> None:
        """Populate self.plan_library with agent-specific plan templates."""
        ...

    @abstractmethod
    def build_output(self) -> BaseModel:
        """Construct the final Pydantic output model from belief_base."""
        ...

    def run(self) -> BaseModel:
        """Execute the full BDI agent lifecycle."""
        self.state.log_event(self.agent_name, "started", f"BDI {self.agent_name} agent starting")

        title = self.agent_name.replace("_", " ").title()
        console.print(f"\n[bold green]{'=' * 60}[/]")
        console.print(f"[bold green]  {title} (BDI Agent)[/]")
        console.print(f"[bold green]{'=' * 60}[/]\n")

        # Phase 1: Load beliefs from SharedState
        self.belief_base.update_from_state(self.state, self.get_input_keys())
        self.initialize_beliefs()

        # Phase 2: Define desires and plans
        self.define_desires()
        self.define_plans()

        # Phase 3: BDI reasoning cycle
        self._bdi_cycle()

        # Phase 4: Build and store output
        output = self.build_output()
        self.state.set(self.get_output_key(), output)

        self.state.log_event(
            self.agent_name,
            "completed",
            f"BDI {self.agent_name} produced output",
            {"output_key": self.get_output_key()},
        )
        console.print(f"  [green][OK] BDI {self.agent_name} completed[/]\n")
        return output

    def _bdi_cycle(self, max_iterations: int = 50) -> None:
        """Standard BDI reasoning cycle with Rich console output."""
        for iteration in range(max_iterations):
            # Step 1: Deliberation — update goal statuses
            self.goal_hierarchy.update_statuses(self.belief_base)
            active = self.goal_hierarchy.active_desires()

            if not active:
                console.print(f"\n  [green]All goals achieved in {iteration + 1} cycle(s)[/]")
                break

            console.print(f"\n  [bold]Cycle {iteration + 1}:[/]")
            self._print_belief_summary()

            # Track which goals we addressed this cycle to avoid duplicates
            addressed_goals: set[str] = set()

            # Step 2: Plan selection for active goals
            for desire in active:
                if desire.name in addressed_goals:
                    continue

                applicable = self.plan_library.applicable_plans(desire.name, self.belief_base)
                if applicable:
                    selected = applicable[0]
                    self.intention_stack.push(selected)
                    addressed_goals.add(desire.name)

            # Step 3: Execute intentions
            while not self.intention_stack.is_empty():
                plan = self.intention_stack.pop()
                goal = next(
                    (d for d in self.goal_hierarchy.desires if d.name == plan.goal_name),
                    None,
                )
                status_before = goal.status if goal else GoalStatus.ACTIVE

                console.print(f"    Goal: {plan.goal_name} [{status_before.value.upper()}]")
                console.print(f"      -> Plan: {plan.name}")

                plan.body(self.belief_base, self)

                # Re-check if this specific goal is now satisfied
                if goal and goal.status == GoalStatus.ACTIVE and goal.is_satisfied(self.belief_base):
                    goal.status = GoalStatus.ACHIEVED

                console.print(f"      -> Done")

            # Step 4: Final re-check
            self.goal_hierarchy.update_statuses(self.belief_base)
            if not self.goal_hierarchy.active_desires():
                console.print(f"\n  [green]All goals achieved in {iteration + 1} cycle(s)[/]")
                break
        else:
            console.print(f"\n  [yellow]Max iterations ({max_iterations}) reached[/]")

    def _print_belief_summary(self) -> None:
        """Print a concise summary of current beliefs."""
        parts = []
        for key in self.get_input_keys():
            val = self.belief_base.get(key)
            if isinstance(val, list):
                parts.append(f"{len(val)} {key}")
            elif val is not None:
                parts.append(f"{key}: loaded")
        if parts:
            console.print(f"    Beliefs: {', '.join(parts)}")
