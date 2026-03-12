"""Conflict resolver: checks for cross-agent conflicts after all agents run."""

from __future__ import annotations

from rich.console import Console

from csc.orchestrator.state import SharedState

console = Console()


def resolve_conflicts(state: SharedState) -> list[str]:
    """Check for and report conflicts between agent outputs.

    Returns a list of conflict descriptions. An empty list means no conflicts.
    """
    conflicts: list[str] = []

    # Check 1: Supply plan feasibility vs capacity
    if state.depot_capacity_plan and not state.depot_capacity_plan.feasible:
        conflicts.append(
            "DEPOT CAPACITY: Supply plan exceeds depot capacity. "
            f"Adjustments needed: {'; '.join(state.depot_capacity_plan.adjustments)}"
        )

    if state.plant_capacity_plan and not state.plant_capacity_plan.feasible:
        conflicts.append(
            "PLANT CAPACITY: Supply plan exceeds plant capacity. "
            f"Adjustments needed: {'; '.join(state.plant_capacity_plan.adjustments)}"
        )

    # Check 2: Supply shortfalls
    if state.supply_plan and state.supply_plan.shortfall_alerts:
        for alert in state.supply_plan.shortfall_alerts:
            conflicts.append(f"SUPPLY SHORTFALL: {alert}")

    # Check 3: Demand exceeds what can be supplied
    if state.demand_plan and state.supply_plan:
        if state.demand_plan.total_kit_demand > 0 and len(state.supply_plan.batches) == 0:
            conflicts.append(
                f"SUPPLY GAP: Demand plan requires {state.demand_plan.total_kit_demand} kits "
                "but no production batches were scheduled"
            )

    if conflicts:
        console.print(f"\n[bold red]Found {len(conflicts)} conflict(s):[/]")
        for c in conflicts:
            console.print(f"  [red]• {c}[/]")
    else:
        console.print("\n[bold green]No cross-agent conflicts detected.[/]")

    return conflicts
