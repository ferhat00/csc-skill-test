"""Terminal reporter: rich console output for pipeline results."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from csc.orchestrator.state import SharedState

console = Console()


def print_summary(state: SharedState) -> None:
    """Print a rich summary of the pipeline results to the terminal."""
    console.print("\n[bold magenta]{'='*60}[/]")
    console.print("[bold magenta]  Clinical Supply Chain Pipeline Summary[/]")
    console.print(f"[bold magenta]{'='*60}[/]\n")

    # Demand summary
    if state.demand_plan:
        dp = state.demand_plan
        console.print("[bold]Demand Plan[/]")
        console.print(f"  Horizon: {dp.horizon_start} to {dp.horizon_end}")
        console.print(f"  Total Kit Demand: {dp.total_kit_demand:,}")

        table = Table(title="Demand by Trial")
        table.add_column("Protocol", style="cyan")
        table.add_column("Total Kits", justify="right", style="green")
        for protocol, kits in sorted(dp.demand_by_trial.items()):
            table.add_row(protocol, f"{kits:,}")
        console.print(table)

    # Portfolio summary
    if state.portfolio_plan:
        pp = state.portfolio_plan
        console.print("\n[bold]Portfolio Plan[/]")
        if pp.ranked_trials:
            table = Table(title="Trial Priority Ranking")
            table.add_column("Rank", justify="center")
            table.add_column("Protocol", style="cyan")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Phase")
            table.add_column("Therapy Area")
            for t in pp.ranked_trials[:10]:
                table.add_row(
                    str(t.get("rank", "")),
                    t.get("protocol", ""),
                    str(t.get("priority_score", "")),
                    t.get("phase", ""),
                    t.get("therapy_area", ""),
                )
            console.print(table)

        if pp.conflicts:
            console.print(f"\n  [yellow]Conflicts: {len(pp.conflicts)}[/]")
            for c in pp.conflicts[:5]:
                console.print(f"    • {c.get('description', c)}")

    # Supply summary
    if state.supply_plan:
        sp = state.supply_plan
        console.print(f"\n[bold]Supply Plan[/]")
        console.print(f"  Total Batches: {len(sp.batches)}")
        console.print(f"  Total Orders: {len(sp.orders)}")
        if sp.shortfall_alerts:
            console.print(f"  [red]Shortfall Alerts: {len(sp.shortfall_alerts)}[/]")

    # Capacity summaries
    if state.depot_capacity_plan:
        dcp = state.depot_capacity_plan
        status = "[green]FEASIBLE[/]" if dcp.feasible else "[red]INFEASIBLE[/]"
        console.print(f"\n[bold]Depot Capacity:[/] {status}")
        if dcp.adjustments:
            for adj in dcp.adjustments:
                console.print(f"  • {adj}")

    if state.plant_capacity_plan:
        pcp = state.plant_capacity_plan
        status = "[green]FEASIBLE[/]" if pcp.feasible else "[red]INFEASIBLE[/]"
        console.print(f"\n[bold]Plant Capacity:[/] {status}")
        if pcp.adjustments:
            for adj in pcp.adjustments:
                console.print(f"  • {adj}")

    # Event count
    console.print(f"\n[dim]Pipeline events: {len(state.events)}[/]")
