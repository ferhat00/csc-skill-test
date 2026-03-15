"""BDI-based pipeline orchestration with iterative replanning."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from csc.bdi.agents.demand_agent import DemandBDIAgent
from csc.bdi.agents.depot_agent import DepotCapacityBDIAgent
from csc.bdi.agents.plant_agent import PlantCapacityBDIAgent
from csc.bdi.agents.portfolio_agent import PortfolioBDIAgent
from csc.bdi.agents.supply_agent import SupplyBDIAgent
from csc.bdi.base_agent import BaseBDIAgent
from csc.config import Config
from csc.orchestrator.state import SharedState

console = Console()

BDI_AGENT_SEQUENCE: list[tuple[str, type[BaseBDIAgent]]] = [
    ("demand_review", DemandBDIAgent),
    ("portfolio_review", PortfolioBDIAgent),
    ("supply_review", SupplyBDIAgent),
    ("depot_capacity", DepotCapacityBDIAgent),
    ("plant_capacity", PlantCapacityBDIAgent),
]

MAX_REPLANS = 2


class BDISupplyChainPipeline:
    """Orchestrates BDI agent execution with iterative replanning.

    Produces the same SharedState outputs as LLM and RL pipelines.
    If capacity agents detect infeasibility, the pipeline re-runs with
    capacity constraints injected as beliefs for upstream agents.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.state = SharedState()

    def load_data(self, data_dir: Path) -> None:
        self.state.load_from_dir(data_dir)
        console.print(f"[green]Loaded data from {data_dir}[/]")

    def run_full(self) -> SharedState:
        console.print("\n[bold magenta]Running BDI Supply Chain Pipeline[/]\n")

        for attempt in range(1 + MAX_REPLANS):
            if attempt > 0:
                console.print(f"\n[bold yellow]{'=' * 60}[/]")
                console.print(f"[bold yellow]  REPLANNING ITERATION {attempt + 1}[/]")
                console.print(f"[bold yellow]{'=' * 60}[/]\n")

            # Run all 5 agents in sequence
            for _agent_name, agent_cls in BDI_AGENT_SEQUENCE:
                agent = agent_cls(state=self.state)
                agent.run()

            # Check feasibility
            depot_ok = self.state.depot_capacity_plan and self.state.depot_capacity_plan.feasible
            plant_ok = self.state.plant_capacity_plan and self.state.plant_capacity_plan.feasible

            if depot_ok and plant_ok:
                console.print("\n[bold green]Pipeline feasible — no replanning needed[/]")
                break

            if attempt < MAX_REPLANS:
                console.print(f"\n[bold yellow]Infeasibility detected — replanning (attempt {attempt + 2}/{1 + MAX_REPLANS})[/]")

                # Collect capacity constraints from infeasible plans
                constraints: list[str] = []
                if self.state.depot_capacity_plan and not self.state.depot_capacity_plan.feasible:
                    constraints.extend(self.state.depot_capacity_plan.adjustments)
                    console.print(f"  [yellow]Depot constraints: {len(self.state.depot_capacity_plan.adjustments)}[/]")
                if self.state.plant_capacity_plan and not self.state.plant_capacity_plan.feasible:
                    constraints.extend(self.state.plant_capacity_plan.adjustments)
                    console.print(f"  [yellow]Plant constraints: {len(self.state.plant_capacity_plan.adjustments)}[/]")

                # Inject constraints for next iteration
                self.state._bdi_capacity_constraints = constraints

                # Clear agent outputs for re-run
                self.state.demand_plan = None
                self.state.portfolio_plan = None
                self.state.supply_plan = None
                self.state.depot_capacity_plan = None
                self.state.plant_capacity_plan = None
            else:
                console.print(f"\n[bold red]Max replanning attempts ({MAX_REPLANS}) reached — proceeding with infeasible plan[/]")

        return self.state

    def run_agent(self, agent_name: str) -> None:
        """Run a single BDI agent by name."""
        agent_map = dict(BDI_AGENT_SEQUENCE)
        if agent_name not in agent_map:
            available = ", ".join(agent_map.keys())
            raise ValueError(f"Unknown BDI agent: {agent_name}. Available: {available}")

        agent_cls = agent_map[agent_name]
        agent = agent_cls(state=self.state)
        agent.run()
