"""Pipeline orchestrator: sequences agent execution through the supply chain."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from csc.agents.demand_review.agent import DemandReviewAgent
from csc.agents.depot_capacity.agent import DepotCapacityAgent
from csc.agents.plant_capacity.agent import PlantCapacityAgent
from csc.agents.portfolio_review.agent import PortfolioReviewAgent
from csc.agents.supply_review.agent import SupplyReviewAgent
from csc.config import Config
from csc.orchestrator.state import SharedState

console = Console()

# Agent execution order (depot and plant capacity can run sequentially after supply)
AGENT_SEQUENCE = [
    "demand_review",
    "portfolio_review",
    "supply_review",
    "depot_capacity",
    "plant_capacity",
]

AGENT_CLASSES = {
    "demand_review": DemandReviewAgent,
    "portfolio_review": PortfolioReviewAgent,
    "supply_review": SupplyReviewAgent,
    "depot_capacity": DepotCapacityAgent,
    "plant_capacity": PlantCapacityAgent,
}


class SupplyChainPipeline:
    """Orchestrates agent execution in the correct sequence."""

    def __init__(self, config: Config):
        self.config = config
        self.state = SharedState()

    def load_data(self, data_dir: Path | None = None) -> None:
        """Load synthetic data into shared state."""
        path = data_dir or self.config.data_dir
        console.print(f"[bold]Loading data from {path}[/]")
        self.state.load_from_dir(path)
        console.print(
            f"  Loaded: {len(self.state.trials)} trials, "
            f"{len(self.state.sites)} sites, "
            f"{len(self.state.depots)} depots, "
            f"{len(self.state.plants)} plants"
        )

    def run_full(self) -> SharedState:
        """Run all agents in sequence."""
        console.print("\n[bold magenta]Starting full pipeline[/]\n")
        for agent_name in AGENT_SEQUENCE:
            self.run_agent(agent_name)
        console.print("\n[bold green]Pipeline completed successfully![/]\n")
        return self.state

    def run_agent(self, agent_name: str) -> None:
        """Run a single agent."""
        if agent_name not in AGENT_CLASSES:
            raise ValueError(f"Unknown agent: {agent_name}. Choose from: {list(AGENT_CLASSES)}")

        agent_cls = AGENT_CLASSES[agent_name]
        agent = agent_cls(
            model=self.config.model,
            state=self.state,
            max_turns=self.config.max_agent_turns,
        )
        agent.run()
