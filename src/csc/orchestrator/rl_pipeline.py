"""RL-based pipeline orchestration — drop-in alternative to SupplyChainPipeline."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from csc.config import Config
from csc.orchestrator.state import SharedState
from csc.rl.agents.batch_agent import BatchRLAgent
from csc.rl.agents.capacity_agent import CapacityRLAgent
from csc.rl.agents.demand_agent import DemandRLAgent
from csc.rl.agents.inventory_agent import InventoryRLAgent
from csc.rl.base_agent import BaseRLAgent

console = Console()

# Execution order mirrors the LLM pipeline logic:
# 1. Forecast demand
# 2. Allocate capacity (also produces portfolio + plant plans)
# 3. Schedule batches
# 4. Manage inventory & safety stock
RL_AGENT_SEQUENCE: list[tuple[str, type[BaseRLAgent]]] = [
    ("demand_forecast", DemandRLAgent),
    ("capacity_allocation", CapacityRLAgent),
    ("batch_scheduling", BatchRLAgent),
    ("inventory_safety_stock", InventoryRLAgent),
]


class RLSupplyChainPipeline:
    """Orchestrates RL agent execution, producing the same SharedState outputs
    as the LLM-based SupplyChainPipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.state = SharedState()

    def load_data(self, data_dir: Path) -> None:
        """Load reference data into shared state."""
        self.state.load_from_dir(data_dir)
        console.print(f"[green]Loaded data from {data_dir}[/]")

    def run_full(self) -> SharedState:
        """Run all RL agents in sequence."""
        console.print("\n[bold magenta]Running RL Supply Chain Pipeline[/]\n")

        for agent_name, agent_cls in RL_AGENT_SEQUENCE:
            model_path = self.config.rl_model_dir / f"{agent_name}.zip"
            agent = agent_cls(state=self.state, model_path=model_path)
            agent.load_model()
            agent.run()

        return self.state

    def run_agent(self, agent_name: str) -> None:
        """Run a single RL agent by name."""
        agent_map = dict(RL_AGENT_SEQUENCE)
        if agent_name not in agent_map:
            available = ", ".join(agent_map.keys())
            raise ValueError(f"Unknown RL agent: {agent_name}. Available: {available}")

        agent_cls = agent_map[agent_name]
        model_path = self.config.rl_model_dir / f"{agent_name}.zip"
        agent = agent_cls(state=self.state, model_path=model_path)
        agent.load_model()
        agent.run()
