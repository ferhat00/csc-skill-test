"""Training orchestrator for RL agents."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from csc.config import Config
from csc.rl.agents.batch_agent import BatchRLAgent
from csc.rl.agents.capacity_agent import CapacityRLAgent
from csc.rl.agents.demand_agent import DemandRLAgent
from csc.rl.agents.inventory_agent import InventoryRLAgent
from csc.orchestrator.state import SharedState

console = Console()

AGENT_CLASSES = {
    "demand_forecast": DemandRLAgent,
    "capacity_allocation": CapacityRLAgent,
    "batch_scheduling": BatchRLAgent,
    "inventory_safety_stock": InventoryRLAgent,
}

AGENT_NAMES = list(AGENT_CLASSES.keys())


class RLTrainer:
    """Trains one or all RL agents using SB3 PPO."""

    def __init__(self, config: Config):
        self.config = config

    def train(
        self,
        agent_name: str | None = None,
        total_timesteps: int | None = None,
        seed: int | None = None,
    ) -> list[Path]:
        """Train RL agents.

        Args:
            agent_name: Train a specific agent, or None to train all.
            total_timesteps: Override training timesteps.
            seed: Override training seed.

        Returns:
            List of saved model paths.
        """
        timesteps = total_timesteps or self.config.rl_training_timesteps
        train_seed = seed or self.config.rl_training_seed
        model_dir = self.config.rl_model_dir

        if agent_name:
            if agent_name not in AGENT_CLASSES:
                raise ValueError(
                    f"Unknown agent: {agent_name}. "
                    f"Available: {', '.join(AGENT_NAMES)}"
                )
            agents_to_train = [agent_name]
        else:
            agents_to_train = AGENT_NAMES

        saved_paths: list[Path] = []

        for name in agents_to_train:
            console.print(f"\n[bold]{'='*60}[/]")
            console.print(f"[bold]  Training: {name}[/]")
            console.print(f"[bold]{'='*60}[/]")

            # Create a dummy state (agent only needs it for structure, not for training)
            state = SharedState()
            agent_cls = AGENT_CLASSES[name]
            agent = agent_cls(state=state)

            save_path = model_dir / f"{name}.zip"
            agent.train(
                total_timesteps=timesteps,
                seed=train_seed,
                save_path=save_path,
            )
            saved_paths.append(save_path)

        console.print(f"\n[bold green]Training complete. Models saved:[/]")
        for p in saved_paths:
            console.print(f"  {p}")

        return saved_paths
