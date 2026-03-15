"""Base RL agent class — mirrors BaseAgent's SharedState contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from rich.console import Console

from csc.orchestrator.state import SharedState

console = Console()


class BaseRLAgent(ABC):
    """Abstract base for all RL supply chain agents.

    Each RL agent:
    1. Reads from SharedState (same keys as its LLM counterpart)
    2. Builds a numpy observation vector from the state
    3. Uses a trained SB3 model to predict an action
    4. Maps the action back to a Pydantic model output
    5. Writes the output to SharedState
    """

    def __init__(self, state: SharedState, model_path: Path | None = None):
        self.state = state
        self.model_path = model_path
        self._sb3_model: Any | None = None
        self._index_maps: dict[str, dict[int, Any]] = {}

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Short identifier, e.g. 'demand_forecast'."""
        ...

    @abstractmethod
    def get_input_keys(self) -> list[str]:
        """Which SharedState keys this agent reads."""
        ...

    @abstractmethod
    def get_output_key(self) -> str:
        """Which SharedState key this agent writes to."""
        ...

    @abstractmethod
    def build_observation(self) -> np.ndarray:
        """Convert SharedState data into a flat numpy observation vector."""
        ...

    @abstractmethod
    def map_action_to_output(self, action: np.ndarray) -> BaseModel:
        """Convert the SB3 model's raw action array into a Pydantic model."""
        ...

    @abstractmethod
    def create_env(self, seed: int = 42) -> Any:
        """Create the Gymnasium environment for training."""
        ...

    def load_model(self, path: Path | None = None) -> None:
        """Load a trained SB3 PPO model from disk."""
        from stable_baselines3 import PPO

        load_path = path or self.model_path
        if load_path is None:
            raise FileNotFoundError(
                f"No model path provided for {self.agent_name}. "
                "Train a model first with: python -m csc.cli train"
            )
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(
                f"Model not found at {load_path}. "
                "Train a model first with: python -m csc.cli train"
            )
        self._sb3_model = PPO.load(str(load_path))
        console.print(f"  [green]Loaded RL model from {load_path}[/]")

    def run(self) -> BaseModel:
        """Inference: build obs, predict action, map to output, store in state."""
        self.state.log_event(self.agent_name, "started", f"RL {self.agent_name} agent starting")
        console.print(f"\n[bold cyan]{'='*60}[/]")
        console.print(f"[bold cyan]  {self.agent_name.replace('_', ' ').title()} (RL Agent)[/]")
        console.print(f"[bold cyan]{'='*60}[/]\n")

        if self._sb3_model is None:
            raise RuntimeError(
                f"No model loaded for {self.agent_name}. Call load_model() first."
            )

        obs = self.build_observation()
        action, _ = self._sb3_model.predict(obs, deterministic=True)
        output = self.map_action_to_output(action)

        self.state.set(self.get_output_key(), output)
        self.state.log_event(
            self.agent_name, "completed",
            f"RL {self.agent_name} produced output",
            {"output_key": self.get_output_key()},
        )
        console.print(f"  [green]RL agent completed[/]")
        return output

    def train(
        self,
        total_timesteps: int = 500_000,
        seed: int = 42,
        save_path: Path | None = None,
    ) -> Path:
        """Train the agent's policy using SB3 PPO."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor

        env = self.create_env(seed=seed)
        env = Monitor(env)

        console.print(f"  [bold]Training {self.agent_name} for {total_timesteps:,} timesteps...[/]")

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
        model.learn(total_timesteps=total_timesteps)

        out = save_path or Path(f"src/csc/rl/models/{self.agent_name}.zip")
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        console.print(f"  [green]Model saved to {out}[/]")

        self._sb3_model = model
        return out
