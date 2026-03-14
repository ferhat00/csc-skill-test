"""SB3 training callbacks for checkpointing and evaluation."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)


def get_training_callbacks(
    model_dir: Path,
    agent_name: str,
    eval_env=None,
    checkpoint_freq: int = 50_000,
    eval_freq: int = 100_000,
    n_eval_episodes: int = 5,
) -> list[BaseCallback]:
    """Create standard training callbacks.

    Args:
        model_dir: Directory to save checkpoints.
        agent_name: Agent name for file prefixes.
        eval_env: Optional evaluation environment.
        checkpoint_freq: Steps between checkpoints.
        eval_freq: Steps between evaluations.
        n_eval_episodes: Episodes per evaluation.

    Returns:
        List of SB3 callbacks.
    """
    callbacks: list[BaseCallback] = []

    # Periodic checkpoints
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix=agent_name,
            verbose=1,
        )
    )

    # Evaluation on held-out env
    if eval_env is not None:
        eval_dir = model_dir / "eval_logs"
        eval_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(eval_dir),
                log_path=str(eval_dir),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                verbose=1,
            )
        )

    return callbacks
