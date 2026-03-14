"""Global configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    model: str = "claude-sonnet-4-20250514"
    max_agent_turns: int = 20
    data_dir: Path = field(default_factory=lambda: Path("src/csc/data/output"))
    report_dir: Path = field(default_factory=lambda: Path("reports"))

    # Agent method: "llm" or "rl"
    method: str = "llm"

    # RL-specific settings
    rl_model_dir: Path = field(default_factory=lambda: Path("src/csc/rl/models"))
    rl_training_timesteps: int = 500_000
    rl_training_seed: int = 42

    # Per-provider API keys (all optional; set whichever you use)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    nebius_api_key: str = ""

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> Config:
        load_dotenv(dotenv_path or ".env")

        method = os.environ.get("CSC_METHOD", "llm")

        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        nebius_api_key = os.environ.get("NEBIUS_API_KEY", "")

        # API keys only required for LLM method
        if method == "llm" and not any([anthropic_api_key, openai_api_key, gemini_api_key, nebius_api_key]):
            raise ValueError(
                "At least one provider API key is required for LLM method. "
                "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, NEBIUS_API_KEY\n"
                "Or set CSC_METHOD=rl to use reinforcement learning agents instead."
            )

        # Export keys into env so LiteLLM can pick them up automatically
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
        if nebius_api_key:
            os.environ["NEBIUS_API_KEY"] = nebius_api_key

        return cls(
            model=os.environ.get("CSC_MODEL", cls.model),
            max_agent_turns=int(os.environ.get("CSC_MAX_AGENT_TURNS", cls.max_agent_turns)),
            method=method,
            rl_model_dir=Path(os.environ.get("CSC_RL_MODEL_DIR", "src/csc/rl/models")),
            rl_training_timesteps=int(os.environ.get("CSC_RL_TRAINING_TIMESTEPS", 500_000)),
            rl_training_seed=int(os.environ.get("CSC_RL_TRAINING_SEED", 42)),
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            nebius_api_key=nebius_api_key,
        )
