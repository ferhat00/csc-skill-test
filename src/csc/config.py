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

    # Per-provider API keys (all optional; set whichever you use)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    nebius_api_key: str = ""

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> Config:
        load_dotenv(dotenv_path or ".env")

        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        nebius_api_key = os.environ.get("NEBIUS_API_KEY", "")

        if not any([anthropic_api_key, openai_api_key, gemini_api_key, nebius_api_key]):
            raise ValueError(
                "At least one provider API key is required. "
                "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, NEBIUS_API_KEY"
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
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
            gemini_api_key=gemini_api_key,
            nebius_api_key=nebius_api_key,
        )
