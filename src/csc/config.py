"""Global configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    api_key: str
    model: str = "claude-sonnet-4-20250514"
    max_agent_turns: int = 20
    data_dir: Path = field(default_factory=lambda: Path("src/csc/data/output"))
    report_dir: Path = field(default_factory=lambda: Path("reports"))

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> Config:
        load_dotenv(dotenv_path or ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return cls(
            api_key=api_key,
            model=os.environ.get("CSC_MODEL", cls.model),
            max_agent_turns=int(os.environ.get("CSC_MAX_AGENT_TURNS", cls.max_agent_turns)),
        )
