"""Base agent class with the Anthropic tool-use agentic loop."""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from typing import Any

import anthropic
from pydantic import BaseModel
from rich.console import Console

from csc.orchestrator.state import SharedState

console = Console()


class BaseAgent(ABC):
    """Abstract base for all supply chain agents.

    Each agent:
    1. Has a system prompt defining its role and domain expertise
    2. Has a set of tools (Python functions) it can invoke
    3. Reads from and writes to SharedState
    4. Produces a structured output (a Pydantic model)
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        state: SharedState,
        max_turns: int = 20,
    ):
        self.client = client
        self.model = model
        self.state = state
        self.max_turns = max_turns
        self.conversation: list[dict] = []

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Short identifier for this agent, e.g. 'demand_review'."""
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt that defines this agent's role."""
        ...

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return Anthropic tool definitions for this agent."""
        ...

    @abstractmethod
    def get_tool_handlers(self) -> dict[str, Any]:
        """Return a mapping of tool name -> callable."""
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
    def parse_output(self, raw: str) -> BaseModel:
        """Parse the agent's final text response into a structured output."""
        ...

    def run(self) -> BaseModel:
        """Execute the agentic loop: call Claude with tools until it produces a final answer."""
        self.state.log_event(self.agent_name, "started", f"{self.agent_name} agent starting")
        console.print(f"\n[bold blue]{'='*60}[/]")
        console.print(f"[bold blue]  {self.agent_name.replace('_', ' ').title()} Agent[/]")
        console.print(f"[bold blue]{'='*60}[/]\n")

        # Build initial context from shared state
        context_msg = self._build_context_message()
        self.conversation = [{"role": "user", "content": context_msg}]

        tool_handlers = self.get_tool_handlers()
        tools = self.get_tools()

        for turn in range(self.max_turns):
            console.print(f"  [dim]Turn {turn + 1}/{self.max_turns}...[/]")

            response = self.client.messages.create(
                model=self.model,
                system=self.get_system_prompt(),
                messages=self.conversation,
                tools=tools,
                max_tokens=8192,
            )

            # Process the response
            assistant_content = response.content
            self.conversation.append({"role": "assistant", "content": assistant_content})

            # Check if there are tool uses
            tool_uses = [block for block in assistant_content if block.type == "tool_use"]

            if not tool_uses:
                # Agent is done — extract final text
                text_blocks = [block.text for block in assistant_content if block.type == "text"]
                final_text = "\n".join(text_blocks)
                console.print(f"  [green]Agent completed after {turn + 1} turns[/]")

                # Parse and store output
                try:
                    output = self.parse_output(final_text)
                    self.state.set(self.get_output_key(), output)
                    self.state.log_event(
                        self.agent_name, "completed",
                        f"{self.agent_name} produced output",
                        {"output_key": self.get_output_key()},
                    )
                    return output
                except Exception as e:
                    self.state.log_event(
                        self.agent_name, "error",
                        f"Failed to parse output: {e}",
                    )
                    console.print(f"  [red]Parse error: {e}[/]")
                    # Ask the agent to fix its output
                    self.conversation.append({
                        "role": "user",
                        "content": f"Your output could not be parsed. Error: {e}\n\nPlease provide your final output as valid JSON that matches the expected schema.",
                    })
                    continue

            # Execute tool calls
            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input
                console.print(f"    [yellow]Tool: {tool_name}[/]")

                self.state.log_event(
                    self.agent_name, "tool_call",
                    f"Calling {tool_name}",
                    {"tool_name": tool_name, "input": str(tool_input)[:200]},
                )

                handler = tool_handlers.get(tool_name)
                if handler:
                    try:
                        result = handler(**tool_input)
                        result_str = json.dumps(result, default=str) if not isinstance(result, str) else result
                    except Exception as e:
                        result_str = json.dumps({"error": str(e), "traceback": traceback.format_exc()})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_str[:10000],  # cap result size
                })

            self.conversation.append({"role": "user", "content": tool_results})

        # Max turns exhausted
        console.print(f"  [red]Max turns ({self.max_turns}) exhausted[/]")
        self.state.log_event(self.agent_name, "error", "Max turns exhausted")
        raise RuntimeError(f"{self.agent_name} exceeded maximum turns ({self.max_turns})")

    def _build_context_message(self) -> str:
        """Pull relevant data from SharedState and format as context for the agent."""
        parts = [
            f"You are the {self.agent_name.replace('_', ' ').title()} agent.",
            "Below is the current state data relevant to your analysis.",
            "Use the available tools to analyze this data and produce your output.",
            "When you are done, provide your final structured output as a JSON object.",
            "",
        ]

        for key in self.get_input_keys():
            value = self.state.get(key)
            if value is None:
                parts.append(f"## {key}\nNo data available.\n")
                continue

            if isinstance(value, BaseModel):
                serialized = value.model_dump(mode="json")
            elif isinstance(value, list) and value:
                if isinstance(value[0], BaseModel):
                    serialized = [item.model_dump(mode="json") for item in value]
                else:
                    serialized = value
            else:
                serialized = value

            # Truncate large datasets to avoid exceeding context
            json_str = json.dumps(serialized, default=str, indent=1)
            if len(json_str) > 30000:
                json_str = json_str[:30000] + "\n... (truncated)"

            parts.append(f"## {key}\n```json\n{json_str}\n```\n")

        return "\n".join(parts)
