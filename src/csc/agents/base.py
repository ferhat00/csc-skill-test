"""Base agent class with a LiteLLM-powered tool-use agentic loop."""

from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any

import litellm
from pydantic import BaseModel
from rich.console import Console

from csc.orchestrator.state import SharedState

console = Console()

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


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
        model: str,
        state: SharedState,
        max_turns: int = 20,
    ):
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
        """Return tool definitions in Anthropic format (converted internally to LiteLLM format)."""
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
        """Execute the agentic loop using LiteLLM until the agent produces a final answer."""
        self.state.log_event(self.agent_name, "started", f"{self.agent_name} agent starting")
        console.print(f"\n[bold blue]{'='*60}[/]")
        console.print(f"[bold blue]  {self.agent_name.replace('_', ' ').title()} Agent[/]")
        console.print(f"[bold blue]{'='*60}[/]\n")

        self.conversation = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self._build_context_message()},
        ]

        tool_handlers = self.get_tool_handlers()
        litellm_tools = self._to_litellm_tools(self.get_tools())

        for turn in range(self.max_turns):
            console.print(f"  [dim]Turn {turn + 1}/{self.max_turns}...[/]")

            response = self._call_with_retry(
                model=self.model,
                messages=self.conversation,
                tools=litellm_tools,
                max_tokens=8192,
            )

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                # Agent is done — extract final text
                final_text = message.content or ""
                console.print(f"  [green]Agent completed after {turn + 1} turns[/]")

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
                    self.conversation.append({"role": "assistant", "content": final_text})
                    self.conversation.append({
                        "role": "user",
                        "content": (
                            f"Your output could not be parsed. Error: {e}\n\n"
                            "Please provide your final output as valid JSON that matches the expected schema."
                        ),
                    })
                    continue

            # Append the assistant message (with tool calls) to the conversation
            asst_msg: dict[str, Any] = {"role": "assistant", "content": message.content}
            asst_msg["tool_calls"] = [tc.model_dump() for tc in tool_calls]
            self.conversation.append(asst_msg)

            # Execute each tool call and append results
            for tc in tool_calls:
                tool_name = tc.function.name
                console.print(f"    [yellow]Tool: {tool_name}[/]")

                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                self.state.log_event(
                    self.agent_name, "tool_call",
                    f"Calling {tool_name}",
                    {"tool_name": tool_name, "input": str(tool_input)[:200]},
                )

                handler = tool_handlers.get(tool_name)
                if handler:
                    try:
                        result = handler(**tool_input)
                        result_str = (
                            json.dumps(result, default=str)
                            if not isinstance(result, str)
                            else result
                        )
                    except Exception as e:
                        result_str = json.dumps({"error": str(e), "traceback": traceback.format_exc()})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})

                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str[:10000],
                })

        # Max turns exhausted
        console.print(f"  [red]Max turns ({self.max_turns}) exhausted[/]")
        self.state.log_event(self.agent_name, "error", "Max turns exhausted")
        raise RuntimeError(f"{self.agent_name} exceeded maximum turns ({self.max_turns})")

    def _to_litellm_tools(self, anthropic_tools: list[dict]) -> list[dict]:
        """Convert Anthropic-style tool definitions to LiteLLM/OpenAI format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in anthropic_tools
        ]

    def _call_with_retry(self, **kwargs) -> litellm.ModelResponse:
        """Call LiteLLM with exponential backoff on rate limit errors."""
        max_retries = 5
        delay = 60
        for attempt in range(max_retries):
            try:
                return litellm.completion(**kwargs)
            except litellm.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                console.print(
                    f"  [yellow]Rate limit hit — waiting {delay}s before retry "
                    f"{attempt + 2}/{max_retries}...[/]"
                )
                time.sleep(delay)
                delay = min(delay * 2, 300)

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

            json_str = json.dumps(serialized, default=str, indent=1)
            if len(json_str) > 30000:
                json_str = json_str[:30000] + "\n... (truncated)"

            parts.append(f"## {key}\n```json\n{json_str}\n```\n")

        return "\n".join(parts)
