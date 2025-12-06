"""
Agent configuration for games.

Defines how agents behave, what tools they have access to, and their prompts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any
from pydantic import BaseModel


class AgentRole(Enum):
    """Base roles that can be extended per game."""
    PLAYER = "player"
    NPC = "npc"
    OBSERVER = "observer"


@dataclass
class AgentTool:
    """Defines a tool/action available to an agent."""
    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description
    handler: Callable[..., Any] | None = None  # Optional handler function

    def to_prompt_string(self) -> str:
        """Convert tool to a string for inclusion in prompts."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({params}): {self.description}"


@dataclass
class AgentPromptConfig:
    """Configuration for agent prompts."""

    # Core identity
    role_description: str = "You are an embodied agent in a 3D simulation."

    # Game-specific instructions
    game_instructions: str = ""

    # Available tools/actions (will be formatted into prompt)
    tools: list[AgentTool] = field(default_factory=list)

    # Win/lose conditions to communicate to agent
    win_condition: str = ""
    lose_condition: str = ""

    # Additional context (e.g., teammate info, rules)
    additional_context: str = ""

    # Response format instructions
    response_format: str = """
Respond with a JSON object containing:
- "thought": Your reasoning about the current situation
- "action": The action you want to take
- "action_params_json": A JSON string with parameters for the action (e.g., '{"direction": "forward", "duration": 3}')
"""

    def build_system_prompt(self) -> str:
        """Build the complete system prompt for this agent."""
        sections = [self.role_description]

        if self.game_instructions:
            sections.append(f"\n## Game Instructions\n{self.game_instructions}")

        if self.win_condition:
            sections.append(f"\n## Win Condition\n{self.win_condition}")

        if self.lose_condition:
            sections.append(f"\n## Lose Condition\n{self.lose_condition}")

        if self.tools:
            tool_strings = [tool.to_prompt_string() for tool in self.tools]
            sections.append(f"\n## Available Actions\n" + "\n".join(tool_strings))

        if self.additional_context:
            sections.append(f"\n## Additional Information\n{self.additional_context}")

        sections.append(f"\n## Response Format\n{self.response_format}")

        return "\n".join(sections)


@dataclass
class AgentConfig:
    """Complete configuration for a game agent."""

    agent_id: str
    role: AgentRole | str  # Can use custom string roles
    team: str | None = None

    # LLM configuration
    model_name: str = "gpt-4o"
    provider: str = "openai"
    temperature: float = 0.7

    # Prompt configuration
    prompt_config: AgentPromptConfig = field(default_factory=AgentPromptConfig)

    # Spawn configuration
    spawn_position: tuple[float, float] | None = None
    spawn_direction: tuple[float, float] = (1, 0)

    # Custom metadata for game-specific use
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentAction(BaseModel):
    """Standard response format for agent actions."""
    model_config = {"extra": "forbid"}

    thought: str = ""
    action: str = "do_nothing"
    # Parameters as JSON string to avoid OpenAI schema restrictions
    # Example: '{"direction": "forward", "duration": 3}'
    action_params_json: str = "{}"

    @property
    def action_params(self) -> dict[str, Any]:
        """Parse action_params_json to dict."""
        import json
        try:
            return json.loads(self.action_params_json)
        except json.JSONDecodeError:
            return {}


class AgentObservation(BaseModel):
    """Standard observation format for agents."""
    position: tuple[float, float]
    direction: tuple[float, float]
    camera_image: Any | None = None  # numpy array or base64 string
    nearby_agents: list[dict[str, Any]] = []
    nearby_objects: list[dict[str, Any]] = []
    game_state: dict[str, Any] = {}
    messages: list[str] = []  # Messages from other agents or game
