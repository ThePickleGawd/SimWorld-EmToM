"""
Base game class for SimWorld games.

Provides the foundation for creating multi-agent LLM games with clear
win/fail conditions and customizable agent behaviors.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.llm.a2a_llm import A2ALLM
from simworld.utils.vector import Vector

from games.agent_config import (
    AgentConfig,
    AgentObservation,
    AgentAction,
    AgentTool,
    AgentPromptConfig,
)

logger = logging.getLogger(__name__)


class GameState(Enum):
    """Possible states of a game."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    WIN = "win"
    LOSE = "lose"
    DRAW = "draw"
    ERROR = "error"


@dataclass
class GameResult:
    """Result of a completed game."""
    state: GameState
    winner: str | list[str] | None = None  # Agent ID(s) or team name
    reason: str = ""
    total_steps: int = 0
    final_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameAgent:
    """Runtime container for an agent in the game."""
    config: AgentConfig
    humanoid: Humanoid | None = None
    llm: A2ALLM | None = None
    score: float = 0.0
    is_alive: bool = True
    action_history: list[AgentAction] = field(default_factory=list)
    message_inbox: list[str] = field(default_factory=list)


class BaseGame(ABC):
    """
    Abstract base class for SimWorld games.

    Subclasses must implement:
        - setup(): Initialize game-specific state
        - get_agent_configs(): Return list of agent configurations
        - check_win_condition(): Check if game is won
        - check_lose_condition(): Check if game is lost
        - build_observation(): Build observation for an agent
        - process_action(): Process an agent's action

    Optional overrides:
        - on_game_start(): Called when game starts
        - on_game_end(): Called when game ends
        - on_step_start(): Called at the start of each step
        - on_step_end(): Called at the end of each step
    """

    # Game metadata (override in subclass)
    name: str = "base_game"
    description: str = "Base game class"
    min_players: int = 1
    max_players: int = 10
    version: str = "1.0.0"

    def __init__(
        self,
        unrealcv: UnrealCV | None = None,
        communicator: Communicator | None = None,
        max_steps: int = 1000,
        verbose: bool = False,
    ):
        """
        Initialize the base game.

        Args:
            unrealcv: UnrealCV connection (created if not provided)
            communicator: Communicator instance (created if not provided)
            max_steps: Maximum steps before game ends in draw
            verbose: Enable verbose logging
        """
        self.unrealcv = unrealcv
        self.communicator = communicator
        self.max_steps = max_steps
        self.verbose = verbose

        # Runtime state
        self.agents: dict[str, GameAgent] = {}
        self.current_step: int = 0
        self.game_state: GameState = GameState.NOT_STARTED
        self.game_data: dict[str, Any] = {}  # Game-specific shared state

        # Connection state
        self._connected = False

    def connect(self, port: int = 9000, ip: str = "127.0.0.1") -> bool:
        """Connect to the Unreal Engine simulation."""
        if self._connected:
            return True

        try:
            if self.unrealcv is None:
                self.unrealcv = UnrealCV(port=port, ip=ip)
            if self.communicator is None:
                self.communicator = Communicator(self.unrealcv)
            self._connected = True
            logger.info(f"Connected to simulation at {ip}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from the simulation."""
        self._connected = False
        self.unrealcv = None
        self.communicator = None

    # =========================================================================
    # Abstract methods (must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize game-specific state.

        Called after agents are spawned but before the game loop starts.
        Use this to:
            - Spawn game objects
            - Set initial positions
            - Initialize game_data
        """
        pass

    @abstractmethod
    def get_agent_configs(self) -> list[AgentConfig]:
        """
        Return configurations for all agents in this game.

        This defines:
            - How many agents
            - Their roles, teams, prompts
            - Available tools/actions
            - Spawn positions
        """
        pass

    @abstractmethod
    def check_win_condition(self) -> tuple[bool, str | list[str] | None, str]:
        """
        Check if the game has been won.

        Returns:
            (is_won, winner, reason)
            - is_won: True if game is won
            - winner: Agent ID, list of IDs, or team name
            - reason: Human-readable explanation
        """
        pass

    @abstractmethod
    def check_lose_condition(self) -> tuple[bool, str]:
        """
        Check if the game has been lost.

        Returns:
            (is_lost, reason)
            - is_lost: True if game is lost
            - reason: Human-readable explanation
        """
        pass

    @abstractmethod
    def build_observation(self, agent_id: str) -> AgentObservation:
        """
        Build the observation for a specific agent.

        Args:
            agent_id: The agent to build observation for

        Returns:
            AgentObservation with current state visible to this agent
        """
        pass

    @abstractmethod
    def process_action(self, agent_id: str, action: AgentAction) -> dict[str, Any]:
        """
        Process an agent's action and update game state.

        Args:
            agent_id: The agent taking the action
            action: The action to process

        Returns:
            Result dict with at least {"success": bool, "message": str}
        """
        pass

    # =========================================================================
    # Optional hooks (override as needed)
    # =========================================================================

    def on_game_start(self) -> None:
        """Called when the game starts."""
        pass

    def on_game_end(self, result: GameResult) -> None:
        """Called when the game ends."""
        pass

    def on_step_start(self, step: int) -> None:
        """Called at the start of each game step."""
        pass

    def on_step_end(self, step: int) -> None:
        """Called at the end of each game step."""
        pass

    def on_agent_action(self, agent_id: str, action: AgentAction, result: dict) -> None:
        """Called after each agent action is processed."""
        pass

    # =========================================================================
    # Core game loop
    # =========================================================================

    def initialize(self) -> bool:
        """
        Initialize the game: spawn agents and set up game state.

        Returns:
            True if initialization succeeded
        """
        if not self._connected:
            logger.error("Not connected to simulation. Call connect() first.")
            return False

        # Clear any existing state
        self.agents.clear()
        self.current_step = 0
        self.game_state = GameState.NOT_STARTED
        self.game_data.clear()

        # Get agent configurations from subclass
        agent_configs = self.get_agent_configs()

        # Spawn each agent
        for config in agent_configs:
            if not self._spawn_agent(config):
                logger.error(f"Failed to spawn agent {config.agent_id}")
                return False

        # Run game-specific setup
        self.setup()

        logger.info(f"Game '{self.name}' initialized with {len(self.agents)} agents")
        return True

    def _spawn_agent(self, config: AgentConfig) -> bool:
        """Spawn a single agent based on its configuration."""
        try:
            # Create position vectors
            pos = Vector(config.spawn_position[0], config.spawn_position[1]) \
                if config.spawn_position else Vector(0, 0)
            dir = Vector(config.spawn_direction[0], config.spawn_direction[1])

            # Create humanoid
            humanoid = Humanoid(
                position=pos,
                direction=dir,
                communicator=self.communicator,
            )

            # Spawn in simulation
            self.communicator.spawn_agent(humanoid, name=config.agent_id, type='humanoid')

            # Create LLM for this agent
            llm = A2ALLM(
                model_name=config.model_name,
                provider=config.provider,
            )

            # Create game agent container
            game_agent = GameAgent(
                config=config,
                humanoid=humanoid,
                llm=llm,
            )

            self.agents[config.agent_id] = game_agent
            logger.info(f"Spawned agent {config.agent_id} at {pos}")
            return True

        except Exception as e:
            logger.error(f"Error spawning agent {config.agent_id}: {e}")
            return False

    def run(self) -> GameResult:
        """
        Run the complete game loop.

        Returns:
            GameResult with final outcome
        """
        if self.game_state != GameState.NOT_STARTED:
            logger.warning("Game already started or finished")

        self.game_state = GameState.RUNNING
        self.on_game_start()

        result = None

        try:
            while self.game_state == GameState.RUNNING:
                result = self._run_step()
                if result is not None:
                    break

                self.current_step += 1

                if self.current_step >= self.max_steps:
                    result = GameResult(
                        state=GameState.DRAW,
                        reason=f"Max steps ({self.max_steps}) reached",
                        total_steps=self.current_step,
                    )
                    break

        except Exception as e:
            logger.error(f"Game error: {e}")
            result = GameResult(
                state=GameState.ERROR,
                reason=str(e),
                total_steps=self.current_step,
            )

        if result is None:
            result = GameResult(
                state=GameState.DRAW,
                reason="Game ended without result",
                total_steps=self.current_step,
            )

        # Add final scores
        result.final_scores = {
            agent_id: agent.score
            for agent_id, agent in self.agents.items()
        }

        self.game_state = result.state
        self.on_game_end(result)

        return result

    def _run_step(self) -> GameResult | None:
        """Run a single game step. Returns GameResult if game ended."""
        self.on_step_start(self.current_step)

        # Check end conditions before agent actions
        is_won, winner, win_reason = self.check_win_condition()
        if is_won:
            return GameResult(
                state=GameState.WIN,
                winner=winner,
                reason=win_reason,
                total_steps=self.current_step,
            )

        is_lost, lose_reason = self.check_lose_condition()
        if is_lost:
            return GameResult(
                state=GameState.LOSE,
                reason=lose_reason,
                total_steps=self.current_step,
            )

        # Collect data for recording
        observations: dict[str, AgentObservation] = {}
        actions: dict[str, AgentAction] = {}
        results: dict[str, dict[str, Any]] = {}

        # Get actions from all agents
        for agent_id, agent in self.agents.items():
            if not agent.is_alive:
                continue

            # Build observation for this agent
            observation = self.build_observation(agent_id)
            observations[agent_id] = observation

            # Get action from LLM
            action = self._get_agent_action(agent_id, observation)
            actions[agent_id] = action

            # Process the action
            result = self.process_action(agent_id, action)
            results[agent_id] = result

            # Store action history
            agent.action_history.append(action)

            # Hook for subclasses
            self.on_agent_action(agent_id, action, result)

        # Record this step if recorder is attached
        if hasattr(self, '_recorder') and self._recorder:
            self._recorder.record_step(
                step=self.current_step,
                observations=observations,
                actions=actions,
                results=results,
            )

        self.on_step_end(self.current_step)
        return None

    def _get_agent_action(self, agent_id: str, observation: AgentObservation) -> AgentAction:
        """Get an action from an agent's LLM."""
        agent = self.agents[agent_id]

        # Build the system prompt from config
        system_prompt = agent.config.prompt_config.build_system_prompt()

        # Build user prompt with current observation
        user_prompt = self._build_user_prompt(agent_id, observation)

        try:
            # Get response from LLM
            response, _ = agent.llm.generate_instructions(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=[observation.camera_image] if observation.camera_image is not None else [],
                temperature=agent.config.temperature,
                response_format=AgentAction,
            )

            if response is None:
                logger.warning(f"Agent {agent_id} returned None response")
                return AgentAction()

            # Parse response
            if isinstance(response, str):
                return AgentAction.model_validate_json(response)
            elif isinstance(response, dict):
                return AgentAction.model_validate(response)
            else:
                return response

        except Exception as e:
            logger.error(f"Error getting action from agent {agent_id}: {e}")
            return AgentAction()

    def _build_user_prompt(self, agent_id: str, observation: AgentObservation) -> str:
        """Build the user prompt for an agent."""
        agent = self.agents[agent_id]

        sections = []

        # Current state
        sections.append(f"## Current State (Step {self.current_step})")
        sections.append(f"Position: ({observation.position[0]:.1f}, {observation.position[1]:.1f})")
        sections.append(f"Direction: ({observation.direction[0]:.2f}, {observation.direction[1]:.2f})")

        # Nearby agents
        if observation.nearby_agents:
            sections.append("\n## Nearby Agents")
            for nearby in observation.nearby_agents:
                sections.append(f"- {nearby.get('id', 'unknown')}: {nearby}")

        # Nearby objects
        if observation.nearby_objects:
            sections.append("\n## Nearby Objects")
            for obj in observation.nearby_objects:
                sections.append(f"- {obj.get('name', 'unknown')}: {obj}")

        # Game state
        if observation.game_state:
            sections.append("\n## Game State")
            sections.append(json.dumps(observation.game_state, indent=2))

        # Messages
        if observation.messages:
            sections.append("\n## Messages")
            for msg in observation.messages:
                sections.append(f"- {msg}")

        # Recent action history
        if agent.action_history:
            sections.append("\n## Your Recent Actions")
            for action in agent.action_history[-5:]:
                sections.append(f"- {action.action}: {action.thought[:50]}...")

        sections.append("\n## Your Turn")
        sections.append("What action do you take?")

        return "\n".join(sections)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_agent_position(self, agent_id: str) -> Vector | None:
        """Get an agent's current position."""
        agent = self.agents.get(agent_id)
        if agent and agent.humanoid:
            return agent.humanoid.position
        return None

    def get_distance_between_agents(self, agent1_id: str, agent2_id: str) -> float | None:
        """Get distance between two agents."""
        pos1 = self.get_agent_position(agent1_id)
        pos2 = self.get_agent_position(agent2_id)
        if pos1 and pos2:
            return pos1.distance(pos2)
        return None

    def send_message_to_agent(self, target_agent_id: str, message: str) -> None:
        """Send a message to an agent's inbox."""
        if target_agent_id in self.agents:
            self.agents[target_agent_id].message_inbox.append(message)

    def broadcast_message(self, message: str, exclude: list[str] | None = None) -> None:
        """Broadcast a message to all agents."""
        exclude = exclude or []
        for agent_id in self.agents:
            if agent_id not in exclude:
                self.send_message_to_agent(agent_id, message)

    def get_agents_by_team(self, team: str) -> list[str]:
        """Get all agent IDs belonging to a team."""
        return [
            agent_id for agent_id, agent in self.agents.items()
            if agent.config.team == team
        ]

    def get_alive_agents(self) -> list[str]:
        """Get all alive agent IDs."""
        return [
            agent_id for agent_id, agent in self.agents.items()
            if agent.is_alive
        ]
