"""
Hide and Seek game implementation.

A multi-agent game where hiders try to stay hidden while seekers try to find them.
"""

import logging
import random
from typing import Any

from simworld.utils.vector import Vector

from games.base import BaseGame, GameResult, GameState
from games.agent_config import (
    AgentConfig,
    AgentObservation,
    AgentAction,
    AgentPromptConfig,
)
from games.registry import register_game
from games.examples.hide_and_seek.tools import HIDER_TOOLS, SEEKER_TOOLS

logger = logging.getLogger(__name__)


# Custom roles as strings (AgentConfig.role accepts str | AgentRole)
ROLE_HIDER = "hider"
ROLE_SEEKER = "seeker"


@register_game
class HideAndSeekGame(BaseGame):
    """
    Hide and Seek: A classic game with LLM-powered agents.

    Rules:
    - Hiders get a head start to find hiding spots (hiding phase)
    - Seekers then search for hiders (seeking phase)
    - Seekers win if they tag all hiders
    - Hiders win if time runs out with at least one still hidden
    """

    name = "hide_and_seek"
    description = "Classic hide and seek with LLM agents"
    min_players = 2
    max_players = 8
    version = "1.0.0"

    # Game-specific constants
    TAG_DISTANCE = 200.0  # cm (2 meters)
    HIDING_PHASE_STEPS = 50
    SPAWN_AREA_SIZE = 1000.0  # cm

    def __init__(
        self,
        num_hiders: int = 2,
        num_seekers: int = 1,
        hiding_phase_steps: int = 50,
        model_name: str = "gpt-4o",
        **kwargs
    ):
        """
        Initialize Hide and Seek game.

        Args:
            num_hiders: Number of hider agents
            num_seekers: Number of seeker agents
            hiding_phase_steps: Steps hiders get to hide before seekers start
            model_name: LLM model to use for agents
        """
        super().__init__(**kwargs)

        self.num_hiders = num_hiders
        self.num_seekers = num_seekers
        self.hiding_phase_steps = hiding_phase_steps
        self.model_name = model_name

        # Game state
        self.hider_ids: list[str] = []
        self.seeker_ids: list[str] = []
        self.tagged_hiders: set[str] = set()
        self.is_hiding_phase: bool = True

    def get_agent_configs(self) -> list[AgentConfig]:
        """Create configurations for all hiders and seekers."""
        configs = []

        # Create hider configs
        for i in range(self.num_hiders):
            agent_id = f"hider_{i}"
            self.hider_ids.append(agent_id)

            configs.append(AgentConfig(
                agent_id=agent_id,
                role=ROLE_HIDER,
                team="hiders",
                model_name=self.model_name,
                temperature=0.8,  # More creative hiding strategies
                spawn_position=self._get_spawn_position(i, "hider"),
                prompt_config=self._create_hider_prompt_config(),
            ))

        # Create seeker configs
        for i in range(self.num_seekers):
            agent_id = f"seeker_{i}"
            self.seeker_ids.append(agent_id)

            configs.append(AgentConfig(
                agent_id=agent_id,
                role=ROLE_SEEKER,
                team="seekers",
                model_name=self.model_name,
                temperature=0.6,  # More methodical searching
                spawn_position=self._get_spawn_position(i, "seeker"),
                prompt_config=self._create_seeker_prompt_config(),
            ))

        return configs

    def _get_spawn_position(self, index: int, role: str) -> tuple[float, float]:
        """Get spawn position for an agent."""
        if role == "hider":
            # Hiders spawn spread out
            angle = (index / max(self.num_hiders, 1)) * 360
            import math
            x = self.SPAWN_AREA_SIZE * 0.3 * math.cos(math.radians(angle))
            y = self.SPAWN_AREA_SIZE * 0.3 * math.sin(math.radians(angle))
            return (x, y)
        else:
            # Seekers spawn at center
            return (0, 0)

    def _create_hider_prompt_config(self) -> AgentPromptConfig:
        """Create prompt configuration for hiders."""
        return AgentPromptConfig(
            role_description="""You are a HIDER in a game of Hide and Seek.
You are an embodied agent in a 3D city environment. Your goal is to find
a good hiding spot and stay hidden from the seekers.""",

            game_instructions="""
GAME PHASES:
1. HIDING PHASE: You have limited time to find a hiding spot. Move quickly!
2. SEEKING PHASE: Stay hidden. Seekers are actively looking for you.

STRATEGY TIPS:
- Look for objects, buildings, or structures to hide behind
- Consider line of sight - can seekers see you from common paths?
- Stay still when you hear seekers nearby
- You can peek to check if seekers are coming, but it's risky
""",

            win_condition="Survive until time runs out without being tagged.",
            lose_condition="You are tagged by a seeker (they get within 2 meters of you).",

            tools=HIDER_TOOLS,

            additional_context="""
ENVIRONMENT:
- You are in a procedurally generated city with buildings, roads, and objects
- Distance is measured in centimeters (100cm = 1 meter)
- You can see nearby objects and agents in your observations
""",

            response_format="""
Respond with a JSON object:
{
    "thought": "Your reasoning about the situation and strategy",
    "action": "The action to take (move, turn, hide, peek, wait, look_around)",
    "action_params_json": "{\\"direction\\": \\"forward\\", \\"duration\\": 3}"
}
"""
        )

    def _create_seeker_prompt_config(self) -> AgentPromptConfig:
        """Create prompt configuration for seekers."""
        return AgentPromptConfig(
            role_description="""You are a SEEKER in a game of Hide and Seek.
You are an embodied agent in a 3D city environment. Your goal is to find
and tag all the hiders before time runs out.""",

            game_instructions="""
GAME PHASES:
1. HIDING PHASE: Wait at the starting position. Hiders are finding spots.
2. SEEKING PHASE: Search the area systematically to find hiders.

STRATEGY TIPS:
- Search systematically - don't just wander randomly
- Check behind objects, buildings, and in corners
- Listen for movement clues in the game state
- Coordinate with other seekers if present (use call_out)
- When you spot a hider, close in quickly before they can escape
""",

            win_condition="Tag all hiders (get within 2 meters of each one).",
            lose_condition="Time runs out with hiders still hidden.",

            tools=SEEKER_TOOLS,

            additional_context="""
ENVIRONMENT:
- You are in a procedurally generated city with buildings, roads, and objects
- Distance is measured in centimeters (100cm = 1 meter)
- Tag distance: 200cm (2 meters)
- You can see nearby objects and agents in your observations
""",

            response_format="""
Respond with a JSON object:
{
    "thought": "Your reasoning about where to search and why",
    "action": "The action to take (move, turn, search, tag, call_out, wait, look_around)",
    "action_params_json": "{\\"direction\\": \\"forward\\", \\"duration\\": 3}"
}
"""
        )

    def setup(self) -> None:
        """Initialize game-specific state."""
        self.tagged_hiders = set()
        self.is_hiding_phase = True

        self.game_data = {
            "hiding_phase": True,
            "hiding_phase_remaining": self.hiding_phase_steps,
            "tagged_count": 0,
            "total_hiders": self.num_hiders,
        }

        logger.info(f"Hide and Seek setup complete: {self.num_hiders} hiders, {self.num_seekers} seekers")

    def on_step_start(self, step: int) -> None:
        """Handle phase transitions."""
        # Check if hiding phase should end
        if self.is_hiding_phase and step >= self.hiding_phase_steps:
            self.is_hiding_phase = False
            self.game_data["hiding_phase"] = False
            self.broadcast_message("HIDING PHASE OVER! Seekers are now searching!")
            logger.info("Hiding phase ended, seeking phase begun")

        # Update game data
        if self.is_hiding_phase:
            self.game_data["hiding_phase_remaining"] = self.hiding_phase_steps - step

    def check_win_condition(self) -> tuple[bool, str | list[str] | None, str]:
        """Check if seekers have won (all hiders tagged)."""
        if len(self.tagged_hiders) >= self.num_hiders:
            return True, "seekers", f"All {self.num_hiders} hiders have been tagged!"
        return False, None, ""

    def check_lose_condition(self) -> tuple[bool, str]:
        """Check if hiders have won (time ran out)."""
        # This is from the seekers' perspective - they "lose" if time runs out
        # But we'll report it as hiders winning
        remaining_hiders = self.num_hiders - len(self.tagged_hiders)
        if self.current_step >= self.max_steps - 1 and remaining_hiders > 0:
            # Actually return False here - we'll handle this as a WIN for hiders
            return False, ""
        return False, ""

    def build_observation(self, agent_id: str) -> AgentObservation:
        """Build observation for an agent."""
        agent = self.agents[agent_id]

        # Get position and direction
        pos = agent.humanoid.position if agent.humanoid else Vector(0, 0)
        dir = agent.humanoid.direction if agent.humanoid else Vector(1, 0)

        # Get camera image if available
        camera_image = None
        if agent.humanoid and hasattr(agent.humanoid, 'camera_id'):
            try:
                camera_image = self.communicator.get_camera_image(agent.humanoid.camera_id)
            except Exception:
                pass

        # Find nearby agents (within visibility range)
        nearby_agents = []
        visibility_range = 1000.0  # 10 meters

        for other_id, other_agent in self.agents.items():
            if other_id == agent_id:
                continue
            if not other_agent.humanoid:
                continue

            other_pos = other_agent.humanoid.position
            distance = pos.distance(other_pos)

            if distance <= visibility_range:
                # Don't reveal exact positions of hiders to seekers during hiding phase
                if self.is_hiding_phase and agent_id in self.seeker_ids:
                    continue

                nearby_agents.append({
                    "id": other_id,
                    "role": str(other_agent.config.role),
                    "distance": round(distance, 1),
                    "is_tagged": other_id in self.tagged_hiders,
                })

        # Build game state visible to this agent
        game_state = {
            "phase": "hiding" if self.is_hiding_phase else "seeking",
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "your_role": str(agent.config.role),
            "your_team": agent.config.team,
        }

        if agent_id in self.seeker_ids:
            game_state["hiders_tagged"] = len(self.tagged_hiders)
            game_state["hiders_remaining"] = self.num_hiders - len(self.tagged_hiders)
        else:
            game_state["you_are_tagged"] = agent_id in self.tagged_hiders

        # Get messages
        messages = agent.message_inbox.copy()
        agent.message_inbox.clear()

        return AgentObservation(
            position=(pos.x, pos.y),
            direction=(dir.x, dir.y),
            camera_image=camera_image,
            nearby_agents=nearby_agents,
            nearby_objects=[],  # Could be populated from simulation
            game_state=game_state,
            messages=messages,
        )

    def process_action(self, agent_id: str, action: AgentAction) -> dict[str, Any]:
        """Process an agent's action."""
        agent = self.agents[agent_id]

        # Skip if agent is tagged (hiders only)
        if agent_id in self.tagged_hiders:
            return {"success": False, "message": "You have been tagged and cannot act"}

        # Skip seeker actions during hiding phase
        if self.is_hiding_phase and agent_id in self.seeker_ids:
            return {"success": True, "message": "Waiting for hiding phase to end"}

        action_name = action.action.lower()
        params = action.action_params

        # Process based on action type
        if action_name == "move":
            return self._process_move(agent, params)
        elif action_name == "turn":
            return self._process_turn(agent, params)
        elif action_name == "tag":
            return self._process_tag(agent_id, agent, params)
        elif action_name == "hide":
            return self._process_hide(agent, params)
        elif action_name == "search":
            return self._process_search(agent_id, agent)
        elif action_name == "call_out":
            return self._process_call_out(agent_id, params)
        elif action_name in ("wait", "look_around", "peek"):
            return {"success": True, "message": f"You {action_name}"}
        else:
            return {"success": True, "message": "No action taken"}

    def _process_move(self, agent, params: dict) -> dict[str, Any]:
        """Process a move action."""
        direction = params.get("direction", "forward")
        duration = min(max(float(params.get("duration", 1)), 1), 5)

        # humanoid_step_forward(humanoid_id, duration, direction)
        # direction: 0 = forward, 1 = backward
        if direction == "forward":
            self.communicator.humanoid_step_forward(agent.humanoid.id, duration, 0)
        elif direction == "backward":
            self.communicator.humanoid_step_forward(agent.humanoid.id, duration, 1)
        # Left/right would require lateral movement or turn+move

        return {"success": True, "message": f"Moved {direction} for {duration}s"}

    def _process_turn(self, agent, params: dict) -> dict[str, Any]:
        """Process a turn action."""
        angle = min(max(float(params.get("angle", 45)), 0), 180)
        direction = params.get("direction", "right")

        self.communicator.humanoid_rotate(agent.humanoid.id, angle, direction)

        return {"success": True, "message": f"Turned {direction} {angle} degrees"}

    def _process_tag(self, agent_id: str, agent, params: dict) -> dict[str, Any]:
        """Process a tag action (seeker only)."""
        if agent_id not in self.seeker_ids:
            return {"success": False, "message": "Only seekers can tag"}

        target_id = params.get("target")
        if not target_id or target_id not in self.hider_ids:
            return {"success": False, "message": f"Invalid target: {target_id}"}

        if target_id in self.tagged_hiders:
            return {"success": False, "message": f"{target_id} is already tagged"}

        # Check distance
        distance = self.get_distance_between_agents(agent_id, target_id)
        if distance is None or distance > self.TAG_DISTANCE:
            return {"success": False, "message": f"Too far to tag (distance: {distance:.0f}cm, need: {self.TAG_DISTANCE}cm)"}

        # Tag successful!
        self.tagged_hiders.add(target_id)
        self.agents[target_id].is_alive = False
        self.game_data["tagged_count"] = len(self.tagged_hiders)

        # Award points
        agent.score += 100

        # Notify everyone
        self.broadcast_message(f"{target_id} has been tagged by {agent_id}!")
        self.send_message_to_agent(target_id, "You have been tagged! You're out.")

        logger.info(f"{agent_id} tagged {target_id}")

        return {"success": True, "message": f"Successfully tagged {target_id}!"}

    def _process_hide(self, agent, params: dict) -> dict[str, Any]:
        """Process a hide action (hider only)."""
        target = params.get("target", "nearby object")
        # In a full implementation, this would check for nearby objects
        # and position the agent appropriately
        return {"success": True, "message": f"Attempting to hide behind {target}"}

    def _process_search(self, agent_id: str, agent) -> dict[str, Any]:
        """Process a search action (seeker only)."""
        if agent_id not in self.seeker_ids:
            return {"success": False, "message": "Only seekers can search"}

        # Check if any hiders are nearby
        found = []
        pos = agent.humanoid.position

        for hider_id in self.hider_ids:
            if hider_id in self.tagged_hiders:
                continue

            hider = self.agents[hider_id]
            if hider.humanoid:
                distance = pos.distance(hider.humanoid.position)
                if distance <= self.TAG_DISTANCE * 3:  # Search range is 3x tag distance
                    found.append((hider_id, distance))

        if found:
            found_str = ", ".join(f"{h[0]} ({h[1]:.0f}cm away)" for h in found)
            return {"success": True, "message": f"Search revealed: {found_str}"}
        else:
            return {"success": True, "message": "No hiders found in immediate area"}

    def _process_call_out(self, agent_id: str, params: dict) -> dict[str, Any]:
        """Process a call_out action (communication between seekers)."""
        message = params.get("message", "")

        # Send to all other seekers
        for seeker_id in self.seeker_ids:
            if seeker_id != agent_id:
                self.send_message_to_agent(seeker_id, f"[{agent_id}]: {message}")

        return {"success": True, "message": f"Called out to teammates: {message}"}

    def on_game_end(self, result: GameResult) -> None:
        """Handle game end."""
        # If game ended due to max steps, hiders win
        if result.state == GameState.DRAW and len(self.tagged_hiders) < self.num_hiders:
            result.state = GameState.WIN
            result.winner = "hiders"
            result.reason = f"Time ran out! {self.num_hiders - len(self.tagged_hiders)} hider(s) survived."

            # Award points to surviving hiders
            for hider_id in self.hider_ids:
                if hider_id not in self.tagged_hiders:
                    self.agents[hider_id].score += 100

        logger.info(f"Game ended: {result.state.value} - {result.reason}")
