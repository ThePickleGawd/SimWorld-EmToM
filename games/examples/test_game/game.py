"""
Scripted test game for SimWorld.

Spawns a single humanoid agent that walks toward a box, pauses so the pickup
is clearly visible, and records both first- and third-person views for
verification.
"""

import json
import logging
import math
import time
from typing import Any

from simworld.utils.vector import Vector

from games.base import BaseGame, CameraStream, GameAgent, GameResult, GameState
from games.agent_config import (
    AgentAction,
    AgentConfig,
    AgentObservation,
    AgentPromptConfig,
    AgentTool,
)
from games.registry import register_game

logger = logging.getLogger(__name__)

# Simple tool set exposed in prompts (kept minimal because actions are scripted)
MOVE_TOOL = AgentTool(
    name="move",
    description="Walk forward or backward for a few seconds",
    parameters={"direction": "forward or backward", "duration": "seconds to walk"},
)
TURN_TOOL = AgentTool(
    name="turn",
    description="Rotate left or right",
    parameters={"angle": "degrees to rotate", "direction": "left or right"},
)
PICK_UP_TOOL = AgentTool(
    name="pick_up",
    description="Pick up the shared prop in front of you",
    parameters={"object": "name of the object to pick (defaults to shared prop)"},
)
WAVE_TOOL = AgentTool(
    name="wave",
    description="Play a wave animation for a quick visual check",
    parameters={},
)
WAIT_TOOL = AgentTool(
    name="wait",
    description="Do nothing for a short pause",
    parameters={"duration": "seconds to wait"},
)


@register_game
class TestGame(BaseGame):
    """
    Minimal scripted game for sanity checks.

    A single agent walks toward a box, pauses to make the pickup visible,
    and both first-person and trailing third-person cameras are recorded.
    """

    name = "test_game"
    description = "Scripted demo: one agent walks to a box and picks it up (1st + 3rd person cameras)."
    min_players = 1
    max_players = 1
    version = "0.2.0"

    def __init__(
        self,
        object_model_path: str = "/Game/InteractableAsset/Box/BP_Interactable_Box.BP_Interactable_Box_C",
        object_name: str = "TEST_BP_BOX",
        center_x: float = 0.0,
        center_y: float = 0.0,
        object_height: float = 20.0,
        agent_offset: float = 300.0,
        scripted_step_seconds: float = 2.0,
        post_pickup_hold: float = 3.0,
        max_steps: int = 20,
        third_person_cam_id: int = 0,
        third_person_distance: float = 600.0,
        third_person_height: float = 350.0,
        third_person_pitch: float = -20.0,
        third_person_yaw_offset: float = 0.0,
        third_person_side_offset: float = 150.0,
        third_person_fov: float = 90.0,
        third_person_resolution: tuple[int, int] = (1280, 720),
        verbose: bool = False,
        unrealcv=None,
        communicator=None,
        **kwargs,
    ):
        super().__init__(
            unrealcv=unrealcv,
            communicator=communicator,
            max_steps=max_steps,
            verbose=verbose,
        )
        self.object_model_path = object_model_path
        self.object_name = object_name
        self.object_height = float(object_height)
        self.center = Vector(float(center_x), float(center_y))
        self.agent_offset = float(agent_offset)
        self.scripted_step_seconds = float(scripted_step_seconds)
        self.post_pickup_hold = float(post_pickup_hold)
        self.third_person_cam_id = int(third_person_cam_id)
        self.third_person_distance = float(third_person_distance)
        self.third_person_height = float(third_person_height)
        self.third_person_pitch = float(third_person_pitch)
        self.third_person_yaw_offset = float(third_person_yaw_offset)
        self.third_person_side_offset = float(third_person_side_offset)
        self.third_person_fov = float(third_person_fov)
        self.third_person_resolution = tuple(third_person_resolution)

        # Scripted action tracking
        self.script_actions: dict[str, list[tuple[str, dict[str, Any], str]]] = {}
        self.script_progress: dict[str, int] = {}
        self.object_picked_by: str | None = None

        # IDs
        self.player_agent_id = "agent_main"
        self.third_person_agent_id = "third_person_camera"

    # ------------------------------------------------------------------
    # BaseGame protocol implementations
    # ------------------------------------------------------------------
    def initialize(self) -> bool:
        """Spawn the player agent and set up the trailing camera."""
        if not self._connected:
            logger.error("Not connected to simulation. Call connect() first.")
            return False

        # Clear any existing state
        self.agents.clear()
        self.current_step = 0
        self.game_state = GameState.NOT_STARTED
        self.game_data.clear()

        # Spawn player
        for config in self.get_agent_configs():
            if not self._spawn_agent(config):
                logger.error(f"Failed to spawn agent {config.agent_id}")
                return False

        # Configure third-person camera agent (no humanoid)
        cam_config = AgentConfig(
            agent_id=self.third_person_agent_id,
            role="camera",
            team="demo",
            model_name="none",
            prompt_config=AgentPromptConfig(
                role_description="Third-person trailing camera for recording.",
                tools=[],
                response_format="{}",
            ),
            spawn_position=(self.center.x, self.center.y),
        )
        self.agents[self.third_person_agent_id] = GameAgent(
            config=cam_config,
            humanoid=None,
            llm=None,
        )
        self._setup_third_person_camera()

        # Run game-specific setup
        self.setup()

        logger.info(f"Game '{self.name}' initialized with {len(self.agents)} agents (including camera)")
        return True

    def get_agent_configs(self) -> list[AgentConfig]:
        """Create a single agent aimed at the shared box."""
        prompt = AgentPromptConfig(
            role_description="You are a scripted test agent in a simple demo.",
            game_instructions="Follow the pre-written choreography to help validate the simulation.",
            tools=[MOVE_TOOL, TURN_TOOL, PICK_UP_TOOL, WAVE_TOOL, WAIT_TOOL],
            win_condition="Complete the short scripted sequence.",
            lose_condition="No lose condition; the sequence simply ends.",
            response_format="""
Respond with a JSON object:
{
  "thought": "What you are about to do",
  "action": "one of: move, turn, pick_up, wave, wait",
  "action_params_json": "{\"duration\": 1.5}"
}
""",
        )

        spawn = (self.center.x - self.agent_offset, self.center.y)

        return [
            AgentConfig(
                agent_id=self.player_agent_id,
                role="tester",
                team="demo",
                model_name="gpt-5-mini",
                temperature=0.1,
                spawn_position=spawn,
                spawn_direction=(1, 0),  # Face the box
                prompt_config=prompt,
            ),
        ]

    def setup(self) -> None:
        """Spawn the shared prop and prepare the scripted action list."""
        self._spawn_prop()
        self._build_script()

        self.game_data = {
            "object_name": self.object_name,
            "object_model_path": self.object_model_path,
            "object_picked_by": self.object_picked_by,
            "script_progress": self.script_progress.copy(),
        }

        logger.info("Test game setup complete: one agent, a box, and a trailing camera are ready.")

    def check_win_condition(self) -> tuple[bool, str | list[str] | None, str]:
        scripts_done = all(
            self.script_progress.get(agent_id, 0) >= len(actions)
            for agent_id, actions in self.script_actions.items()
        )
        if scripts_done:
            reason = "Scripted sequence finished."
            if self.object_picked_by:
                reason += f" {self.object_picked_by} picked up {self.object_name}."
            return True, "demo", reason

        return False, None, ""

    def check_lose_condition(self) -> tuple[bool, str]:
        # This is a demonstration; there is no lose condition.
        return False, ""

    def build_observation(self, agent_id: str) -> AgentObservation:
        """Return a light-weight observation with camera frames for recording."""
        self._refresh_agent_poses()

        agent = self.agents[agent_id]
        pos = agent.humanoid.position if agent.humanoid else Vector(0, 0)
        direction = agent.humanoid.direction if agent.humanoid else Vector(1, 0)

        camera_image = None
        if agent.humanoid and hasattr(agent.humanoid, "camera_id"):
            try:
                camera_image = self.communicator.get_camera_observation(
                    agent.humanoid.camera_id,
                    viewmode="lit",
                    mode="direct",
                )
            except Exception as exc:
                if self.verbose:
                    logger.warning(f"Camera capture failed for {agent_id}: {exc}")

        observation_state = {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "object_picked_by": self.object_picked_by,
        }

        return AgentObservation(
            position=(pos.x, pos.y),
            direction=(direction.x, direction.y),
            camera_image=camera_image,
            nearby_agents=[],
            nearby_objects=[{"name": self.object_name}],
            game_state=observation_state,
            messages=[],
        )

    def process_action(self, agent_id: str, action: AgentAction) -> dict[str, Any]:
        """Execute the scripted action on the humanoid."""
        agent = self.agents[agent_id]
        action_name = action.action.lower()
        params = action.action_params

        if action_name == "move":
            return self._process_move(agent, params)
        if action_name == "turn":
            return self._process_turn(agent, params)
        if action_name == "pick_up":
            return self._process_pick_up(agent_id, agent, params)
        if action_name == "wave":
            return self._process_wave(agent)
        if action_name == "wait":
            duration = float(params.get("duration", 0.5))
            time.sleep(max(duration, 0))
            return {"success": True, "message": f"Waited {duration:.1f}s"}

        return {"success": True, "message": "No action executed"}

    def on_game_end(self, result: GameResult) -> None:
        logger.info(f"Test game finished with state={result.state} reason='{result.reason}'")

    # Override the base step loop so the third-person camera can record without actions.
    def _run_step(self) -> GameResult | None:
        if self.verbose:
            logger.info(f"=== Step {self.current_step} ===")
        self.on_step_start(self.current_step)

        # End conditions before acting
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

        observations: dict[str, AgentObservation] = {}
        actions: dict[str, AgentAction] = {}
        results: dict[str, dict[str, Any]] = {}

        # Player agent acts
        player_id = self.player_agent_id
        if player_id in self.agents:
            observation = self.build_observation(player_id)
            observations[player_id] = observation

            action = self._get_agent_action(player_id, observation)
            actions[player_id] = action

            result = self.process_action(player_id, action)
            results[player_id] = result

            self.agents[player_id].action_history.append(action)
            self.on_agent_action(player_id, action, result)

        # Third-person camera captures frame (no action)
        camera_obs = self._build_third_person_observation()
        observations[self.third_person_agent_id] = camera_obs
        actions[self.third_person_agent_id] = AgentAction(
            thought="Capture third-person frame",
            action="wait",
            action_params_json=json.dumps({"duration": 0}),
        )
        results[self.third_person_agent_id] = {"success": True, "message": "Captured third-person frame"}

        # Record this step if recorder is attached
        if hasattr(self, "_recorder") and self._recorder:
            self._recorder.record_step(
                step=self.current_step,
                observations=observations,
                actions=actions,
                results=results,
            )

        self.on_step_end(self.current_step)
        return None

    # ------------------------------------------------------------------
    # Scripted control
    # ------------------------------------------------------------------
    def _build_script(self) -> None:
        """Create per-agent action queues."""
        self.script_actions = {
            self.player_agent_id: [
                ("move", {"direction": "forward", "duration": self.scripted_step_seconds},
                 "Walk toward the box."),
                ("wait", {"duration": 1.0},
                 "Pause so the approach is visible."),
                ("pick_up", {"object": self.object_name},
                 "Grab the box for a clear interaction test."),
                ("wait", {"duration": self.post_pickup_hold},
                 "Hold the box so it stays visible."),
                ("turn", {"direction": "left", "angle": 45},
                 "Turn slightly while holding the box."),
            ],
        }
        self.script_progress = {agent_id: 0 for agent_id in self.script_actions}

    def _get_agent_action(self, agent_id: str, observation: AgentObservation) -> AgentAction:
        """
        Override to provide deterministic, scripted actions instead of using an LLM.
        """
        actions = self.script_actions.get(agent_id, [])
        idx = self.script_progress.get(agent_id, 0)

        if idx < len(actions):
            name, params, thought = actions[idx]
            self.script_progress[agent_id] = idx + 1
            self.game_data["script_progress"] = self.script_progress.copy()
            return AgentAction(
                thought=thought,
                action=name,
                action_params_json=json.dumps(params),
            )

        # Fallback to idle once the script is done
        return AgentAction(
            thought="Script complete; waiting.",
            action="wait",
            action_params_json=json.dumps({"duration": 0.5}),
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------
    def _process_move(self, agent, params: dict) -> dict[str, Any]:
        direction = params.get("direction", "forward")
        duration = max(float(params.get("duration", 1.0)), 0.2)

        if direction == "forward":
            self.communicator.humanoid_step_forward(agent.humanoid.id, duration, 0)
        elif direction == "backward":
            self.communicator.humanoid_step_forward(agent.humanoid.id, duration, 1)
        else:
            return {"success": False, "message": f"Unknown move direction '{direction}'"}

        return {"success": True, "message": f"Moved {direction} for {duration:.1f}s"}

    def _process_turn(self, agent, params: dict) -> dict[str, Any]:
        angle = min(max(float(params.get("angle", 45.0)), 0.0), 180.0)
        direction = params.get("direction", "right")

        self.communicator.humanoid_rotate(agent.humanoid.id, angle, direction)
        time.sleep(1.0)

        return {"success": True, "message": f"Turned {direction} {angle:.0f} degrees"}

    def _process_pick_up(self, agent_id: str, agent, params: dict) -> dict[str, Any]:
        target_object = params.get("object", self.object_name)
        success = self.communicator.humanoid_pick_up_object(agent.humanoid.id, target_object)

        if success:
            self.object_picked_by = agent_id
            self.game_data["object_picked_by"] = agent_id
            return {"success": True, "message": f"Picked up {target_object}"}

        return {"success": False, "message": f"Failed to pick up {target_object}"}

    def _process_wave(self, agent) -> dict[str, Any]:
        humanoid_name = self.communicator.get_humanoid_name(agent.humanoid.id)
        try:
            self.communicator.unrealcv.humanoid_wave_to_dog(humanoid_name)
            time.sleep(1.0)
            self.communicator.unrealcv.humanoid_stop_current_action(humanoid_name)
            return {"success": True, "message": "Wave animation played"}
        except Exception as exc:
            return {"success": False, "message": f"Wave failed: {exc}"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _spawn_prop(self) -> None:
        """Spawn the box in the center of the scene."""
        location = (self.center.x, self.center.y, self.object_height)
        self.communicator.spawn_object(
            self.object_name,
            self.object_model_path,
            position=location,
            direction=(0, 0, 0),
        )
        logger.info(f"Spawned prop '{self.object_name}' at {location}")

    def _setup_third_person_camera(self) -> None:
        """Configure the trailing third-person camera."""
        try:
            cam = self.communicator.unrealcv
            cam.set_camera_resolution(self.third_person_cam_id, self.third_person_resolution)
            cam.set_camera_fov(self.third_person_cam_id, self.third_person_fov)
            # Initial placement will be updated once agent pose is refreshed
            self._update_third_person_camera_position()
        except Exception as exc:
            logger.warning(f"Failed to configure third-person camera: {exc}")

    def _refresh_agent_poses(self) -> None:
        """Sync local poses with the simulator to keep logs accurate."""
        try:
            humanoid_ids = [
                agent.humanoid.id for agent in self.agents.values() if agent.humanoid
            ]
            if not humanoid_ids:
                return

            poses = self.communicator.get_position_and_direction(
                humanoid_ids=humanoid_ids
            )

            for agent_id, agent in self.agents.items():
                if not agent.humanoid:
                    continue
                key = ("humanoid", agent.humanoid.id)
                if key in poses:
                    pos, yaw = poses[key]
                    agent.humanoid.position = pos
                    agent.humanoid.direction = yaw
        except Exception as exc:
            if self.verbose:
                logger.warning(f"Failed to refresh agent poses: {exc}")

    def _update_third_person_camera_position(self) -> None:
        """Place the trailing camera behind and above the player."""
        player = self.agents.get(self.player_agent_id)
        if not player or not player.humanoid:
            return

        pos = player.humanoid.position
        dir_vec = player.humanoid.direction.normalize()
        if dir_vec.length() == 0:
            dir_vec = Vector(1.0, 0.0)
        # Offset slightly to the side for a clearer view
        perpendicular = Vector(-dir_vec.y, dir_vec.x).normalize()
        offset = dir_vec * (-self.third_person_distance) + perpendicular * self.third_person_side_offset
        cam_pos = Vector(pos.x + offset.x, pos.y + offset.y)

        # Look at the agent
        look_vec = Vector(pos.x - cam_pos.x, pos.y - cam_pos.y)
        yaw = math.degrees(math.atan2(look_vec.y, look_vec.x)) + self.third_person_yaw_offset

        horizontal_dist = math.sqrt(look_vec.x ** 2 + look_vec.y ** 2) + 1e-3
        pitch = -math.degrees(math.atan2(self.third_person_height, horizontal_dist))
        # Blend toward configured pitch to keep a consistent downward angle
        pitch = 0.7 * pitch + 0.3 * self.third_person_pitch

        try:
            cam = self.communicator.unrealcv
            cam.set_camera_location(self.third_person_cam_id, (cam_pos.x, cam_pos.y, self.third_person_height))
            cam.set_camera_rotation(self.third_person_cam_id, (pitch, yaw, 0))
        except Exception as exc:
            if self.verbose:
                logger.warning(f"Failed to move third-person camera: {exc}")

    def _build_third_person_observation(self) -> AgentObservation:
        """Capture a frame from the trailing camera for recording."""
        self._update_third_person_camera_position()
        image = None
        try:
            image = self.communicator.get_camera_observation(
                self.third_person_cam_id,
                viewmode="lit",
                mode="direct",
            )
        except Exception as exc:
            if self.verbose:
                logger.warning(f"Third-person camera capture failed: {exc}")

        return AgentObservation(
            position=(0.0, 0.0),
            direction=(0.0, 1.0),
            camera_image=image,
            nearby_agents=[],
            nearby_objects=[],
            game_state={
                "camera": "third_person",
                "current_step": self.current_step,
            },
            messages=[],
        )

    def get_camera_streams(self):
        """Expose both first- and third-person streams for recording."""
        streams = super().get_camera_streams()
        streams.append(CameraStream(
            agent_id=self.third_person_agent_id,
            label="third_person",
            camera_id=self.third_person_cam_id,
            fetch_fn=self._fetch_third_person_frame,
        ))
        return streams

    def _fetch_third_person_frame(self):
        """Capture a third-person frame for continuous video."""
        self._update_third_person_camera_position()
        return self.communicator.get_camera_observation(
            self.third_person_cam_id,
            viewmode="lit",
            mode="direct",
        )
