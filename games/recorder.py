"""
Game recording system for capturing agent POVs, reasoning logs, and game state.
"""

import json
import logging
import os
import shutil
from games.process_frames import encode_video
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING, Tuple
import threading
import time

import numpy as np

if TYPE_CHECKING:
    from games.base import BaseGame, GameResult, CameraStream
    from games.agent_config import AgentAction, AgentObservation

logger = logging.getLogger(__name__)


class GameRecorder:
    """
    Records game sessions including:
    - First-person video from each agent's camera
    - Reasoning logs (thoughts, actions, results)
    - Game state history
    - Top-down overview (if available)
    """

    def __init__(
        self,
        game: "BaseGame",
        output_dir: str | Path = "results",
        video_fps: int = 30,
        enable_live_preview: bool = False,
        keep_frames: bool = False,
    ):
        """
        Initialize the recorder.

        Args:
            game: The game instance to record
            output_dir: Base directory for results
            video_fps: Target FPS for output videos
            enable_live_preview: Show live preview window (requires display)
            keep_frames: Keep raw frames after video encoding
        """
        self.game = game
        self.base_output_dir = Path(output_dir)
        self.video_fps = video_fps
        self.enable_live_preview = enable_live_preview
        self.keep_frames = keep_frames

        # Session directory (created on start)
        self.session_dir: Path | None = None
        self.agents_dir: Path | None = None

        # Frame storage per agent
        self.frame_counts: dict[str, int] = {}
        self.frame_dirs: dict[Tuple[str, str], Path] = {}
        self.streams: list["CameraStream"] = []
        self.stream_frame_counts: dict[Tuple[str, str], int] = {}
        self.step_frame_dirs: dict[str, Path] = {}

        # Log file handles
        self.reasoning_logs: dict[str, Any] = {}
        self.events_log: Any = None
        self.game_log: Any = None

        # State
        self._started = False
        self._finalized = False
        self._stop_capture = False
        self._capture_thread: threading.Thread | None = None

        # Live preview window (opencv)
        self._preview_window = None

    def start(self) -> Path:
        """
        Start recording session. Creates directory structure.

        Returns:
            Path to the session directory
        """
        if self._started:
            return self.session_dir

        # Create session directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_name = f"{timestamp}_{self.game.name}"
        self.session_dir = self.base_output_dir / session_name
        self.agents_dir = self.session_dir / "agents"

        # Create directories
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(exist_ok=True)

        # Collect camera streams (first/third-person)
        try:
            self.streams = self.game.get_camera_streams()
        except Exception:
            self.streams = []

        # Create per-agent directories and logs
        for agent_id in self.game.agents:
            agent_dir = self.agents_dir / agent_id
            agent_dir.mkdir(exist_ok=True)

            # Step-level observation frames (one per LLM step)
            step_dir = agent_dir / "step_observations"
            step_dir.mkdir(exist_ok=True)
            self.step_frame_dirs[agent_id] = step_dir

            # Stream-specific frame dirs
            for stream in [s for s in self.streams if s.agent_id == agent_id]:
                stream_dir = agent_dir / stream.label
                stream_dir.mkdir(exist_ok=True)
                frames_dir = stream_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                key = (agent_id, stream.label)
                self.frame_dirs[key] = frames_dir
                self.stream_frame_counts[key] = 0

            # Open reasoning log
            log_path = agent_dir / "reasoning.log"
            self.reasoning_logs[agent_id] = open(log_path, "w")

        # Open game event log
        self.events_log = open(self.session_dir / "events.log", "w")
        self.game_log = open(self.session_dir / "game_log.jsonl", "w")

        # Write initial game info
        self._log_event(f"Game started: {self.game.name}")
        self._log_event(f"Agents: {list(self.game.agents.keys())}")

        # Start continuous capture thread for video FPS
        if self.streams:
            self._stop_capture = False
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

        self._started = True
        logger.info(f"Recording session started: {self.session_dir}")

        return self.session_dir

    # ------------------------------------------------------------------ #
    # Continuous capture for smooth video (decoupled from LLM steps)
    # ------------------------------------------------------------------ #
    def _capture_loop(self) -> None:
        """Capture frames from all streams at a fixed FPS."""
        interval = 1.0 / max(self.video_fps, 1)
        while not self._stop_capture:
            start = time.time()
            try:
                self._capture_all_streams()
            except Exception as e:
                logger.debug(f"Capture loop error: {e}")
            elapsed = time.time() - start
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    def _capture_all_streams(self) -> None:
        """Capture one frame from each configured stream."""
        for stream in self.streams:
            image = None
            try:
                if stream.fetch_fn:
                    image = stream.fetch_fn()
                elif stream.camera_id is not None:
                    image = self.game.communicator.get_camera_observation(
                        stream.camera_id,
                        viewmode=stream.viewmode,
                        mode=stream.mode,
                    )
            except Exception as exc:
                logger.debug(f"Capture failed for {stream.agent_id}/{stream.label}: {exc}")
                continue

            if image is not None:
                self._save_stream_frame(stream.agent_id, stream.label, image)

    def record_step(
        self,
        step: int,
        observations: dict[str, "AgentObservation"],
        actions: dict[str, "AgentAction"],
        results: dict[str, dict[str, Any]],
    ) -> None:
        """
        Record a single game step.

        Args:
            step: Current step number
            observations: Observations for each agent
            actions: Actions taken by each agent
            results: Results of each action
        """
        if not self._started:
            self.start()

        # Record each agent's frame and reasoning
        for agent_id in self.game.agents:
            obs = observations.get(agent_id)
            action = actions.get(agent_id)
            result = results.get(agent_id, {})

            # Save per-step observation frame (what the agent saw this step)
            if obs and obs.camera_image is not None:
                self._save_step_frame(agent_id, step, obs.camera_image)

            # Write reasoning log
            if action:
                self._write_reasoning(agent_id, step, obs, action, result)

        # Write game state to JSONL
        state = {
            "step": step,
            "game_state": self.game.game_data.copy(),
            "agent_positions": {
                agent_id: {
                    "position": [
                        agent.humanoid.position.x,
                        agent.humanoid.position.y,
                    ] if agent.humanoid and agent.humanoid.position else None,
                    "direction": [
                        agent.humanoid.direction.x,
                        agent.humanoid.direction.y,
                    ] if agent.humanoid and agent.humanoid.direction else None,
                }
                for agent_id, agent in self.game.agents.items()
            },
        }
        self.game_log.write(json.dumps(state) + "\n")
        self.game_log.flush()

        # Live preview
        if self.enable_live_preview:
            self._update_preview(observations)

    def record_event(self, event: str) -> None:
        """Record a game event."""
        self._log_event(event)

    def finalize(self, result: "GameResult" = None) -> None:
        """
        Finalize recording session. Encodes videos and writes summary.

        Args:
            result: Final game result (optional)
        """
        if self._finalized or not self._started:
            return

        logger.info("Finalizing recording session...")

        # Close log files
        for log in self.reasoning_logs.values():
            log.close()
        if self.events_log:
            self.events_log.close()
        if self.game_log:
            self.game_log.close()

        # Write game summary
        if result:
            summary = {
                "game": self.game.name,
                "result": result.state.value,
                "winner": result.winner,
                "reason": result.reason,
                "total_steps": result.total_steps,
                "final_scores": result.final_scores,
                "agents": list(self.game.agents.keys()),
            }
            with open(self.session_dir / "game_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Close preview window
        if self._preview_window:
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass

        # Stop capture thread
        self._stop_capture = True
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)

        # Encode videos per stream
        if self.agents_dir:
            for key, frames_dir in self.frame_dirs.items():
                agent_id, label = key
                output_path = self.agents_dir / agent_id / f"{label}.mp4"
                encode_video(frames_dir, output_path, self.video_fps)
                if not self.keep_frames:
                    for frame_file in frames_dir.glob("*.jpg"):
                        frame_file.unlink()
                    try:
                        frames_dir.rmdir()
                    except OSError:
                        pass

        self._finalized = True
        logger.info(f"Recording finalized: {self.session_dir}")

    def _save_stream_frame(self, agent_id: str, label: str, image: np.ndarray) -> None:
        """Save a continuous-stream frame as JPEG."""
        key = (agent_id, label)
        if key not in self.frame_dirs:
            return

        frame_num = self.stream_frame_counts.get(key, 0)
        frame_path = self.frame_dirs[key] / f"{frame_num:06d}.jpg"

        try:
            # Handle different image formats
            if isinstance(image, np.ndarray):
                import cv2
                # Convert RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), image)
            else:
                # Assume it's already bytes or path
                with open(frame_path, "wb") as f:
                    f.write(image)

            self.stream_frame_counts[key] = frame_num + 1

        except Exception as e:
            logger.warning(f"Failed to save frame for {agent_id}/{label}: {e}")

    def _save_step_frame(self, agent_id: str, step: int, image: np.ndarray) -> None:
        """Save the observation frame associated with a specific LLM step."""
        step_dir = self.step_frame_dirs.get(agent_id)
        if not step_dir:
            return

        frame_path = step_dir / f"{step:06d}.jpg"

        try:
            if isinstance(image, np.ndarray):
                import cv2
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), image)
            else:
                with open(frame_path, "wb") as f:
                    f.write(image)
        except Exception as e:
            logger.warning(f"Failed to save step frame for {agent_id}: {e}")

    def _write_reasoning(
        self,
        agent_id: str,
        step: int,
        obs: "AgentObservation",
        action: "AgentAction",
        result: dict,
    ) -> None:
        """Write reasoning entry to agent's log."""
        log = self.reasoning_logs.get(agent_id)
        if not log:
            return

        lines = [
            f"\n{'='*60}",
            f"=== Step {step} ===",
            f"{'='*60}",
        ]

        if obs:
            lines.append(f"Position: ({obs.position[0]:.1f}, {obs.position[1]:.1f})")
            lines.append(f"Direction: ({obs.direction[0]:.2f}, {obs.direction[1]:.2f})")

            if obs.nearby_agents:
                lines.append(f"Nearby agents: {obs.nearby_agents}")

            if obs.game_state:
                lines.append(f"Game state: {obs.game_state}")

            if obs.messages:
                lines.append(f"Messages: {obs.messages}")

        lines.append("")
        lines.append(f"[THOUGHT] {action.thought}")
        lines.append(f"[ACTION] {action.action}")
        lines.append(f"[PARAMS] {action.action_params}")
        lines.append(f"[RESULT] {result.get('success', '?')} - {result.get('message', '')}")

        log.write("\n".join(lines) + "\n")
        log.flush()

    def _log_event(self, event: str) -> None:
        """Log a game event with timestamp."""
        if self.events_log:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.events_log.write(f"[{timestamp}] {event}\n")
            self.events_log.flush()

    def _update_preview(self, observations: dict[str, "AgentObservation"]) -> None:
        """Update live preview window."""
        try:
            import cv2

            # Combine all agent views into a grid
            images = []
            for agent_id, obs in observations.items():
                if obs and obs.camera_image is not None:
                    img = obs.camera_image
                    if isinstance(img, np.ndarray):
                        # Add agent label
                        img = img.copy()
                        cv2.putText(
                            img, agent_id, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )
                        images.append(img)

            if images:
                # Simple horizontal stack
                combined = np.hstack(images) if len(images) > 1 else images[0]
                # Convert RGB to BGR for display
                if len(combined.shape) == 3 and combined.shape[2] == 3:
                    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                cv2.imshow("Game Preview", combined)
                cv2.waitKey(1)
                self._preview_window = True

        except ImportError:
            pass  # OpenCV not available
        except Exception as e:
            logger.debug(f"Preview error: {e}")
