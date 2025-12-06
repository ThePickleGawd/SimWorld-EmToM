"""
Game recording system for capturing agent POVs, reasoning logs, and game state.
"""

import atexit
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from games.base import BaseGame, GameResult
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
        self.frame_dirs: dict[str, Path] = {}

        # Log file handles
        self.reasoning_logs: dict[str, Any] = {}
        self.events_log: Any = None
        self.game_log: Any = None

        # State
        self._started = False
        self._finalized = False

        # Live preview window (opencv)
        self._preview_window = None

        # Register cleanup handlers
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        atexit.register(self._cleanup)

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

        # Create per-agent directories and logs
        for agent_id in self.game.agents:
            agent_dir = self.agents_dir / agent_id
            agent_dir.mkdir(exist_ok=True)

            frames_dir = agent_dir / "frames"
            frames_dir.mkdir(exist_ok=True)

            self.frame_dirs[agent_id] = frames_dir
            self.frame_counts[agent_id] = 0

            # Open reasoning log
            log_path = agent_dir / "reasoning.log"
            self.reasoning_logs[agent_id] = open(log_path, "w")

        # Open game event log
        self.events_log = open(self.session_dir / "events.log", "w")
        self.game_log = open(self.session_dir / "game_log.jsonl", "w")

        # Write initial game info
        self._log_event(f"Game started: {self.game.name}")
        self._log_event(f"Agents: {list(self.game.agents.keys())}")

        self._started = True
        logger.info(f"Recording session started: {self.session_dir}")

        return self.session_dir

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

            # Save camera frame
            if obs and obs.camera_image is not None:
                self._save_frame(agent_id, obs.camera_image)

            # Write reasoning log
            if action:
                self._write_reasoning(agent_id, step, obs, action, result)

        # Write game state to JSONL
        state = {
            "step": step,
            "game_state": self.game.game_data.copy(),
            "agent_positions": {
                agent_id: {
                    "position": list(agent.humanoid.position) if agent.humanoid else None,
                    "direction": list(agent.humanoid.direction) if agent.humanoid else None,
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

        # Encode videos for each agent
        for agent_id, frames_dir in self.frame_dirs.items():
            frame_count = self.frame_counts[agent_id]
            if frame_count > 0:
                video_path = self.agents_dir / agent_id / "first_person.mp4"
                self._encode_video(frames_dir, video_path, frame_count)

                # Delete frames if video created successfully
                if video_path.exists() and not self.keep_frames:
                    shutil.rmtree(frames_dir)
                    logger.debug(f"Deleted frames for {agent_id}")

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

        self._finalized = True
        logger.info(f"Recording finalized: {self.session_dir}")

    def _save_frame(self, agent_id: str, image: np.ndarray) -> None:
        """Save a camera frame as JPEG."""
        frame_num = self.frame_counts[agent_id]
        frame_path = self.frame_dirs[agent_id] / f"{frame_num:06d}.jpg"

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

            self.frame_counts[agent_id] = frame_num + 1

        except Exception as e:
            logger.warning(f"Failed to save frame for {agent_id}: {e}")

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

    def _encode_video(self, frames_dir: Path, output_path: Path, frame_count: int) -> bool:
        """Encode frames to MP4 using ffmpeg."""
        if frame_count == 0:
            return False

        # Build ffmpeg command
        input_pattern = str(frames_dir / "%06d.jpg")
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(self.video_fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",  # Quality (lower = better)
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Encoded video: {output_path} ({frame_count} frames)")
                return True
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out")
            return False
        except FileNotFoundError:
            logger.error("ffmpeg not found - install it to encode videos")
            return False

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

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle Ctrl+C - finalize recording before exit."""
        logger.info("\nInterrupt received - saving recording...")
        self.finalize()

        # Restore original handler and re-raise
        signal.signal(signal.SIGINT, self._original_sigint)
        sys.exit(0)

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        if self._started and not self._finalized:
            self.finalize()
