#!/usr/bin/env python3
"""
Convert recorded frames into videos using ffmpeg.

Usage:
    python -m games.process_frames /path/to/session --fps 30 --keep-frames

The script looks for agent frame folders under:
    <session>/agents/<agent_id>/frames/*.jpg
and writes:
    <session>/agents/<agent_id>/first_person.mp4
"""

import argparse
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> bool:
    """Encode a directory of sequential JPG frames into an MP4."""
    if not frames_dir.exists():
        logger.warning(f"Frames directory not found: {frames_dir}")
        return False

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        logger.warning(f"No frames found in {frames_dir}")
        return False

    input_pattern = str(frames_dir / "%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(output_path),
    ]

    logger.info(f"Encoding {len(frame_files)} frames from {frames_dir} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed for {frames_dir}: {result.stderr}")
        return False
    return True


def process_session(session_dir: Path, fps: int, keep_frames: bool) -> None:
    """Process all agent frame folders in a session directory."""
    agents_root = session_dir / "agents"
    if not agents_root.exists():
        logger.error(f"No agents/ directory found in session: {session_dir}")
        return

    for agent_dir in sorted(agents_root.iterdir()):
        if not agent_dir.is_dir():
            continue
        frames_dir = agent_dir / "frames"
        output_path = agent_dir / "first_person.mp4"

        success = encode_video(frames_dir, output_path, fps)
        if success:
            logger.info(f"Encoded video for {agent_dir.name}: {output_path}")
            if not keep_frames:
                # Remove raw frames after successful encode
                for frame_file in frames_dir.glob("*.jpg"):
                    frame_file.unlink()
                try:
                    frames_dir.rmdir()
                except OSError:
                    pass  # Directory not empty; ignore
        else:
            logger.warning(f"Skipping cleanup for {agent_dir.name} due to encode failure")


def main():
    parser = argparse.ArgumentParser(description="Encode recorded frames into videos.")
    parser.add_argument(
        "session",
        nargs="?",
        default=None,
        help="Path to a recording session directory (default: most recent under results/)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS (default: 30)")
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep raw frames after encoding (default: delete)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Resolve session directory (default to most recent under results/)
    if args.session:
        session_dir = Path(args.session).expanduser().resolve()
    else:
        results_root = Path("results")
        if not results_root.exists():
            logger.error("No session specified and 'results/' not found.")
            return
        sessions = sorted(
            [p for p in results_root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not sessions:
            logger.error("No session specified and no session directories found in results/.")
            return
        session_dir = sessions[0]
        logger.info(f"No session specified. Using most recent: {session_dir}")

    if not session_dir.exists():
        logger.error(f"Session directory does not exist: {session_dir}")
        return

    process_session(session_dir, args.fps, args.keep_frames)


if __name__ == "__main__":
    main()
