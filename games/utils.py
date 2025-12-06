"""
Utility functions for SimWorld games.
"""

import math
from typing import Any

from simworld.utils.vector import Vector


def calculate_angle_between(from_pos: Vector, to_pos: Vector) -> float:
    """Calculate angle in degrees from one position to another."""
    dx = to_pos.x - from_pos.x
    dy = to_pos.y - from_pos.y
    return math.degrees(math.atan2(dy, dx))


def calculate_relative_angle(
    agent_pos: Vector,
    agent_direction: Vector,
    target_pos: Vector
) -> float:
    """
    Calculate relative angle to target from agent's perspective.

    Returns:
        Angle in degrees (-180 to 180)
        Negative = target is to the left
        Positive = target is to the right
    """
    # Angle to target
    target_angle = calculate_angle_between(agent_pos, target_pos)

    # Agent's facing angle
    facing_angle = math.degrees(math.atan2(agent_direction.y, agent_direction.x))

    # Relative angle
    relative = target_angle - facing_angle

    # Normalize to -180 to 180
    while relative > 180:
        relative -= 360
    while relative < -180:
        relative += 360

    return relative


def is_in_view(
    agent_pos: Vector,
    agent_direction: Vector,
    target_pos: Vector,
    fov_degrees: float = 90.0,
    max_distance: float = float('inf')
) -> bool:
    """
    Check if a target is within the agent's field of view.

    Args:
        agent_pos: Agent's position
        agent_direction: Agent's facing direction
        target_pos: Target's position
        fov_degrees: Field of view in degrees (default 90)
        max_distance: Maximum viewing distance

    Returns:
        True if target is visible
    """
    # Check distance
    distance = agent_pos.distance(target_pos)
    if distance > max_distance:
        return False

    # Check angle
    relative_angle = abs(calculate_relative_angle(agent_pos, agent_direction, target_pos))
    return relative_angle <= fov_degrees / 2


def format_position(pos: Vector | tuple[float, float]) -> str:
    """Format a position for display."""
    if isinstance(pos, Vector):
        return f"({pos.x:.0f}, {pos.y:.0f})"
    return f"({pos[0]:.0f}, {pos[1]:.0f})"


def format_distance(distance_cm: float) -> str:
    """Format a distance for display (converts cm to meters if large)."""
    if distance_cm >= 100:
        return f"{distance_cm / 100:.1f}m"
    return f"{distance_cm:.0f}cm"


def spawn_in_circle(
    center: tuple[float, float],
    radius: float,
    count: int,
    offset_angle: float = 0.0
) -> list[tuple[float, float]]:
    """
    Generate spawn positions in a circle.

    Args:
        center: Center of the circle
        radius: Radius of the circle
        count: Number of positions to generate
        offset_angle: Starting angle offset in degrees

    Returns:
        List of (x, y) positions
    """
    positions = []
    for i in range(count):
        angle = math.radians(offset_angle + (i / count) * 360)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        positions.append((x, y))
    return positions


def spawn_in_grid(
    origin: tuple[float, float],
    spacing: float,
    rows: int,
    cols: int
) -> list[tuple[float, float]]:
    """
    Generate spawn positions in a grid pattern.

    Args:
        origin: Top-left corner of the grid
        spacing: Distance between positions
        rows: Number of rows
        cols: Number of columns

    Returns:
        List of (x, y) positions
    """
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = origin[0] + c * spacing
            y = origin[1] + r * spacing
            positions.append((x, y))
    return positions


def random_position_in_area(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float
) -> tuple[float, float]:
    """Generate a random position within a rectangular area."""
    import random
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    return (x, y)
