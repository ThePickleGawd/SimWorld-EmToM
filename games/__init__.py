"""
SimWorld Games Framework

A framework for creating multi-agent LLM games on top of SimWorld.
"""

from games.base import BaseGame, GameState, GameResult
from games.agent_config import AgentConfig, AgentRole
from games.registry import GameRegistry, register_game

__all__ = [
    'BaseGame',
    'GameState',
    'GameResult',
    'AgentConfig',
    'AgentRole',
    'GameRegistry',
    'register_game',
]
