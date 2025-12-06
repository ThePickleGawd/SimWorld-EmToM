"""
Game registry for discovering and loading games.

Provides automatic discovery of games and a factory for instantiation.
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Type

from games.base import BaseGame

logger = logging.getLogger(__name__)

# Global registry of games
_GAME_REGISTRY: dict[str, Type[BaseGame]] = {}


def register_game(cls: Type[BaseGame]) -> Type[BaseGame]:
    """
    Decorator to register a game class.

    Usage:
        @register_game
        class MyGame(BaseGame):
            name = "my_game"
            ...
    """
    if not hasattr(cls, 'name') or not cls.name:
        raise ValueError(f"Game class {cls.__name__} must have a 'name' attribute")

    if cls.name in _GAME_REGISTRY:
        logger.warning(f"Game '{cls.name}' already registered, overwriting")

    _GAME_REGISTRY[cls.name] = cls
    logger.debug(f"Registered game: {cls.name}")
    return cls


class GameRegistry:
    """
    Registry for discovering and loading games.

    Automatically discovers games in the games/examples directory
    and provides factory methods for instantiation.
    """

    @staticmethod
    def discover_games(games_path: str | Path | None = None) -> dict[str, Type[BaseGame]]:
        """
        Discover all games in the examples directory.

        Args:
            games_path: Path to games directory (default: games/examples)

        Returns:
            Dictionary mapping game names to game classes
        """
        if games_path is None:
            games_path = Path(__file__).parent / "examples"
        else:
            games_path = Path(games_path)

        if not games_path.exists():
            logger.warning(f"Games path does not exist: {games_path}")
            return _GAME_REGISTRY.copy()

        # Import all game modules to trigger @register_game decorators
        for item in games_path.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                # Try to import the game module
                module_name = f"games.examples.{item.name}"
                try:
                    # First try to import __init__.py
                    importlib.import_module(module_name)
                    logger.debug(f"Loaded game module: {module_name}")
                except ImportError as e:
                    # Try to import game.py directly
                    try:
                        importlib.import_module(f"{module_name}.game")
                        logger.debug(f"Loaded game module: {module_name}.game")
                    except ImportError:
                        logger.debug(f"Could not load game module {module_name}: {e}")

        return _GAME_REGISTRY.copy()

    @staticmethod
    def get_game(name: str) -> Type[BaseGame] | None:
        """
        Get a game class by name.

        Args:
            name: The game name (as defined in the class)

        Returns:
            The game class, or None if not found
        """
        # Ensure games are discovered
        GameRegistry.discover_games()
        return _GAME_REGISTRY.get(name)

    @staticmethod
    def create_game(name: str, **kwargs) -> BaseGame | None:
        """
        Create an instance of a game.

        Args:
            name: The game name
            **kwargs: Arguments to pass to the game constructor

        Returns:
            A game instance, or None if game not found
        """
        game_cls = GameRegistry.get_game(name)
        if game_cls is None:
            logger.error(f"Game not found: {name}")
            return None

        try:
            return game_cls(**kwargs)
        except Exception as e:
            logger.error(f"Error creating game {name}: {e}")
            return None

    @staticmethod
    def list_games() -> list[dict[str, str]]:
        """
        List all registered games with their metadata.

        Returns:
            List of dicts with game info (name, description, players, version)
        """
        GameRegistry.discover_games()

        games = []
        for name, cls in _GAME_REGISTRY.items():
            games.append({
                "name": name,
                "description": getattr(cls, 'description', ''),
                "min_players": getattr(cls, 'min_players', 1),
                "max_players": getattr(cls, 'max_players', 1),
                "version": getattr(cls, 'version', '1.0.0'),
            })

        return sorted(games, key=lambda x: x['name'])

    @staticmethod
    def register(cls: Type[BaseGame]) -> None:
        """Manually register a game class."""
        register_game(cls)

    @staticmethod
    def unregister(name: str) -> bool:
        """Unregister a game by name."""
        if name in _GAME_REGISTRY:
            del _GAME_REGISTRY[name]
            return True
        return False

    @staticmethod
    def clear() -> None:
        """Clear all registered games."""
        _GAME_REGISTRY.clear()
