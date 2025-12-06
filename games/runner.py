#!/usr/bin/env python3
"""
Game runner script for SimWorld games.

Usage:
    python -m games.runner --list                    # List all available games
    python -m games.runner --game hide_and_seek      # Run a specific game
    python -m games.runner --game hide_and_seek --config config.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

from games.registry import GameRegistry
from games.base import GameState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_games():
    """List all available games."""
    games = GameRegistry.list_games()

    if not games:
        print("No games found. Add games to games/examples/")
        return

    print("\n" + "=" * 60)
    print("Available Games")
    print("=" * 60)

    for game in games:
        print(f"\n  {game['name']} (v{game['version']})")
        print(f"    {game['description']}")
        print(f"    Players: {game['min_players']}-{game['max_players']}")

    print("\n" + "=" * 60)
    print(f"Total: {len(games)} game(s)")
    print("=" * 60 + "\n")


def load_config(config_path: str | None) -> dict:
    """Load configuration from a YAML or JSON file."""
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    with open(path) as f:
        if path.suffix in ('.yaml', '.yml'):
            return yaml.safe_load(f) or {}
        elif path.suffix == '.json':
            return json.load(f)
        else:
            logger.warning(f"Unknown config format: {path.suffix}")
            return {}


def run_game(
    game_name: str,
    config_path: str | None = None,
    ip: str = "127.0.0.1",
    port: int = 9000,
    max_steps: int = 1000,
    verbose: bool = False,
) -> int:
    """
    Run a game by name.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Load config
    config = load_config(config_path)

    # Override with command line args
    if max_steps:
        config['max_steps'] = max_steps
    if verbose:
        config['verbose'] = verbose

    # Create game instance
    game = GameRegistry.create_game(game_name, **config)
    if game is None:
        print(f"Error: Game '{game_name}' not found")
        print("Use --list to see available games")
        return 1

    print(f"\n{'=' * 60}")
    print(f"Starting: {game.name}")
    print(f"Description: {game.description}")
    print(f"Max Steps: {game.max_steps}")
    print(f"{'=' * 60}\n")

    # Connect to simulation
    print(f"Connecting to simulation at {ip}:{port}...")
    if not game.connect(port=port, ip=ip):
        print("Error: Failed to connect to simulation")
        print("Make sure the Unreal Engine simulation is running")
        return 1

    print("Connected!\n")

    # Initialize game
    print("Initializing game...")
    if not game.initialize():
        print("Error: Failed to initialize game")
        game.disconnect()
        return 1

    print(f"Initialized with {len(game.agents)} agents\n")

    # Run the game
    print("Starting game loop...\n")
    start_time = time.time()

    try:
        result = game.run()
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user")
        result = None
    finally:
        game.disconnect()

    elapsed = time.time() - start_time

    # Print results
    print(f"\n{'=' * 60}")
    print("GAME OVER")
    print(f"{'=' * 60}")

    if result:
        state_emoji = {
            GameState.WIN: "🏆",
            GameState.LOSE: "❌",
            GameState.DRAW: "🤝",
            GameState.ERROR: "⚠️",
        }

        print(f"\nResult: {state_emoji.get(result.state, '')} {result.state.value.upper()}")

        if result.winner:
            print(f"Winner: {result.winner}")

        print(f"Reason: {result.reason}")
        print(f"Total Steps: {result.total_steps}")
        print(f"Time Elapsed: {elapsed:.1f}s")

        if result.final_scores:
            print("\nFinal Scores:")
            for agent_id, score in sorted(result.final_scores.items(), key=lambda x: -x[1]):
                print(f"  {agent_id}: {score:.1f}")
    else:
        print("\nGame did not complete normally")

    print(f"\n{'=' * 60}\n")

    return 0 if result and result.state in (GameState.WIN, GameState.DRAW) else 1


def interactive_select() -> str | None:
    """Interactively select a game to run."""
    games = GameRegistry.list_games()

    if not games:
        print("No games available")
        return None

    print("\nSelect a game to run:\n")
    for i, game in enumerate(games, 1):
        print(f"  {i}. {game['name']} - {game['description']}")

    print(f"\n  0. Exit\n")

    while True:
        try:
            choice = input("Enter choice: ").strip()
            if choice == '0' or choice.lower() in ('q', 'quit', 'exit'):
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(games):
                return games[idx]['name']
            else:
                print("Invalid choice, try again")
        except ValueError:
            # Try to match by name
            for game in games:
                if game['name'].lower() == choice.lower():
                    return game['name']
            print("Invalid choice, try again")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SimWorld Game Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --list
    %(prog)s --game hide_and_seek
    %(prog)s --game hide_and_seek --max-steps 500
    %(prog)s --game hide_and_seek --config my_config.yaml
    %(prog)s -i  # Interactive mode
        """
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available games'
    )

    parser.add_argument(
        '--game', '-g',
        type=str,
        help='Name of the game to run'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to game configuration file (YAML or JSON)'
    )

    parser.add_argument(
        '--ip',
        type=str,
        default='127.0.0.1',
        help='IP address of the simulation server (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=9000,
        help='Port of the simulation server (default: 9000)'
    )

    parser.add_argument(
        '--max-steps', '-m',
        type=int,
        default=1000,
        help='Maximum game steps (default: 1000)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode - select game from menu'
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Discover games first
    GameRegistry.discover_games()

    # Handle commands
    if args.list:
        list_games()
        return 0

    if args.interactive:
        game_name = interactive_select()
        if game_name is None:
            return 0
        return run_game(
            game_name,
            config_path=args.config,
            ip=args.ip,
            port=args.port,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )

    if args.game:
        return run_game(
            args.game,
            config_path=args.config,
            ip=args.ip,
            port=args.port,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )

    # No command specified - show help
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
