# Game Framework Usage

Quick commands for running and recording multi‑agent games built on SimWorld.

## Run a Game
```bash
python -m games.runner --game hide_and_seek --ip 127.0.0.1 --port 9000 --max-steps 1000
```
- Core:  
  - `--game NAME` (see `--list`)  
  - `--config PATH` for YAML/JSON overrides  
  - `--ip`, `--port` for UnrealCV host/port  
  - `--max-steps` (default: use game config)  
  - `--verbose`
- Recording (on by default):  
  - `--output-dir` (default `results/`)  
  - `--fps` (default 30)  
  - `--live-preview`  
  - `--keep-frames` (preserve raw JPEGs)  
  - `--no-record` to disable recording
- Utility:  
  - `--list` to show available games

Outputs per recorded session:
- `agents/<id>/first_person.mp4` — POV video per agent.
- `agents/<id>/reasoning.log` — thoughts/actions/results per step.
- `game_log.jsonl` — stepwise game/state snapshot.
- `events.log` — timestamped game events.
- `game_summary.json` — final result metadata.

## Re-encode Frames After a Run
If you kept frames or need to re-encode (defaults to most recent session under `results/` if no path is provided):
```bash
python -m games.process_frames [results/<session_dir>] --fps 30
```
- Looks for `agents/*/frames/*.jpg` and writes `first_person.mp4`.
- If no session is specified, the latest directory in `results/` is used.
- Deletes frames after a successful encode.

## Writing a New Game
1. Create a module under `games/examples/your_game/` and register it with `@register_game`.
2. Implement a subclass of `BaseGame`:
   - `get_agent_configs`, `setup`, `build_observation`, `process_action`, `check_win_condition`, `check_lose_condition`.
3. Optional: add a `config.yaml` and reference it via `--config`.
4. Run with `python -m games.runner --game your_game`.

## Tips
- Keep agent camera resolution modest to reduce frame storage/encode time.
- If LLM calls are slow, add cheap fast-path actions (e.g., wait during pregame phases) in your game logic.
- Disable recording when iterating quickly (`--no-record`), re-enable for real runs.
