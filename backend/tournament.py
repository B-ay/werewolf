"""Batch tournament runner for the multi-agent Werewolf engine.

The module offers both a small Python API and a dependency-free JSON CLI::

    python -m backend.tournament --seeds 1,2,3 --games-per-seed 2

``TournamentRunner`` accepts an injectable ``game_factory`` so tournament
orchestration can be tested independently of the concrete game engine.  The
default adapter uses :class:`backend.engine.WerewolfGame`.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import inspect
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from .evaluation import build_leaderboard, evaluate_game


GameFactory = Callable[..., Any]
Evaluator = Callable[..., dict[str, Any]]


class TournamentError(RuntimeError):
    """Raised when a scheduled game cannot produce a terminal result."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_private_dict"):
        return _jsonable(value.to_private_dict())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _status_value(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    return str(value or "").strip().lower()


def _call_supported(callable_object: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Call a hook with only keyword arguments declared by its signature."""

    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return callable_object(*args, **kwargs)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = kwargs if accepts_kwargs else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return callable_object(*args, **supported)


@dataclass(frozen=True)
class Competitor:
    """An agent/model version that can occupy one or more seats."""

    name: str
    model: str = "heuristic"
    version: str = "baseline"
    factory: Callable[..., Any] | None = field(default=None, compare=False, repr=False)

    @property
    def key(self) -> str:
        qualifiers = ":".join(value for value in (self.model, self.version) if value)
        return f"{self.name}@{qualifiers}" if qualifiers else self.name

    def metadata(self) -> dict[str, str]:
        return {
            "agent": self.name,
            "model": self.model,
            "version": self.version,
            "leaderboard_key": self.key,
        }

    @classmethod
    def coerce(cls, value: Any, index: int = 0) -> "Competitor":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            parts = value.split(":", 2)
            return cls(
                name=parts[0] or f"agent-{index + 1}",
                model=parts[1] if len(parts) > 1 and parts[1] else "heuristic",
                version=parts[2] if len(parts) > 2 and parts[2] else "baseline",
            )
        if isinstance(value, Mapping):
            name = value.get("name", value.get("agent", value.get("agent_id", f"agent-{index + 1}")))
            return cls(
                name=str(name),
                model=str(value.get("model", value.get("model_name", "heuristic"))),
                version=str(value.get("version", value.get("agent_version", "baseline"))),
                factory=value.get("factory", value.get("agent_factory")),
            )
        if callable(value) or hasattr(value, "decide"):
            name = getattr(value, "__name__", value.__class__.__name__)
            # Preserve explicit policy instances so the tournament adapter can
            # clone their state.  Wrapping an instance in ``lambda: value``
            # would silently reuse one long-term memory across seats/games.
            return cls(str(name), factory=value)
        raise TypeError(f"unsupported competitor specification: {type(value).__name__}")


@dataclass(frozen=True)
class TournamentConfig:
    """Serializable schedule configuration."""

    seeds: tuple[int, ...] = (1,)
    games_per_seed: int = 1
    game_config: Mapping[str, Any] = field(default_factory=dict)
    competitors: tuple[Competitor, ...] = ()
    max_steps: int = 10_000
    rotate_seats: bool = True
    fail_fast: bool = False
    tournament_id: str = "tournament"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("seeds must not be empty")
        if self.games_per_seed < 1:
            raise ValueError("games_per_seed must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.tournament_id.strip():
            raise ValueError("tournament_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": list(self.seeds),
            "games_per_seed": self.games_per_seed,
            "game_config": _jsonable(self.game_config),
            "competitors": [item.metadata() for item in self.competitors],
            "max_steps": self.max_steps,
            "rotate_seats": self.rotate_seats,
            "fail_fast": self.fail_fast,
            "tournament_id": self.tournament_id,
        }


def _game_config_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    if is_dataclass(value):
        return copy.deepcopy(asdict(value))
    raise TypeError("game_config must be a mapping, dataclass, or expose to_dict()")


def _default_game_factory(config: Mapping[str, Any], **kwargs: Any) -> Any:
    from .engine import WerewolfGame

    return WerewolfGame(config, agent_factory=kwargs.get("agent_factory"))


def _instantiate_agent(competitor: Competitor, player_id: str, role: Any) -> Any:
    if competitor.factory is None:
        from .agents import default_agent_factory

        # Baseline seats use the concrete implementation for their assigned
        # role, so every tournament participant keeps its own goal, memory,
        # prompt, and planner instead of sharing the legacy generic policy.
        return default_agent_factory(player_id, role)
    factory = competitor.factory
    if hasattr(factory, "decide") and not inspect.isclass(factory):
        # Every seat in every game must own its policy state.  This is
        # especially important now that role agents have persistent memories.
        clone = getattr(factory, "clone", None)
        if callable(clone):
            candidate = clone()
        else:
            try:
                candidate = copy.deepcopy(factory)
            except Exception as exc:  # pragma: no cover - defensive API guard
                raise TypeError(
                    "stateful competitor instances must be deepcopy-able, expose clone(), "
                    "or be supplied as a factory"
                ) from exc
        if candidate is factory:
            raise TypeError("competitor clone() must return an independent policy instance")
        return candidate
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(player_id, role)
    parameters = [
        item
        for item in signature.parameters.values()
        if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [item for item in parameters if item.default is inspect.Parameter.empty]
    has_varargs = any(item.kind is inspect.Parameter.VAR_POSITIONAL for item in signature.parameters.values())
    names = {item.name for item in parameters}
    if has_varargs or len(required) >= 2 or {"player_id", "role"}.issubset(names):
        return factory(player_id, role)
    if len(required) == 1 or names.intersection({"player_id", "pid", "actor_id"}):
        return factory(player_id)
    return factory()


def _seat_assignments(
    competitors: Sequence[Competitor], player_count: int, rotation: int
) -> dict[str, Competitor]:
    if player_count < 1:
        return {}
    if not competitors:
        # Separate baseline seats keep self-play Elo meaningful instead of
        # collapsing both opposing factions into one identity.
        return {
            f"p{seat}": Competitor(f"baseline-seat-{seat}")
            for seat in range(1, player_count + 1)
        }
    offset = rotation % len(competitors)
    return {
        f"p{seat + 1}": competitors[(seat + offset) % len(competitors)]
        for seat in range(player_count)
    }


def _extract_state(game: Any, run_result: Any = None) -> dict[str, Any]:
    state: dict[str, Any] = {}
    public_state = getattr(game, "public_state", None)
    if callable(public_state):
        value = _call_supported(public_state, viewer_id=None, omniscient=True)
        if isinstance(_jsonable(value), Mapping):
            state.update(_jsonable(value))
    if isinstance(_jsonable(run_result), Mapping):
        state.update(_jsonable(run_result))
    result_method = getattr(game, "result", None)
    if callable(result_method):
        result = _jsonable(result_method())
        if isinstance(result, Mapping):
            state.update(result)
    if "players" not in state:
        players = getattr(game, "players", [])
        if isinstance(players, Mapping):
            state["players"] = [_jsonable(player) for player in players.values()]
        else:
            state["players"] = _jsonable(players)
    if not state.get("winner"):
        state["winner"] = _jsonable(getattr(game, "winner", None))
    state.setdefault("status", _status_value(getattr(game, "status", "")))
    return state


def _extract_events(game: Any) -> list[dict[str, Any]]:
    event_log = getattr(game, "event_log", None)
    if callable(event_log):
        values = _call_supported(event_log, viewer_id=None, omniscient=True)
    elif event_log is not None:
        values = event_log
    else:
        values = getattr(game, "events", [])
    result = _jsonable(values) or []
    return list(result) if isinstance(result, list) else []


def _run_game(game: Any, max_steps: int) -> tuple[Any, int | None]:
    run = getattr(game, "run", None)
    if callable(run):
        result = _call_supported(run, max_steps=max_steps)
        return result, None

    step = getattr(game, "step", None)
    if not callable(step):
        raise TournamentError("game exposes neither run() nor step()")
    for steps in range(1, max_steps + 1):
        result = step()
        status = _status_value(getattr(game, "status", ""))
        if status in {"finished", "complete", "completed", "ended", "game_over"}:
            return result, steps
        if status in {"waiting_for_human", "waiting", "paused"}:
            raise TournamentError("autonomous tournament stopped waiting for human input")
    return result, max_steps


def _terminal_or_raise(game: Any, state: Mapping[str, Any], max_steps: int) -> None:
    status = _status_value(state.get("status") or getattr(game, "status", ""))
    winner = state.get("winner") or getattr(game, "winner", None)
    if status in {"waiting_for_human", "waiting", "paused"}:
        raise TournamentError("autonomous tournament stopped waiting for human input")
    if status and status not in {"finished", "complete", "completed", "ended", "game_over"} and not winner:
        raise TournamentError(f"game did not finish within {max_steps} steps (status={status})")
    if not status and not winner:
        raise TournamentError("game did not expose a terminal status or winner")


def _participant_metadata(
    players: Any,
    seats: Mapping[str, Competitor],
) -> dict[str, dict[str, str]]:
    if isinstance(players, Mapping):
        iterable = []
        for key, value in players.items():
            item = dict(value) if isinstance(value, Mapping) else {"id": key}
            item.setdefault("id", key)
            iterable.append(item)
    elif isinstance(players, Iterable) and not isinstance(players, (str, bytes)):
        iterable = [value if isinstance(value, Mapping) else {} for value in players]
    else:
        iterable = []
    result: dict[str, dict[str, str]] = {}
    ordered_seats = list(seats.values())
    for index, player in enumerate(iterable):
        player_id = str(player.get("id", player.get("player_id", f"p{index + 1}")))
        competitor = seats.get(player_id)
        if competitor is None and index < len(ordered_seats):
            competitor = ordered_seats[index]
        if competitor is not None:
            result[player_id] = competitor.metadata()
    return result


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return round(statistics.fmean(numeric), 4) if numeric else None


def summarize_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]] = (),
    *,
    scheduled_games: int | None = None,
) -> dict[str, Any]:
    """Build compact outcome, process, seed, and competitor summaries."""

    completed = len(evaluations)
    scheduled = scheduled_games if scheduled_games is not None else completed + len(failures)
    winners: dict[str, int] = {}
    by_seed: dict[str, dict[str, Any]] = {}
    competitor_values: dict[str, dict[str, Any]] = {}
    for evaluation in evaluations:
        result = evaluation.get("result_metrics", {})
        winner = str(result.get("winner") or "unknown")
        winners[winner] = winners.get(winner, 0) + 1
        metadata = evaluation.get("metadata", {})
        seed_key = str(metadata.get("seed", "unknown"))
        seed_summary = by_seed.setdefault(seed_key, {"games": 0, "winners": {}})
        seed_summary["games"] += 1
        seed_summary["winners"][winner] = seed_summary["winners"].get(winner, 0) + 1

        participant_metadata = evaluation.get("participant_metadata", {})
        for player_id, metric in evaluation.get("player_metrics", {}).items():
            participant = participant_metadata.get(player_id, {})
            key = str(participant.get("leaderboard_key") or participant.get("agent_key") or player_id)
            item = competitor_values.setdefault(
                key,
                {
                    "agent": participant.get("agent", key),
                    "model": participant.get("model"),
                    "version": participant.get("version"),
                    "appearances": 0,
                    "wins": 0,
                    "contribution_scores": [],
                    "roles": {},
                },
            )
            item["appearances"] += 1
            item["wins"] += metric.get("won") is True
            score = metric.get("contribution_score")
            if isinstance(score, (int, float)):
                item["contribution_scores"].append(score)
            role = str(metric.get("role") or "unknown")
            role_stats = item["roles"].setdefault(role, {"games": 0, "wins": 0})
            role_stats["games"] += 1
            role_stats["wins"] += metric.get("won") is True

    competitor_summaries: dict[str, dict[str, Any]] = {}
    for key, item in sorted(competitor_values.items()):
        appearances = item["appearances"]
        competitor_summaries[key] = {
            "agent": item["agent"],
            "model": item["model"],
            "version": item["version"],
            "appearances": appearances,
            "wins": item["wins"],
            "win_rate": round(item["wins"] / appearances, 4) if appearances else None,
            "average_contribution_score": _mean(item["contribution_scores"]),
            "roles": {
                role: {
                    **stats,
                    "win_rate": round(stats["wins"] / stats["games"], 4) if stats["games"] else None,
                }
                for role, stats in sorted(item["roles"].items())
            },
        }

    process_paths = {
        "information_isolation_score": ("information_leakage", "isolation_score"),
        "speech_vote_consistency": ("speech_vote_consistency", "rate"),
        "seer_check_utilization": ("seer_check_utilization", "rate"),
        "wolf_coordination": ("wolf_coordination", "mean_agreement"),
    }
    process_averages: dict[str, float | None] = {}
    for output_name, (section, metric) in process_paths.items():
        process_averages[output_name] = _mean(
            evaluation.get("process_metrics", {}).get(section, {}).get(metric)
            for evaluation in evaluations
        )

    return {
        "scheduled_games": scheduled,
        "completed_games": completed,
        "failed_games": len(failures),
        "completion_rate": round(completed / scheduled, 4) if scheduled else None,
        "winner_counts": dict(sorted(winners.items())),
        "winner_rates": {
            winner: round(count / completed, 4) if completed else None
            for winner, count in sorted(winners.items())
        },
        "average_rounds": _mean(
            evaluation.get("result_metrics", {}).get("duration_rounds") for evaluation in evaluations
        ),
        "average_events": _mean(
            evaluation.get("result_metrics", {}).get("event_count") for evaluation in evaluations
        ),
        "process_averages": process_averages,
        "by_seed": by_seed,
        "competitors": competitor_summaries,
    }


class TournamentRunner:
    """Run reproducible autonomous games and aggregate their evaluations."""

    def __init__(
        self,
        game_factory: GameFactory | None = None,
        evaluator: Evaluator = evaluate_game,
    ) -> None:
        self.game_factory = game_factory or _default_game_factory
        self.evaluator = evaluator

    def run(
        self,
        seeds: Iterable[int] = (1,),
        *,
        games_per_seed: int = 1,
        game_config: Any = None,
        competitors: Sequence[Any] | None = None,
        max_steps: int = 10_000,
        rotate_seats: bool = True,
        fail_fast: bool = False,
        tournament_id: str = "tournament",
    ) -> dict[str, Any]:
        seed_values = tuple(int(seed) for seed in seeds)
        competitor_values = tuple(
            Competitor.coerce(value, index) for index, value in enumerate(competitors or ())
        )
        config = TournamentConfig(
            seeds=seed_values,
            games_per_seed=int(games_per_seed),
            game_config=_game_config_dict(game_config),
            competitors=competitor_values,
            max_steps=int(max_steps),
            rotate_seats=bool(rotate_seats),
            fail_fast=bool(fail_fast),
            tournament_id=str(tournament_id),
        )
        base_game_config = _game_config_dict(config.game_config)
        player_count = int(base_game_config.get("player_count") or len(base_game_config.get("player_names", [])) or 8)
        scheduled = len(config.seeds) * config.games_per_seed
        evaluations: list[dict[str, Any]] = []
        games: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        schedule_index = 0

        for seed in config.seeds:
            for repetition in range(config.games_per_seed):
                game_id = f"{config.tournament_id}-s{seed}-r{repetition + 1}"
                per_game_config = copy.deepcopy(base_game_config)
                per_game_config["seed"] = seed
                # Batch mode must not pause for a human action.
                per_game_config["human_player"] = None
                per_game_config["human_players"] = []
                rotation = schedule_index if config.rotate_seats else 0
                seats = _seat_assignments(config.competitors, player_count, rotation)

                def agent_factory(player_id: str, role: Any, assignments: Mapping[str, Competitor] = seats) -> Any:
                    competitor = assignments.get(player_id)
                    if competitor is None:
                        competitor = Competitor(f"baseline-{player_id}")
                    return _instantiate_agent(competitor, player_id, role)

                try:
                    game = _call_supported(
                        self.game_factory,
                        per_game_config,
                        agent_factory=agent_factory,
                        lineup={player_id: value.metadata() for player_id, value in seats.items()},
                        game_id=game_id,
                    )
                    run_result, steps = _run_game(game, config.max_steps)
                    state = _extract_state(game, run_result)
                    _terminal_or_raise(game, state, config.max_steps)
                    events = _extract_events(game)
                    participant_metadata = _participant_metadata(state.get("players", []), seats)
                    evaluation = self.evaluator(
                        events,
                        state.get("players", []),
                        state.get("winner"),
                        game_id=game_id,
                        metadata={
                            "tournament_id": config.tournament_id,
                            "seed": seed,
                            "repetition": repetition + 1,
                            "schedule_index": schedule_index,
                            "steps": steps,
                            "config": _jsonable(per_game_config),
                        },
                    )
                    evaluation["participant_metadata"] = participant_metadata
                    evaluations.append(evaluation)
                    games.append(
                        {
                            "game_id": game_id,
                            "seed": seed,
                            "repetition": repetition + 1,
                            "status": "completed",
                            "winner": evaluation.get("result_metrics", {}).get("winner"),
                            "evaluation_index": len(evaluations) - 1,
                        }
                    )
                except Exception as exc:
                    failure = {
                        "game_id": game_id,
                        "seed": seed,
                        "repetition": repetition + 1,
                        "schedule_index": schedule_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    games.append({**failure, "status": "failed"})
                    if config.fail_fast:
                        raise
                finally:
                    schedule_index += 1

        leaderboard = build_leaderboard(evaluations)
        summaries = summarize_evaluations(evaluations, failures, scheduled_games=scheduled)
        return {
            "schema_version": "1.0",
            "tournament_id": config.tournament_id,
            "config": config.to_dict(),
            "games": games,
            "evaluations": evaluations,
            "leaderboard": leaderboard,
            "summaries": summaries,
            "failures": failures,
        }


def run_tournament(
    seeds: Iterable[int] = (1,),
    *,
    games_per_seed: int = 1,
    game_config: Any = None,
    competitors: Sequence[Any] | None = None,
    max_steps: int = 10_000,
    rotate_seats: bool = True,
    fail_fast: bool = False,
    tournament_id: str = "tournament",
    game_factory: GameFactory | None = None,
    evaluator: Evaluator = evaluate_game,
) -> dict[str, Any]:
    """Functional API for :class:`TournamentRunner`."""

    return TournamentRunner(game_factory=game_factory, evaluator=evaluator).run(
        seeds,
        games_per_seed=games_per_seed,
        game_config=game_config,
        competitors=competitors,
        max_steps=max_steps,
        rotate_seats=rotate_seats,
        fail_fast=fail_fast,
        tournament_id=tournament_id,
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(part))
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return tuple(result)


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("config JSON must contain an object")
    return dict(value)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行可复现的狼人杀多 Agent 批量竞技")
    parser.add_argument("--seeds", type=_parse_seeds, default=(1,), help="逗号或范围，例如 1,2,5-8")
    parser.add_argument("--games-per-seed", type=int, default=1, help="每个 seed 的对局数")
    parser.add_argument("--player-count", type=int, default=None, help="覆盖玩家人数")
    parser.add_argument("--max-steps", type=int, default=10_000, help="单局最大引擎步数")
    parser.add_argument("--config", help="GameConfig JSON 文件")
    parser.add_argument(
        "--competitor",
        action="append",
        default=[],
        metavar="NAME[:MODEL[:VERSION]]",
        help="添加使用内置启发式策略的参赛版本，可重复指定",
    )
    parser.add_argument("--no-seat-rotation", action="store_true", help="关闭逐局座位轮换")
    parser.add_argument("--fail-fast", action="store_true", help="任一对局失败即退出")
    parser.add_argument("--tournament-id", default="tournament")
    parser.add_argument("--output", help="输出 JSON 文件；省略时写入 stdout")
    parser.add_argument("--compact", action="store_true", help="输出紧凑 JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        game_config = _load_config(args.config)
        if args.player_count is not None:
            game_config["player_count"] = args.player_count
            # Explicit names determine player_count in GameConfig.
            if "player_names" in game_config and len(game_config["player_names"]) != args.player_count:
                game_config.pop("player_names")
        result = run_tournament(
            args.seeds,
            games_per_seed=args.games_per_seed,
            game_config=game_config,
            competitors=args.competitor,
            max_steps=args.max_steps,
            rotate_seats=not args.no_seat_rotation,
            fail_fast=args.fail_fast,
            tournament_id=args.tournament_id,
        )
    except Exception as exc:
        parser.exit(1, f"tournament failed: {type(exc).__name__}: {exc}\n")
    text = json.dumps(
        result,
        ensure_ascii=False,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(output_path)
    else:
        print(text)
    return 0 if not result["failures"] else 1


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI.
    sys.exit(main())


__all__ = [
    "Competitor",
    "TournamentConfig",
    "TournamentError",
    "TournamentRunner",
    "run_tournament",
    "summarize_evaluations",
    "build_argument_parser",
    "main",
]
