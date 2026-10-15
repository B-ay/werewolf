"""多智能体狼人杀引擎的批量竞技与可复现实验入口。

一场比赛并不是复用同一批有状态 Agent，而是为每个座位重新实例化策略；随后用
固定 ``seed`` 驱动裁判随机流，并按局轮换不同版本的座位，减少角色与座次偏差。
单局失败会被隔离并记录，不会默认丢弃整批实验。输出同时包含逐局评测、汇总和
Elo 榜单。

本模块同时提供 Python API 和不依赖第三方库的 JSON CLI::

    python -m backend.evaluation.tournament --seeds 1,2,3 --games-per-seed 2

``TournamentRunner`` 接受可注入的 ``game_factory``，测试替身必须遵循同一
run/result/event_log 接口；默认使用 :class:`backend.game.engine.WerewolfGame`。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

from .evaluator import evaluate_game
from .leaderboard import build_leaderboard
from .common import _mean, _rate, _win_rates
from ..agents.base import Agent
from ..agents.factory import AgentFactory, create_agent
from ..game.engine import WerewolfGame
from ..game.models import GameConfig, Role


GameFactory = Callable[..., WerewolfGame]
Evaluator = Callable[..., dict[str, Any]]


class TournamentError(RuntimeError):
    """Raised when a scheduled game cannot produce a terminal result."""


@dataclass(frozen=True)
class Competitor:
    """An agent/model version that can occupy one or more seats."""

    name: str
    model: str = "heuristic"
    version: str = "baseline"
    factory: AgentFactory | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.factory is not None and not callable(self.factory):
            raise TypeError("competitor factory must be callable as factory(player_id, role); agent prototypes are unsupported")

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
            return cls(
                name=str(value.get("name", f"agent-{index + 1}")),
                model=str(value.get("model", "heuristic")),
                version=str(value.get("version", "baseline")),
                factory=value.get("factory"),
            )
        if callable(value):
            return cls(getattr(value, "__name__", type(value).__name__), factory=value)
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
            "game_config": copy.deepcopy(dict(self.game_config)),
            "competitors": [item.metadata() for item in self.competitors],
            "max_steps": self.max_steps,
            "rotate_seats": self.rotate_seats,
            "fail_fast": self.fail_fast,
            "tournament_id": self.tournament_id,
        }


def _game_config_dict(value: GameConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, GameConfig):
        return value.to_dict()
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise TypeError("game_config must be GameConfig or a mapping")


def _instantiate_agent(competitor: Competitor, player_id: str, role: Role) -> Agent:
    """固定两个参数的工厂；每个座位、每局必须创建独立 Agent。"""
    factory = competitor.factory or create_agent
    try:
        agent = factory(player_id, role)
    except TypeError as exc:
        raise TypeError("competitor factory must accept (player_id, role)") from exc
    if not isinstance(agent, Agent):
        raise TypeError("competitor factory must return an Agent exposing decide()")
    return agent


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


def _run_game(game: WerewolfGame, max_steps: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """使用本项目引擎的确定接口，不猜测外部引擎或补出终局。"""
    game.run(max_steps=max_steps)
    state = game.result()
    status = state["status"]
    if status == "waiting_for_human":
        raise TournamentError("autonomous tournament stopped waiting for human input")
    if status != "finished":
        raise TournamentError(f"game did not finish within {max_steps} steps (status={status})")
    return state, game.event_log(omniscient=True)


def _participant_metadata(players: Sequence[Mapping[str, Any]], seats: Mapping[str, Competitor]) -> dict[str, dict[str, str]]:
    return {str(player["id"]): seats[str(player["id"])].metadata() for player in players}


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
            key = str(participant.get("leaderboard_key") or player_id)
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
            "win_rate": _rate(item["wins"], appearances),
            "average_contribution_score": _mean(item["contribution_scores"]),
            "roles": _win_rates(item["roles"]),
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
        "completion_rate": _rate(completed, scheduled),
        "winner_counts": dict(sorted(winners.items())),
        "winner_rates": {
            winner: _rate(count, completed)
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
    """运行可复现的纯 AI 对局，并聚合评测、失败记录和排行榜。"""

    def __init__(
        self,
        game_factory: GameFactory | None = None,
        evaluator: Evaluator = evaluate_game,
    ) -> None:
        self.game_factory = game_factory or WerewolfGame
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

        # 两层循环定义稳定赛程。相同 seed 可重复多局；若开启座位轮换，每一局都
        # 根据 schedule_index 平移竞争者，避免某个版本长期绑定狼人或关键神职。
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
                    # 工厂在引擎洗牌完成后才知道 player_id 与 role。每次调用都必须
                    # 返回独立策略实例，防止一局的记忆意外泄漏到另一座位或下一局。
                    competitor = assignments.get(player_id)
                    if competitor is None:
                        competitor = Competitor(f"baseline-{player_id}")
                    return _instantiate_agent(competitor, player_id, role)

                try:
                    game = self.game_factory(per_game_config, agent_factory=agent_factory)
                    state, events = _run_game(game, config.max_steps)
                    steps = None  # 保留旧报告字段：同步 run() 不返回实际步数。
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
                            "config": per_game_config,
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
                    # 失败是研究数据的一部分：保留 seed、局号和错误类型，默认继续
                    # 后续赛程；只有 fail_fast=True 时才立即向调用者抛出。
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
