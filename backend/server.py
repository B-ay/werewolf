"""Agent Arena 的无第三方 Web 依赖 HTTP API 与静态资源服务器。

核心引擎不知道 HTTP、浏览器或 DeepSeek 密钥；本适配层负责把请求转换为
``WerewolfGame`` 调用、选择公开/玩家/裁判视角，并管理后台自动运行线程。
服务端有意只使用 Python 标准库，使全新 Python 环境也能启动完整演示。

DeepSeek 的凭据边界也位于本层：启动时从服务器环境变量创建 Provider，创建
对局时只把 Provider 对象注入角色 Agent 工厂。API 响应、对局配置、日志、回放
和评测中均不会保存或返回 API Key。
"""

from __future__ import annotations

import argparse
from functools import partial
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Optional
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

from .agents.factory import create_agent
from .game.engine import WerewolfGame
from .evaluation import build_leaderboard, evaluate_game
from .evaluation.evolution import EvolutionCoordinator, ExperienceStore, profile_factory, scope_key
from .game.models import GameConfig, GameStatus
from .llm import build_deepseek_provider_from_env
from .logger import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
DATA_ROOT = PROJECT_ROOT / "data"
_AUTO_PROVIDER = object()


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
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class GameRecord:
    """服务端对一局对战的运行时包装。

    ``game`` 保存裁判状态，``strategy``/``model`` 仅是可公开的展示信息；
    Provider 与密钥不进入记录。``lock`` 保证 HTTP 请求和后台线程不会并发推进
    同一局状态机。
    """

    def __init__(
        self,
        game_id: str,
        game: WerewolfGame,
        config: GameConfig,
        *,
        strategy: str = "heuristic",
        model: str = "heuristic",
    ) -> None:
        self.game_id = game_id
        self.game = game
        self.config = config
        # Safe display metadata only. Provider credentials never enter a game
        # record, serialized state, event, replay, or evaluation artifact.
        self.strategy = strategy
        self.model = model
        self.created_at = _utc_now()
        self.updated_at = self.created_at
        self.running = False
        self.error: Optional[str] = None
        self.report: Optional[dict[str, Any]] = None
        self.evolution: Optional[dict[str, Any]] = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        # Monotonic ownership token for background runners.  Reusing only a
        # stop Event allows a rapid stop/start to clear the signal while the
        # previous worker is still exiting, causing two workers to advance the
        # same game.
        self.run_generation = 0


class ArenaManager:
    """统一管理实时对局、后台推进线程、评测报告和落盘回放。"""

    def __init__(self, deepseek_provider: Any = _AUTO_PROVIDER, *, evolution_store: ExperienceStore | None = None) -> None:
        """初始化管理器并建立 DeepSeek Provider 的唯一来源。

        默认哨兵 ``_AUTO_PROVIDER`` 表示调用
        :func:`build_deepseek_provider_from_env`，由它读取服务器进程环境中的
        ``DEEPSEEK_API_KEY`` 和可选模型名；端点、超时等参数沿用 Provider 的
        服务端默认值。测试可以显式传入 ``None``（模拟未配置）或假 Provider，
        但普通对局请求不能替换 Provider，从而避免客户端注入密钥或任意网络端点。
        """

        self._records: dict[str, GameRecord] = {}
        self._lock = threading.RLock()
        self._evolution = EvolutionCoordinator(evolution_store) if evolution_store else None
        # Provider 在服务启动时构造一次，并由多局 DeepSeek Agent 共享；其内部
        # 只负责模型调用，不保存任何角色的对局记忆。每个角色 Agent 的记忆仍
        # 隶属于该局、该座位的 Agent 实例。
        self.deepseek_provider = (
            build_deepseek_provider_from_env()
            if deepseek_provider is _AUTO_PROVIDER
            else deepseek_provider
        )
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    def learning(self) -> EvolutionCoordinator:
        with self._lock:
            if self._evolution is None:
                self._evolution = EvolutionCoordinator(ExperienceStore(DATA_ROOT / "evolution" / "strategies.json"))
            return self._evolution

    def evolution_status(self) -> dict[str, Any]:
        try:
            return self.learning().status()
        except (OSError, ValueError):
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "经验库暂不可用，请稍后重试并检查策略文件") from None

    def capabilities(self) -> dict[str, Any]:
        """公开模型是否可用及模型名，但绝不返回 Provider 配置或凭据。"""

        provider = self.deepseek_provider
        model = str(getattr(provider, "model", "deepseek-flash")) if provider else None
        return {
            # 此标准库服务器提供 HTTP 快照与事件接口，不实现 WebSocket 握手。
            # 显式声明同步能力，便于前端和部署工具选择一致的连接方式。
            "transport": {"type": "http_polling", "poll_interval_ms": 2200},
            "providers": {
                "deepseek": {
                    "configured": provider is not None,
                    "model": model,
                }
            }
        }

    def create(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """根据创建请求组装一局游戏及其五类角色 Agent。

        ``strategy=heuristic`` 使用引擎默认角色工厂；``strategy=deepseek`` 则把
        服务端已创建的 Provider 交给 :func:`create_agent`。该入口根据
        每个座位的 ``Role`` 构建 WerewolfAgent、SeerAgent、WitchAgent、
        HunterAgent 或 VillagerAgent，并为每个实例绑定独立目标、记忆、提示词
        和规划器。Provider 只是共享的无状态模型出口，不会合并不同 Agent 的
        上下文。

        请求体中的任何 API Key 字段都会被明确拒绝；密钥只能存在服务器环境，
        也不会写入 ``GameConfig``、``GameRecord`` 或返回状态。
        """

        config_payload = payload.get("config", payload)
        if not isinstance(config_payload, Mapping):
            raise ApiError(HTTPStatus.BAD_REQUEST, "config 必须是 JSON 对象")
        # 防御纵深：即使前端误把密钥放入表单，也拒绝整个请求而不是静默保存。
        secret_names = ("api_key", "deepseek_api_key", "DEEPSEEK_API_KEY")
        if any(
            name in source
            for source in (payload, config_payload)
            for name in secret_names
        ):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "API Key 只能通过服务器环境变量配置，不能随对局请求提交",
            )
        requested_strategy = str(
            config_payload.get("strategy", payload.get("strategy", "heuristic"))
        ).strip().lower()
        aliases = {"heuristic": "heuristic", "deepseek": "deepseek", "llm": "deepseek"}
        strategy = aliases.get(requested_strategy)
        if strategy is None:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "strategy 必须是 'heuristic'、'deepseek' 或兼容别名 'llm'",
            )
        try:
            enable_evolution = config_payload.get("enable_evolution", payload.get("enable_evolution", False))
            if type(enable_evolution) is not bool:
                raise ValueError("enable_evolution 必须是布尔值")
            config = GameConfig.from_dict(config_payload)
            if strategy == "deepseek":
                if self.deepseek_provider is None:
                    raise ApiError(
                        HTTPStatus.BAD_REQUEST,
                        "DeepSeek Provider 尚未配置，请在服务器环境中设置 DEEPSEEK_API_KEY",
                    )
                # 将 Provider 注入“角色工厂”而不是引擎。WerewolfGame 仍只依赖
                # Agent 协议；共享的 BaseAgent 负责组织角色提示词并调用模型，
                # 五类规划器负责离线基线策略，便于在不改规则引擎的前提下替换
                # 模型或策略版本。
                game = WerewolfGame(
                    config,
                    agent_factory=partial(create_agent, provider=self.deepseek_provider),
                )
                model = str(getattr(self.deepseek_provider, "model", "deepseek-flash"))
            else:
                # 本地启发式仍使用同样的五类独立 Agent，只是不调用外部模型。
                game = WerewolfGame(config)
                model = "heuristic"
            experience = None
            if enable_evolution:
                provider = self.deepseek_provider if strategy == "deepseek" else None
                scope = scope_key(config, provider)
                profiles = self.learning().store.profiles(scope)
                game = WerewolfGame(config, agent_factory=profile_factory(profiles, provider))
                experience = {"enabled": True, "scope": scope,
                              "versions": {role.value: value.version for role, value in profiles.items()}}
        except ApiError:
            raise
        except OSError:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "经验库暂不可用，请稍后重试") from None
        except (TypeError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        game_id = f"game-{uuid4().hex[:10]}"
        record = GameRecord(
            game_id,
            game,
            config,
            strategy=strategy,
            model=model,
        )
        record.evolution = experience
        with self._lock:
            self._records[game_id] = record
        human_id = next(
            (
                player.id
                for player in game.players.values()
                if player.controller == "human"
            ),
            None,
        )
        # A mixed game is created directly in its first human player's view;
        # never put the referee snapshot on the wire and rely on the browser to
        # hide it. Pure-AI games are observer sessions by default.
        return self.state(
            game_id,
            viewer_id=human_id,
            omniscient=human_id is None,
        )

    def require(self, game_id: str) -> GameRecord:
        with self._lock:
            record = self._records.get(game_id)
        if record is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"未找到对局 {game_id}")
        return record

    @staticmethod
    def _status_value(game: WerewolfGame) -> str:
        status = getattr(game, "status", GameStatus.RUNNING)
        return status.value if isinstance(status, Enum) else str(status)

    @staticmethod
    def _public_state(
        game: WerewolfGame, viewer_id: Optional[str], omniscient: bool
    ) -> dict[str, Any]:
        """向引擎请求指定视角状态，并统一转换为 JSON 可序列化结构。"""

        try:
            state = game.public_state(viewer_id=viewer_id, omniscient=omniscient)
        except TypeError:
            state = game.public_state(viewer_id)
        result = _jsonable(state)
        return result if isinstance(result, dict) else {"state": result}

    @staticmethod
    def _events(
        game: WerewolfGame, viewer_id: Optional[str], omniscient: bool
    ) -> list[dict[str, Any]]:
        """读取指定视角的事件流；仅裁判路径可请求全知事件。"""

        event_source = getattr(game, "event_log", None)
        if callable(event_source):
            try:
                events = event_source(viewer_id=viewer_id, omniscient=omniscient)
            except TypeError:
                events = event_source(viewer_id)
        elif event_source is not None:
            events = event_source
        else:
            events = getattr(game, "events", [])
        return list(_jsonable(events) or [])

    def state(
        self,
        game_id: str,
        *,
        viewer_id: Optional[str] = None,
        omniscient: bool = False,
    ) -> dict[str, Any]:
        """返回对局状态，并在服务端完成最后一层视角脱敏。

        身份、私有事件和待行动信息由引擎投影；本层还会移除可重建牌序的
        ``roles``/``seed``，并隐藏私有阶段造成的更新时间和后台暂停信号。
        安全性不依赖前端 CSS 或 JavaScript 是否把字段显示出来。
        """

        record = self.require(game_id)
        with record.lock:
            state = self._public_state(record.game, viewer_id, omniscient)
            config_view = record.config.to_dict()
            if not omniscient:
                # Role order is directly secret, while the deterministic seed
                # would let a client reconstruct a shuffled deck.  Both stay
                # referee-only until an explicitly omniscient projection is
                # requested.
                config_view.pop("roles", None)
                config_view.pop("seed", None)
            state.update(
                {
                    "game_id": game_id,
                    "mode": "mixed" if record.config.human_players else "ai",
                    "strategy": record.strategy,
                    "model": record.model,
                    "config": config_view,
                    # When the engine masks a private night prompt as running,
                    # mask the background runner's pause as well; otherwise
                    # the two fields recreate the same role side channel.
                    "running": record.running
                    or (
                        self._status_value(record.game)
                        == GameStatus.WAITING_FOR_HUMAN.value
                        and state.get("status") == GameStatus.RUNNING.value
                    ),
                    "server_error": record.error,
                    "created_at": record.created_at,
                }
            )
            # This timestamp advances after every internal decision, including
            # fully private night actions.  Exposing it to a public/player view
            # would let a polling client count hidden decisions even when the
            # projected state is otherwise unchanged.
            if omniscient:
                state["updated_at"] = record.updated_at
            if record.report is not None:
                state["evaluation"] = record.report
            if record.evolution is not None:
                state["evolution"] = record.evolution
            return state

    def events(
        self,
        game_id: str,
        *,
        after: int = 0,
        viewer_id: Optional[str] = None,
        omniscient: bool = False,
    ) -> dict[str, Any]:
        record = self.require(game_id)
        with record.lock:
            events = self._events(record.game, viewer_id, omniscient)
            selected = [item for item in events if int(item.get("seq", 0)) > after]
            last_seq = max((int(item.get("seq", 0)) for item in events), default=0)
            return {"game_id": game_id, "events": selected, "last_seq": last_seq}

    def step(
        self,
        game_id: str,
        action: Optional[Mapping[str, Any]] = None,
        *,
        viewer_id: Optional[str] = None,
        omniscient: bool = True,
    ) -> dict[str, Any]:
        """手动推进对局，并按调用者权限返回结果。

        对公开/玩家视角，一次请求推进到“下一个可观察变化”而不是固定一个
        裁判内部动作，避免客户端通过无变化的请求次数统计狼人或神职数量。
        所有 Agent 决策仍由引擎依次执行和校验。
        """

        record = self.require(game_id)
        with record.lock:
            if record.running:
                raise ApiError(HTTPStatus.CONFLICT, "自动运行中，不能手动推进")
            try:
                if action:
                    record.game.step(action)
                elif omniscient:
                    record.game.step()
                else:
                    # A low-privilege "step" means one *observable* change,
                    # not one referee-internal decision.  Otherwise callers
                    # can count wolves/special roles by repeatedly stepping
                    # through a night while nothing public changes.
                    before = json.dumps(
                        self._public_state(record.game, viewer_id, False),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    player_count = len(getattr(record.game, "players", {}))
                    for _ in range(max(32, player_count * 8)):
                        record.game.step()
                        after = json.dumps(
                            self._public_state(record.game, viewer_id, False),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if after != before:
                            break
                        if self._status_value(record.game) != GameStatus.RUNNING.value:
                            break
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
            record.updated_at = _utc_now()
            self._finalize_if_needed(record)
        return self.state(game_id, viewer_id=viewer_id, omniscient=omniscient)

    def submit_action(
        self,
        game_id: str,
        action: Mapping[str, Any],
        *,
        viewer_id: Optional[str] = None,
        omniscient: bool = False,
    ) -> dict[str, Any]:
        """提交人类玩家动作；合法性和行动者身份由引擎统一验证。"""

        record = self.require(game_id)
        with record.lock:
            try:
                record.game.submit_action(action)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
            record.updated_at = _utc_now()
            self._finalize_if_needed(record)
        return self.state(game_id, viewer_id=viewer_id, omniscient=omniscient)

    def start_run(
        self,
        game_id: str,
        *,
        delay: float = 0.28,
        max_steps: int = 1000,
        viewer_id: Optional[str] = None,
        omniscient: bool = True,
    ) -> dict[str, Any]:
        record = self.require(game_id)
        with record.lock:
            if self._status_value(record.game) == GameStatus.FINISHED.value:
                self._finalize_if_needed(record)
                return self.state(
                    game_id, viewer_id=viewer_id, omniscient=omniscient
                )
            if record.running:
                return self.state(
                    game_id, viewer_id=viewer_id, omniscient=omniscient
                )
            record.run_generation += 1
            generation = record.run_generation
            record.running = True
            record.error = None
            record.stop_event.clear()
        thread = threading.Thread(
            target=self._run_loop,
            args=(
                record,
                max(0.0, min(float(delay), 5.0)),
                max(1, int(max_steps)),
                generation,
            ),
            name=f"arena-{game_id}",
            daemon=True,
        )
        thread.start()
        return self.state(game_id, viewer_id=viewer_id, omniscient=omniscient)

    def stop_run(
        self,
        game_id: str,
        *,
        viewer_id: Optional[str] = None,
        omniscient: bool = True,
    ) -> dict[str, Any]:
        record = self.require(game_id)
        with record.lock:
            record.run_generation += 1
            record.stop_event.set()
            record.running = False
            record.updated_at = _utc_now()
        return self.state(game_id, viewer_id=viewer_id, omniscient=omniscient)

    def _run_loop(
        self,
        record: GameRecord,
        delay: float,
        max_steps: int,
        generation: int,
    ) -> None:
        """后台连续推进纯 AI 对局，遇到人类输入或终局即暂停。

        每次循环只调用一次引擎 ``step``，因此单个 Agent 的流程仍是：隔离观察
        → 记忆/规划 → 可选 DeepSeek 调用 → 动作校验 → 状态更新。``generation``
        防止快速停止再启动时旧线程与新线程同时推进同一局。
        """

        steps = 0
        try:
            while steps < max_steps:
                with record.lock:
                    if (
                        generation != record.run_generation
                        or record.stop_event.is_set()
                    ):
                        break
                    status = self._status_value(record.game)
                    if status in (
                        GameStatus.FINISHED.value,
                        GameStatus.WAITING_FOR_HUMAN.value,
                    ):
                        break
                    record.game.step()
                    record.updated_at = _utc_now()
                    steps += 1
                if delay:
                    record.stop_event.wait(delay)
        except Exception as exc:  # Keep the server alive and surface runner errors.
            with record.lock:
                if generation == record.run_generation:
                    record.error = f"{type(exc).__name__}: {exc}"
        finally:
            with record.lock:
                if generation == record.run_generation:
                    record.running = False
                    record.updated_at = _utc_now()
                    self._finalize_if_needed(record)

    def _finalize_if_needed(self, record: GameRecord) -> None:
        if self._status_value(record.game) != GameStatus.FINISHED.value:
            return
        if record.report is None:
            events = self._events(record.game, None, True)
            state = self._public_state(record.game, None, True)
            players = state.get("players", [])
            winner = state.get("winner")
            record.report = evaluate_game(
                events,
                players,
                winner,
                game_id=record.game_id,
                metadata={
                    "seed": record.config.seed,
                    "config": record.config.to_dict(),
                },
            )
            if record.evolution is not None:
                try:
                    coordinator = self.learning()
                    provider = self.deepseek_provider if record.strategy == "deepseek" else None
                    feedback = coordinator.feedback(record.game_id, record.game, record.report, provider)
                    record.report["evolution"] = {
                        **feedback, "versions_used": record.evolution["versions"],
                        "validation_status": coordinator.schedule(record.config, provider),
                    }
                except Exception:
                    # 经验库失败不能中断胜负结算，也不把 Provider/文件异常带入回放。
                    record.report["evolution"] = {"status": "error", "reason": "feedback_or_validation_failed"}
        self._persist(record)

    def _persist(self, record: GameRecord) -> None:
        payload = {
            "game_id": record.game_id,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "config": record.config.to_dict(),
            "state": self._public_state(record.game, None, True),
            "events": self._events(record.game, None, True),
            "evaluation": record.report,
        }
        path = DATA_ROOT / f"{record.game_id}.json"
        temporary = DATA_ROOT / f".{record.game_id}.tmp"
        write_json_atomic(path, _jsonable(payload), temporary_path=temporary)

    def evaluation(
        self,
        game_id: str,
        *,
        viewer_id: Optional[str] = None,
        omniscient: bool = True,
    ) -> dict[str, Any]:
        record = self.require(game_id)
        with record.lock:
            status = self._status_value(record.game)
            if status != GameStatus.FINISHED.value and not omniscient:
                # Process metrics (seer utilization, wolf coordination, role
                # contributions, turning points) are themselves derived from
                # private actions.  Do not expose a partial live report to a
                # player/public projection: even aggregates are side channels.
                state = self._public_state(record.game, viewer_id, False)
                return {
                    "game_id": game_id,
                    "status": state.get("status", status),
                    "available": False,
                    "message": "完整评测将在对局结束后开放，以避免过程指标泄露隐藏信息。",
                    "result_metrics": {
                        "winner": state.get("winner"),
                        "win_reason": state.get("win_reason"),
                        "rounds": state.get("round"),
                    },
                    "process_metrics": {},
                    "player_metrics": [],
                    "turning_points": [],
                    "replay": {},
                }
            if record.report is not None:
                return record.report
            state = self._public_state(record.game, None, True)
            report = evaluate_game(
                self._events(record.game, None, True),
                state.get("players", []),
                state.get("winner"),
                game_id=game_id,
                metadata={"seed": record.config.seed, "live": True},
            )
            # A live report is a point-in-time preview. Caching it would make a
            # later finished-game report stale if this endpoint was opened
            # while the game was still running.
            if self._status_value(record.game) == GameStatus.FINISHED.value:
                record.report = report
            return report

    def leaderboard(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records.values())
        evaluations: list[dict[str, Any]] = []
        for record in records:
            with record.lock:
                if record.report:
                    evaluations.append(record.report)
        return build_leaderboard(evaluations)

    def list_games(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records.values())
        games = []
        for record in records:
            with record.lock:
                state = self._public_state(record.game, None, True)
                games.append(
                    {
                        "game_id": record.game_id,
                        "status": state.get("status", self._status_value(record.game)),
                        "winner": state.get("winner"),
                        "round": state.get("round", 0),
                        "phase": state.get("phase"),
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                        "running": record.running,
                        "strategy": record.strategy,
                        "model": record.model,
                    }
                )
        games.sort(key=lambda item: item["created_at"], reverse=True)
        return {"games": games}


MANAGER = ArenaManager()


class ArenaRequestHandler(BaseHTTPRequestHandler):
    server_version = "WerewolfAgentArena/1.0"

    @property
    def manager(self) -> ArenaManager:
        return getattr(self.server, "arena_manager", MANAGER)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "无效的 Content-Length") from exc
        if length == 0:
            return {}
        if length > 1_000_000:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求体过大")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求体必须是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求体必须是 JSON 对象")
        return value

    def _route_parts(self) -> tuple[list[str], dict[str, list[str]]]:
        parsed = urlparse(self.path)
        parts = [unquote(item) for item in parsed.path.split("/") if item]
        return parts, parse_qs(parsed.query)

    @staticmethod
    def _truthy(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on", "observer"}

    def _payload_perspective(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_viewer: Optional[str] = None,
    ) -> tuple[Optional[str], bool]:
        """把请求中的 viewer/omniscient 规范化为不可矛盾的视角参数。

        一旦指定具体玩家或显式 ``public``，即使请求同时声称
        ``omniscient=true`` 也会强制关闭全知模式，防止客户端通过参数组合越权。
        未指定 viewer 的本地观战台默认使用裁判视角。
        """

        raw_viewer = payload.get("viewer", fallback_viewer)
        viewer = str(raw_viewer) if raw_viewer not in (None, "") else None
        perspective = viewer
        if viewer in {"observer", "public"}:
            viewer = None
        raw_omniscient = payload.get("omniscient")
        if raw_omniscient is None:
            omniscient = perspective != "public" and viewer is None
        elif isinstance(raw_omniscient, bool):
            omniscient = raw_omniscient
        else:
            omniscient = self._truthy(str(raw_omniscient))
        # A concrete player id or the explicit public sentinel always selects
        # a non-omniscient projection. Never accept a contradictory flag.
        if perspective == "public" or viewer is not None:
            omniscient = False
        return viewer, omniscient

    def do_GET(self) -> None:  # noqa: N802 - stdlib interface
        try:
            parts, query = self._route_parts()
            if parts[:1] == ["api"]:
                self._get_api(parts[1:], query)
            else:
                self._serve_static(parts)
        except ApiError as exc:
            self._send_json({"error": exc.message}, exc.status)
        except Exception as exc:
            self._send_json(
                {"error": "服务器内部错误", "detail": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_POST(self) -> None:  # noqa: N802 - stdlib interface
        try:
            parts, _query = self._route_parts()
            if parts[:1] != ["api"]:
                raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")
            payload = self._read_json()
            self._post_api(parts[1:], payload)
        except ApiError as exc:
            self._send_json({"error": exc.message}, exc.status)
        except Exception as exc:
            self._send_json(
                {"error": "服务器内部错误", "detail": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib interface
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()

    def _get_api(self, parts: list[str], query: dict[str, list[str]]) -> None:
        manager = self.manager
        if parts == ["health"] or parts == ["state"]:
            if parts == ["state"] and manager._records:
                requested = (query.get("game_id") or [None])[0]
                game_id = requested or next(reversed(manager._records))
                viewer = (query.get("viewer") or [None])[0]
                perspective = viewer
                if viewer in {"observer", "public"}:
                    viewer = None
                # No viewer means the local observer console. A concrete viewer
                # always gets a freshly projected private state from the engine.
                omniscient = self._truthy(
                    (query.get("omniscient") or [None])[0],
                    perspective != "public" and viewer is None,
                )
                if perspective == "public" or viewer is not None:
                    omniscient = False
                self._send_json(
                    manager.state(game_id, viewer_id=viewer, omniscient=omniscient)
                )
            else:
                health = {
                    "ok": True,
                    "service": "werewolf-agent-arena",
                    "active_games": len(manager._records),
                }
                health.update(manager.capabilities())
                self._send_json(health)
            return
        if parts in (["games"], ["replays"]):
            self._send_json(manager.list_games())
            return
        if parts == ["leaderboard"]:
            self._send_json(manager.leaderboard())
            return
        if parts == ["evolution"]:
            self._send_json(manager.evolution_status())
            return
        if len(parts) >= 2 and parts[0] == "games":
            game_id = parts[1]
            viewer = (query.get("viewer") or [None])[0]
            perspective = viewer
            if viewer in {"observer", "public"}:
                viewer = None
            omniscient = self._truthy(
                (query.get("omniscient") or [None])[0],
                perspective != "public" and viewer is None,
            )
            if perspective == "public" or viewer is not None:
                omniscient = False
            if len(parts) == 2:
                self._send_json(
                    manager.state(game_id, viewer_id=viewer, omniscient=omniscient)
                )
                return
            if parts[2] == "events":
                try:
                    after = int((query.get("after") or ["0"])[0])
                except ValueError:
                    after = 0
                self._send_json(
                    manager.events(
                        game_id,
                        after=after,
                        viewer_id=viewer,
                        omniscient=omniscient,
                    )
                )
                return
            if parts[2] == "evaluation":
                self._send_json(
                    manager.evaluation(
                        game_id,
                        viewer_id=viewer,
                        omniscient=omniscient,
                    )
                )
                return
        raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")

    def _post_api(self, parts: list[str], payload: dict[str, Any]) -> None:
        manager = self.manager
        if parts == ["games"]:
            self._send_json(manager.create(payload), HTTPStatus.CREATED)
            return
        if len(parts) == 3 and parts[0] == "games":
            game_id, command = parts[1], parts[2]
            if command == "step":
                action = payload.get("action")
                action_value = action if isinstance(action, Mapping) else None
                fallback = (
                    str(action_value.get("actor_id"))
                    if action_value and action_value.get("actor_id")
                    else None
                )
                viewer, omniscient = self._payload_perspective(
                    payload, fallback_viewer=fallback
                )
                self._send_json(
                    manager.step(
                        game_id,
                        action_value,
                        viewer_id=viewer,
                        omniscient=omniscient,
                    )
                )
                return
            if command == "run":
                viewer, omniscient = self._payload_perspective(payload)
                delay_value = payload.get("delay")
                if delay_value is None:
                    milliseconds = payload.get(
                        "delay_ms", payload.get("step_delay_ms", 280)
                    )
                    delay_value = float(milliseconds) / 1000.0
                self._send_json(
                    manager.start_run(
                        game_id,
                        delay=float(delay_value),
                        max_steps=int(payload.get("max_steps", 1000)),
                        viewer_id=viewer,
                        omniscient=omniscient,
                    )
                )
                return
            if command == "stop":
                viewer, omniscient = self._payload_perspective(payload)
                self._send_json(
                    manager.stop_run(
                        game_id, viewer_id=viewer, omniscient=omniscient
                    )
                )
                return
            if command in {"action", "human-action"}:
                action = payload.get("action", payload)
                if not isinstance(action, Mapping):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "action 必须是 JSON 对象")
                fallback = str(action.get("actor_id")) if action.get("actor_id") else None
                viewer, omniscient = self._payload_perspective(
                    payload, fallback_viewer=fallback
                )
                self._send_json(
                    manager.submit_action(
                        game_id,
                        action,
                        viewer_id=viewer,
                        omniscient=omniscient,
                    )
                )
                return
        raise ApiError(HTTPStatus.NOT_FOUND, "接口不存在")

    def _serve_static(self, parts: list[str]) -> None:
        relative = Path(*parts) if parts else Path("index.html")
        candidate = (FRONTEND_ROOT / relative).resolve()
        try:
            candidate.relative_to(FRONTEND_ROOT.resolve())
        except ValueError as exc:
            raise ApiError(HTTPStatus.FORBIDDEN, "禁止访问") from exc
        if not candidate.is_file():
            # SPA fallback, but do not turn missing assets into HTML.
            if candidate.suffix:
                raise ApiError(HTTPStatus.NOT_FOUND, "文件不存在")
            candidate = FRONTEND_ROOT / "index.html"
        body = candidate.read_bytes()
        media_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if media_type.startswith("text/") or media_type in {
            "application/javascript",
            "application/json",
        }:
            media_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    manager: Optional[ArenaManager] = None,
) -> ThreadingHTTPServer:
    """创建线程化 HTTP 服务器并绑定唯一的 :class:`ArenaManager`。

    默认 ``MANAGER`` 在模块加载时从环境构建 DeepSeek Provider；测试或嵌入式
    调用可传入自己的管理器。请求处理器只访问管理器，不直接接触 Provider，
    从而保持模型配置、对局状态和 HTTP 路由三者解耦。
    """

    server = ThreadingHTTPServer((host, port), ArenaRequestHandler)
    server.arena_manager = (  # type: ignore[attr-defined]
        manager if manager is not None else MANAGER
    )
    return server


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="狼人杀 Agent Team 观战服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    args = parser.parse_args(argv)
    server = create_server(args.host, args.port)
    print(f"Werewolf Agent Arena: http://{args.host}:{args.port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
