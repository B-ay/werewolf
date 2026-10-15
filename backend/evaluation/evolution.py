"""对局→复盘反馈→有界策略提案→独立种子成对验证→晋级/保留。

训练只保存角色维度的汇总和白名单经验，不把旧局身份带入下一局。
验证局不回灌训练统计；同一模型、同一规则板型的版本才可以比较和继承。
默认只运行本地策略；使用真实 Provider 验证需要显式授权。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from ..agents.base import BaseAgent
from ..agents.factory import create_agent
from ..agents.strategy import ROLE_LESSON, RoleStrategy, propose_strategy
from ..game.engine import WerewolfGame
from ..game.models import GameConfig, Role
from ..llm.base import LLMProvider
from ..logger import write_json_atomic
from .evaluator import evaluate_game


DEFAULT_STORE = Path(__file__).resolve().parents[2] / "data" / "evolution" / "strategies.json"
MIN_VALIDATION_PAIRS = 24
FEEDBACK_FIELDS = {"games", "losses", "votes", "wrong_votes", "abilities", "wrong_abilities", "inconsistent_votes"}


def _validate_feedback(feedback: Mapping[str, Any]) -> None:
    if not isinstance(feedback, Mapping):
        raise ValueError("feedback must be a role aggregate mapping")
    for role, values in feedback.items():
        if Role.coerce(role).value != role or not isinstance(values, Mapping):
            raise ValueError("feedback must use canonical roles and aggregate counters")
        if set(values) - FEEDBACK_FIELDS:
            raise ValueError("feedback must contain role aggregates only")
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("feedback counters must be non-negative integers")


def experiment_config(config: GameConfig | Mapping[str, Any] | None = None) -> GameConfig:
    """去掉人名、人工控制与训练 seed；保留会改变规则/角色分配的配置。"""
    config = config if isinstance(config, GameConfig) else GameConfig.from_dict(config or {})
    value = config.to_dict()
    value.pop("player_names", None)
    value["human_players"], value["human_player"], value["seed"] = [], None, 0
    return GameConfig.from_dict(value)


def scope_key(config: GameConfig, provider: LLMProvider | None = None) -> str:
    rules = experiment_config(config).to_dict()
    fingerprint = hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:16]
    model = str(getattr(provider, "model", "configured-model")) if provider else "heuristic"
    return model + ":" + fingerprint


def game_feedback(game: WerewolfGame, report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """完局后才允许用裁判结果归因，输出不含座位、原话、查验或刀口。"""
    state = game.result()
    if state["status"] != "finished":
        raise ValueError("feedback is only allowed after the game is finished")
    players = {player["id"]: player for player in state["players"]}
    winner = state["winner"]
    feedback: dict[str, dict[str, Any]] = {}
    for role in Role:
        members = [player for player in players.values() if player["role"] == role.value and player["controller"] != "human"]
        if members:
            feedback[role.value] = {
                "games": 1, "losses": int(winner not in {role.team.value, "draw"}),
                "votes": 0, "wrong_votes": 0, "abilities": 0,
                "wrong_abilities": 0, "inconsistent_votes": 0,
            }
    for event in game.event_log(omniscient=True):
        actor = players.get(event.get("actor_id"))
        target = players.get(event.get("target_id"))
        if not actor or not target or actor["controller"] == "human":
            continue
        own = feedback[actor["role"]]
        kind = event["event_type"]
        if kind == "vote_cast":
            own["votes"] += 1
            own["wrong_votes"] += actor["team"] == target["team"]
        elif kind in {"witch_poison", "hunter_shot"}:
            # passed 事件不是技能尝试；不能把保留药剂误判为技能成功/失败。
            own["abilities"] += 1
            own["wrong_abilities"] += actor["team"] == target["team"]
    consistency = report.get("process_metrics", {}).get("speech_vote_consistency", {}).get("by_player", {})
    for player_id, metric in consistency.items():
        if player_id in players and players[player_id]["controller"] != "human":
            own = feedback[players[player_id]["role"]]
            own["inconsistent_votes"] += metric["sample_size"] - metric["consistent_count"]
    return feedback


class ExperienceStore:
    """线程锁 + 跨进程文件锁 + 原子落盘；竞争时明确失败，不覆盖别人的版本。"""

    def __init__(self, path: Path | str = DEFAULT_STORE) -> None:
        self.path = Path(path)
        self.lock = threading.RLock()
        self.data: dict[str, Any] = {"schema_version": 1, "scopes": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
                raise ValueError("unsupported evolution store schema")
            if not isinstance(loaded.get("scopes"), dict):
                raise ValueError("invalid evolution scope mapping")
            for lane in loaded["scopes"].values():
                if not isinstance(lane, dict) or set(lane.get("active", {})) != {role.value for role in Role}:
                    raise ValueError("evolution store requires all five role strategies")
                for role, value in lane["active"].items():
                    if RoleStrategy.from_dict(value).role.value != role:
                        raise ValueError("experience role mismatch")
                _validate_feedback(lane.get("feedback", {}))
            self.data = loaded

    @contextmanager
    def transaction(self):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_suffix(".lock").open("a+b") as handle:
                handle.seek(0, 2)
                if not handle.tell():
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                import os
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    self._load()  # 另一个进程可能刚刚晋级了版本，必须重新读取。
                    before = deepcopy(self.data)
                    try:
                        yield
                    except Exception:
                        self.data = before
                        raise
                finally:
                    handle.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _lane(self, scope: str) -> dict[str, Any]:
        return self.data["scopes"].setdefault(scope, {
            "active": {role.value: RoleStrategy(role).to_dict() for role in Role},
            "feedback": {}, "processed_games": [], "training_seeds": [],
            "validation_seeds": [], "revision": 0, "history": [],
        })

    def _save(self) -> None:
        write_json_atomic(self.path, self.data)

    def profiles(self, scope: str) -> dict[Role, RoleStrategy]:
        with self.transaction():
            return {Role.coerce(role): RoleStrategy.from_dict(value) for role, value in self._lane(scope)["active"].items()}

    def check_seeds(self, scope: str, seeds: Iterable[int], *, validation: bool) -> None:
        with self.transaction():
            values = tuple(seeds)
            if any(type(seed) is not int or seed < 0 for seed in values):
                raise ValueError("seeds must be non-negative integers")
            forbidden = self._lane(scope)["training_seeds" if validation else "validation_seeds"]
            if set(values).intersection(forbidden):
                raise ValueError("training and held-out validation seeds must be disjoint")
            if validation:
                lane = self._lane(scope)
                lane["validation_seeds"] = list(dict.fromkeys((*lane["validation_seeds"], *values)))
                self._save()

    def record(self, scope: str, game_id: str, seed: int, feedback: Mapping[str, Any],
               *, versions: Mapping[str, str] | None = None) -> bool:
        _validate_feedback(feedback)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        with self.transaction():
            lane = self._lane(scope)
            if game_id in lane["processed_games"]:
                return False
            if seed in lane["validation_seeds"]:
                raise ValueError("held-out validation seeds cannot be used as training feedback")
            for role, values in feedback.items():
                if versions is not None and versions.get(role) != lane["active"][role]["version"]:
                    # 较早开始的旧版对局可能晚于晋级结束，不能当成新版的错误。
                    continue
                total = lane["feedback"].setdefault(role, {key: 0 for key in values})
                for key, value in values.items():
                    total[key] = total.get(key, 0) + value
            lane["processed_games"].append(game_id)
            if seed not in lane["training_seeds"]:
                lane["training_seeds"].append(seed)
            self._save()
            return True

    def candidates(self, scope: str) -> list[tuple[RoleStrategy, str]]:
        with self.transaction():
            lane = self._lane(scope)
            proposals = []
            for role_name, value in lane["active"].items():
                role, current = Role.coerce(role_name), RoleStrategy.from_dict(value)
                lane["revision"] += 1
                candidate = propose_strategy(current, lane["feedback"].get(role.value, {}), lane["revision"])
                if candidate is not None:
                    proposals.append((candidate, current.version))
            self._save()
            return proposals

    def reserve_validation_seeds(self, scope: str, count: int) -> tuple[int, ...]:
        with self.transaction():
            if not 1 <= count <= 256:
                raise ValueError("validation seed count must be between 1 and 256")
            lane = self._lane(scope)
            used = set(lane["training_seeds"]) | set(lane["validation_seeds"])
            values = []
            nonce = len(lane["validation_seeds"])
            while len(values) < count:
                raw = hashlib.sha256(f"{scope}:{lane['revision']}:{nonce}".encode()).digest()
                seed = int.from_bytes(raw[:4], "big") + 1_000_000
                nonce += 1
                if seed not in used:
                    used.add(seed)
                    values.append(seed)
            lane["validation_seeds"].extend(values)
            self._save()
            return tuple(values)

    def commit(self, scope: str, candidate: RoleStrategy, result: Mapping[str, Any]) -> dict[str, Any]:
        with self.transaction():
            lane = self._lane(scope)
            current = lane["active"][candidate.role.value]
            record = {**deepcopy(dict(result)), "role": candidate.role.value, "candidate": candidate.to_dict()}
            record["feedback_snapshot"] = deepcopy(lane["feedback"].get(candidate.role.value, {}))
            if result["status"] == "accepted" and current["version"] == result["baseline_version"]:
                opponents = result.get("opponent_versions", {})
                evidence = validation_gate(result.get("outcomes", []), result.get("alpha", 0.05))
                if any(lane["active"][role]["version"] != version for role, version in opponents.items()):
                    record["status"], record["reason"] = "rejected", "opponent_versions_changed"
                elif evidence["status"] != "accepted":
                    record["status"], record["reason"] = "rejected", "invalid_validation_evidence"
                else:
                    lane["active"][candidate.role.value] = candidate.to_dict()
                    # 旧版本的错误不能永久推动新版本继续朝同一方向调参。
                    lane["feedback"].pop(candidate.role.value, None)
            elif result["status"] == "accepted":
                record["status"], record["reason"] = "rejected", "baseline_version_changed"
            lane["history"].append(record)
            lane["history"] = lane["history"][-100:]
            self._save()
            return deepcopy(record)

    def summary(self, scope: str) -> dict[str, Any]:
        with self.transaction():
            return self._summary(scope, self._lane(scope))

    @staticmethod
    def _summary(scope: str, lane: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "scope": scope, "active": deepcopy(lane["active"]),
            "feedback": deepcopy(lane["feedback"]), "history": deepcopy(lane["history"]),
            "training_game_count": len(lane["processed_games"]),
        }

    def summaries(self) -> dict[str, Any]:
        with self.transaction():
            return {scope: self._summary(scope, lane) for scope, lane in self.data["scopes"].items()}


def profile_factory(profiles: Mapping[Role, RoleStrategy], provider: LLMProvider | None = None):
    """每次新建独立记忆，仅注入经过白名单验证的策略副本。"""
    frozen = {role: RoleStrategy.from_dict(value.to_dict()) for role, value in profiles.items()}

    def create(player_id: str, role: Role) -> BaseAgent:
        return create_agent(player_id, role, provider=provider, strategy=frozen[role])

    return create


def validation_gate(pairs: Sequence[Mapping[str, Any]], alpha: float = 0.05) -> dict[str, Any]:
    """成对胜负差 + 精确符号检验；样本不足、无提升或安全退化绝不晋级。"""
    if not 0 < alpha <= 0.05:
        raise ValueError("alpha must be between 0 and 0.05")
    for pair in pairs:
        if pair["baseline"] not in (0, 0.5, 1) or pair["candidate"] not in (0, 0.5, 1):
            raise ValueError("invalid paired outcome")
        if not math.isfinite(pair["quality_delta"]) or not -1 <= pair["quality_delta"] <= 1 or type(pair["unsafe"]) is not bool:
            raise ValueError("invalid quality or safety evidence")
    count = len(pairs)
    wins = sum(pair["candidate"] > pair["baseline"] for pair in pairs)
    losses = sum(pair["candidate"] < pair["baseline"] for pair in pairs)
    discordant = wins + losses
    p_value = sum(math.comb(discordant, k) for k in range(wins, discordant + 1)) / (2 ** discordant) if discordant else 1.0
    delta = sum(pair["candidate"] - pair["baseline"] for pair in pairs) / count if count else 0.0
    quality_delta = sum(pair["quality_delta"] for pair in pairs) / count if count else 0.0
    unsafe = any(pair["unsafe"] for pair in pairs)
    if unsafe:
        status, reason = "rejected", "safety_regression"
    elif count < MIN_VALIDATION_PAIRS:
        status, reason = "inconclusive", "insufficient_pairs"
    elif delta < 0.05:
        status, reason = "rejected", "no_meaningful_win_rate_improvement"
    elif quality_delta < -0.02:
        status, reason = "rejected", "role_contribution_regression"
    elif p_value > alpha:
        status, reason = "inconclusive", "insufficient_statistical_evidence"
    else:
        status, reason = "accepted", "paired_improvement"
    return {
        "status": status, "reason": reason, "pairs": count, "paired_wins": wins,
        "paired_losses": losses, "win_rate_delta": round(delta, 6),
        "quality_delta": round(quality_delta, 6), "p_value": round(p_value, 8),
        "alpha": alpha,
    }


@dataclass
class EvolutionRunner:
    store: ExperienceStore
    config: GameConfig | None = None
    provider: LLMProvider | None = None
    allow_paid_validation: bool = False
    max_steps: int = 10_000

    def __post_init__(self) -> None:
        self.config = experiment_config(self.config)
        self.scope = scope_key(self.config, self.provider)

    def _play(self, seed: int, profiles: Mapping[Role, RoleStrategy]) -> tuple[WerewolfGame, dict[str, Any]]:
        value = self.config.to_dict()
        value["seed"] = seed
        game = WerewolfGame(value, agent_factory=profile_factory(profiles, self.provider))
        game.run(max_steps=self.max_steps)
        if game.result()["status"] != "finished":
            raise RuntimeError("evolution game did not finish within max_steps")
        report = evaluate_game(game.event_log(omniscient=True), game.result(), game.winner)
        return game, report

    def train(self, seeds: Iterable[int]) -> list[dict[str, Any]]:
        """训练局可反馈，验证局不可反馈；训练轮之间自然继承已晋级经验。"""
        if self.provider is not None and not self.allow_paid_validation:
            raise ValueError("model evolution requires allow_paid_validation=True")
        feedbacks = []
        for seed in seeds:
            self.store.check_seeds(self.scope, [seed], validation=False)
            profiles = self.store.profiles(self.scope)
            game, report = self._play(int(seed), profiles)
            feedback = game_feedback(game, report)
            self.store.record(self.scope, "training-" + uuid4().hex, int(seed), feedback,
                              versions={role.value: value.version for role, value in profiles.items()})
            for agent in game.agents.values():
                for lesson in _feedback_lessons(agent.role, feedback):
                    agent.memory.add_reflection(lesson)
            feedbacks.append(feedback)
        return feedbacks

    def validate(self, candidate: RoleStrategy, seeds: Sequence[int], alpha: float = 0.05) -> dict[str, Any]:
        if self.provider is not None and not self.allow_paid_validation:
            raise ValueError("paid model validation is not authorized")
        if len(set(seeds)) != len(seeds):
            raise ValueError("validation seeds must be unique")
        self.store.check_seeds(self.scope, seeds, validation=True)
        incumbent = self.store.profiles(self.scope)
        challenger = {**incumbent, candidate.role: candidate}
        pairs = []
        for seed in seeds:
            baseline_game, baseline = self._play(seed, incumbent)
            candidate_game, changed = self._play(seed, challenger)
            team = candidate.role.team.value
            def outcome(game):
                return 0.5 if game.winner == "draw" else float(game.winner == team)
            def quality(report):
                values = [metric["contribution_score"] for metric in report["player_metrics"].values() if metric["role"] == candidate.role.value]
                if not values:
                    raise ValueError("candidate role is not present in the validation board")
                return sum(values) / len(values)
            unsafe = changed["process_metrics"]["information_leakage"]["leak_count"] > 0 or any(
                event.event_type == "agent_action_repaired" for event in candidate_game.events
            )
            def fallback_rate(game):
                attempts = sum(agent.decision_stats["model_attempts"] for agent in game.agents.values())
                fallbacks = sum(agent.decision_stats["model_fallbacks"] for agent in game.agents.values())
                return fallbacks / attempts if attempts else 0.0
            baseline_fallback, candidate_fallback = fallback_rate(baseline_game), fallback_rate(candidate_game)
            planner_repairs = sum(agent.decision_stats["planner_repairs"] for agent in candidate_game.agents.values())
            unsafe = unsafe or candidate_fallback > baseline_fallback or planner_repairs > 0
            pairs.append({
                "seed": seed, "baseline": outcome(baseline_game), "candidate": outcome(candidate_game),
                "quality_delta": quality(changed) - quality(baseline), "unsafe": unsafe,
                "baseline_fallback_rate": baseline_fallback, "candidate_fallback_rate": candidate_fallback,
                "planner_repair_count": planner_repairs,
            })
        return {
            **validation_gate(pairs, alpha), "baseline_version": incumbent[candidate.role].version,
            "opponent_versions": {role.value: value.version for role, value in incumbent.items()},
            "validation_seeds": list(seeds), "outcomes": pairs,
            "mode": str(getattr(self.provider, "model", "configured-model")) if self.provider else "heuristic",
        }

    def optimize(self, validation_games: int = MIN_VALIDATION_PAIRS) -> list[dict[str, Any]]:
        if not 1 <= validation_games <= 256:
            raise ValueError("validation_games must be between 1 and 256")
        if self.provider is not None and not self.allow_paid_validation:
            raise ValueError("paid model validation is not authorized")
        proposals = self.store.candidates(self.scope)
        results = []
        # 同一轮多个候选使用 Bonferroni 门槛，不能反复试到一个偶然好结果就晋级。
        alpha = 0.05 / max(1, len(proposals))
        for candidate, parent_version in proposals:
            seeds = self.store.reserve_validation_seeds(self.scope, validation_games)
            try:
                result = self.validate(candidate, seeds, alpha)
            except Exception:
                # 不序列化 Provider 异常或未知对象，避免报告带出凭据。
                result = {"status": "rejected", "reason": "validation_failed",
                          "baseline_version": parent_version, "pairs": 0}
            committed = self.store.commit(self.scope, candidate, result)
            results.append({"candidate_version": candidate.version, **committed})
        return results

    def run(self, seeds: Iterable[int], rounds: int = 1, validation_games: int = MIN_VALIDATION_PAIRS) -> dict[str, Any]:
        if not 1 <= rounds <= 20:
            raise ValueError("rounds must be between 1 and 20")
        seeds = tuple(int(seed) for seed in seeds)
        if not seeds:
            raise ValueError("at least one training seed is required")
        iterations = []
        for iteration in range(rounds):
            feedback = self.train(seeds)
            updates = self.optimize(validation_games)
            iterations.append({"iteration": iteration + 1, "feedback": feedback, "updates": updates})
        return {"schema_version": "1.0", "scope": self.scope, "iterations": iterations,
                "final": self.store.summary(self.scope)}


def _feedback_lessons(role: Role, feedback: Mapping[str, Any]) -> list[str]:
    from ..agents.strategy import LESSONS
    own = feedback.get(role.value, {})
    if own.get("losses") or own.get("wrong_votes") or own.get("wrong_abilities") or own.get("inconsistent_votes"):
        return [LESSONS["vote_evidence"], LESSONS[ROLE_LESSON[role]]]
    return []


class EvolutionCoordinator:
    """连接实时游戏与版本库；仅本地模式自动进行后台验证。"""

    def __init__(self, store: ExperienceStore) -> None:
        self.store = store
        self.lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}

    def feedback(self, game_id: str, game: WerewolfGame, report: Mapping[str, Any], provider=None) -> dict[str, Any]:
        scope = scope_key(game.config, provider)
        feedback = game_feedback(game, report)
        versions = {agent.role.value: agent.memory.experience.get("version", "v0") for agent in game.agents.values()}
        recorded = self.store.record(scope, game_id, game.config.seed, feedback, versions=versions)
        if recorded:
            for agent in game.agents.values():
                for lesson in _feedback_lessons(agent.role, feedback):
                    agent.memory.add_reflection(lesson)
        return {"scope": scope, "recorded": recorded, "feedback": feedback,
                "lessons": {role.value: _feedback_lessons(role, feedback) for role in Role}}

    def schedule(self, config: GameConfig, provider=None) -> str:
        scope = scope_key(config, provider)
        if provider is not None:
            return "requires_paid_validation"
        with self.lock:
            if self._threads.get(scope) and self._threads[scope].is_alive():
                self._jobs[scope]["pending_feedback"] = True
                return "running"
            self._jobs[scope] = {"status": "running"}
            def work():
                try:
                    updates = []
                    while True:
                        updates.extend(EvolutionRunner(self.store, config).optimize())
                        with self.lock:
                            if self._jobs[scope].pop("pending_feedback", False):
                                continue
                            self._jobs[scope] = {"status": "finished", "updates": updates[-100:]}
                            # 完成标记和 thread 索引在同一锁内更新，避免漏掉紧接着的新反馈。
                            self._threads.pop(scope, None)
                            return
                except Exception:
                    result = {"status": "error", "reason": "evolution_validation_failed"}
                with self.lock:
                    self._jobs[scope] = result
                    self._threads.pop(scope, None)
            thread = threading.Thread(target=work, daemon=True, name="werewolf-evolution")
            self._threads[scope] = thread
            thread.start()
        return "running"

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "jobs": deepcopy(self._jobs),
                "scopes": self.store.summaries(),
            }

    def wait(self, timeout: float = 60.0) -> None:
        with self.lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="狼人杀对局、复盘、优化、验证与经验继承")
    parser.add_argument("--seeds", default="5,42,31415", help="训练种子，逗号分隔；验证种子单独生成")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--validation-games", type=int, default=MIN_VALIDATION_PAIRS)
    parser.add_argument("--player-count", type=int, default=8)
    parser.add_argument("--no-sheriff", action="store_true")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--output")
    parser.add_argument("--deepseek", action="store_true")
    parser.add_argument("--allow-paid-validation", action="store_true")
    args = parser.parse_args(argv)
    if args.deepseek and not args.allow_paid_validation:
        parser.error("DeepSeek 批量训练/验证可能产生大量费用，必须显式提供 --allow-paid-validation")
    provider = None
    if args.deepseek:
        from ..llm import build_deepseek_provider_from_env
        provider = build_deepseek_provider_from_env()
        if provider is None:
            parser.error("DeepSeek Provider 尚未配置")
    try:
        runner = EvolutionRunner(
            ExperienceStore(args.store),
            GameConfig(player_count=args.player_count, enable_sheriff=not args.no_sheriff),
            provider, args.allow_paid_validation,
        )
        report = runner.run(tuple(int(seed) for seed in args.seeds.split(",")), args.rounds, args.validation_games)
        if args.output:
            write_json_atomic(args.output, report)
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, OSError):
        parser.error("进化配置或存储无效，请检查参数、种子隔离和策略文件")


if __name__ == "__main__":
    raise SystemExit(main())
