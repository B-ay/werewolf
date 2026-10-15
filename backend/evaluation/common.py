"""本项目日志契约、一次性事件索引和共享统计函数。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from ..game.models import GameEvent, PlayerState, Role


def _mapping(value: Any) -> dict[str, Any]:
    """只接受本项目字典和领域对象，不再反射任意对象。"""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, GameEvent):
        return value.to_dict()
    if isinstance(value, PlayerState):
        return value.to_private_dict()
    raise TypeError("evaluation expects a mapping, GameEvent or PlayerState")


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """读取明确列出的本项目字段；None 与缺失保持原有回退语义。"""
    return next((mapping[key] for key in keys if mapping.get(key) is not None), default)


def _text(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value).strip() if value is not None else ""


def _norm(value: Any) -> str:
    return _text(value).lower()


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))

def _round_metric(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _rate(numerator: float, denominator: float) -> float | None:
    """零样本表示不可测，不把缺失数据当成零表现。"""
    return _round_metric(numerator / denominator) if denominator else None


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return _round_metric(statistics.fmean(numeric)) if numeric else None


def _win_rates(stats: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    return {key: {**value, "win_rate": _rate(value["wins"], value["games"])} for key, value in sorted(stats.items())}


def normalize_faction(value: Any) -> str | None:
    token = _norm(value)
    if token in {"werewolf", "werewolves"}:
        return "werewolves"
    if token in {"village", "not_werewolf", "villager", "seer", "witch", "hunter"}:
        return "village"
    return "draw" if token == "draw" else None


@dataclass(frozen=True)
class _Event:
    index: int
    seq: int
    round: int | None
    phase: str
    event_type: str
    actor: str | None
    target: str | None
    message: str
    data: dict[str, Any]
    visibility: tuple[str, ...]
    raw: dict[str, Any]


# 保留本项目历史口径：技能与对应 passed 事件原来也进入死亡相关分类。
# 此重构不纠正计分规则，以免既有报告和 Elo 发生隐式变化。
_SPEECH_TYPES = {"speech", "sheriff_speech", "sheriff_pk_speech", "sheriff_summary", "day_pk_speech"}
_DEATH_TYPES = {"death", "death_detail", "witch_poison", "witch_poison_passed", "hunter_shot"}
_BALLOT_TYPES = {"vote_cast", "sheriff_vote_cast", "wolf_vote"}


def _is_public(event: _Event) -> bool:
    return "public" in event.visibility


def _is_speech(event: _Event) -> bool:
    return event.event_type in _SPEECH_TYPES


def _is_ballot_decision(event: _Event) -> bool:
    return event.event_type in _BALLOT_TYPES and bool(event.actor and event.target)


def _is_sheriff_ballot_event(event: _Event) -> bool:
    return event.event_type == "sheriff_vote_cast" and bool(event.actor)


def _is_sheriff_ballot(event: _Event) -> bool:
    return _is_sheriff_ballot_event(event) and bool(event.target) and not event.data.get("abstained")


def _is_wolf_vote(event: _Event) -> bool:
    return event.event_type == "wolf_vote" and bool(event.actor and event.target)


def _is_exile_vote(event: _Event) -> bool:
    return event.event_type == "vote_cast" and bool(event.actor and event.target)


def _is_primary_exile_vote(event: _Event) -> bool:
    return _is_exile_vote(event) and event.phase != "day_pk_vote"


def _is_consistency_speech(event: _Event) -> bool:
    return _is_speech(event) and event.event_type not in {"sheriff_speech", "sheriff_pk_speech"} and event.phase not in {"sheriff_speech", "sheriff_pk_speech"}


def _is_check(event: _Event) -> bool:
    return event.event_type == "seer_check" and bool(event.actor and event.target)


def _is_death(event: _Event) -> bool:
    return event.event_type in _DEATH_TYPES


class _EventIndex(list[_Event]):
    """保留原事件顺序；按类型、行动者及票型建立索引，指标不再重复分类。"""

    def __init__(self, events: Iterable[_Event]) -> None:
        super().__init__(events)
        self.by_type: dict[str, list[_Event]] = defaultdict(list)
        self.by_actor: dict[str, list[_Event]] = defaultdict(list)
        self.by_round: dict[int | None, list[_Event]] = defaultdict(list)
        self.groups: dict[str, list[_Event]] = defaultdict(list)
        selectors = {
            "public": _is_public, "speech": _is_speech, "check": _is_check,
            "wolf_vote": _is_wolf_vote, "exile_vote": _is_exile_vote,
            "primary_exile_vote": _is_primary_exile_vote, "sheriff_ballot": _is_sheriff_ballot,
            "death": _is_death,
        }
        for event in self:
            self.by_type[event.event_type].append(event)
            if event.actor:
                self.by_actor[event.actor].append(event)
            self.by_round[event.round].append(event)
            for group, selector in selectors.items():
                if selector(event):
                    self.groups[group].append(event)


def _canonical_event(raw_event: GameEvent | Mapping[str, Any], index: int) -> _Event:
    raw = _mapping(raw_event)
    if not isinstance(raw.get("event_type"), str):
        raise ValueError("event_type is required; external event formats are unsupported")
    data = _mapping(raw.get("data"))
    round_value = raw.get("round")
    try:
        seq = int(raw.get("seq", index))
        round_number = int(round_value) if round_value is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("event seq and round must be integers") from exc
    visibility = raw.get("visibility", ("public",))
    if isinstance(visibility, str):
        visibility = (visibility,)
    elif not isinstance(visibility, (list, tuple)):
        raise TypeError("event visibility must be a string or sequence")
    return _Event(
        index, seq, round_number, _norm(raw.get("phase")), _norm(raw["event_type"]),
        _text(raw.get("actor_id")) or None,
        _text(_first(raw, "target_id", default=data.get("target_id"))) or None,
        _text(raw.get("message")), data, tuple(_norm(item) for item in visibility), raw,
    )


def _canonical_events(events: Iterable[GameEvent | Mapping[str, Any]] | None) -> _EventIndex:
    if events is None:
        return _EventIndex([])
    if isinstance(events, (str, bytes, Mapping)) or not isinstance(events, Iterable):
        raise TypeError("events must be an iterable of project events")
    return _EventIndex(_canonical_event(event, index) for index, event in enumerate(events))


def _normalize_players(players: Any) -> dict[str, dict[str, Any]]:
    if players is None:
        return {}
    if isinstance(players, Mapping):
        iterable = []
        for key, value in players.items():
            player = _mapping(value)
            player.setdefault("id", key)
            iterable.append(player)
    elif isinstance(players, Iterable) and not isinstance(players, (str, bytes)):
        iterable = [_mapping(player) for player in players]
    else:
        raise TypeError("players must be a list or a mapping keyed by player id")
    result = {}
    for player in iterable:
        player_id = _text(player.get("id"))
        if not player_id:
            raise ValueError("player id is required")
        role = _text(player.get("role")) or None
        if role is not None and role not in {item.value for item in Role}:
            raise ValueError("unsupported role: " + role)
        alive, won = player.get("alive"), player.get("won")
        result[player_id] = {
            **player, "id": player_id, "name": _text(player.get("name")) or player_id,
            "role": role, "team": normalize_faction(_first(player, "team", default=role)),
            "alive": bool(alive) if alive is not None else None,
            "won": bool(won) if won is not None else None,
        }
    return result


def _event_victim(event: _Event) -> str | None:
    return event.target or (event.actor if _is_death(event) else None)


def _extract_winner(
    explicit_winner: Any,
    source: Mapping[str, Any],
    events: Sequence[_Event],
    players: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    winner = normalize_faction(explicit_winner)
    reason = _text(_first(source, "win_reason", "reason", default="")) or None
    if winner is None:
        winner = normalize_faction(source.get("winner"))
    for event in reversed(events):
        if winner is not None:
            break
        if event.event_type == "game_finished":
            winner = normalize_faction(
                event.data.get("winner")
            )
            reason = reason or (_text(event.data.get("reason")) or None)
    if winner is None:
        won_teams = {
            player.get("team")
            for player in players.values()
            if player.get("won") is True and player.get("team") in {"village", "werewolves"}
        }
        if len(won_teams) == 1:
            winner = next(iter(won_teams))
    if winner is None and players and all(player.get("alive") is not None for player in players.values()):
        alive_wolves = sum(player.get("alive") is True and player.get("team") == "werewolves" for player in players.values())
        alive_village = sum(player.get("alive") is True and player.get("team") == "village" for player in players.values())
        if alive_wolves == 0:
            winner = "village"
            reason = reason or "所有狼人出局"
        elif alive_wolves >= alive_village:
            winner = "werewolves"
            reason = reason or "狼人数量达到或超过好人"
    return winner, reason


def _extract_vote_intent(event: _Event) -> str | None:
    intent = _first(event.data, "vote_intent", "sheriff_nomination")
    if intent is not None:
        return _text(intent) or None
    # 旧版本地 Agent 有英文发言，因此保留中英文文本提取，不扩展外部日志字段。
    match = re.search(r"(?:vote(?:\s+for)?|投票给|投给|票给|归票)\s*[:：]?\s*@?([\w\-.\u4e00-\u9fff]+)", event.message, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _extract_check_result(event: _Event, players: Mapping[str, Mapping[str, Any]]) -> str | None:
    result = normalize_faction(event.data.get("result"))
    return result if result is not None else players.get(event.target or "", {}).get("team")


def _mentions_target(event: _Event, target: str) -> bool:
    return bool(target and target.lower() in event.message.lower())
