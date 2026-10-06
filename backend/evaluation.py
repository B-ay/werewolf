"""Deterministic evaluation, replay analysis, and Elo ranking for Werewolf games.

The evaluator deliberately depends only on dictionaries and the Python standard
library.  It understands the engine's canonical event schema::

    {
        "seq": 12,
        "round": 2,
        "phase": "day_vote",
        "event_type": "vote_cast",
        "actor_id": "p1",
        "target_id": "p4",
        "message": "...",
        "data": {...},
        "visibility": ["public"],
    }

Common aliases (``type``, ``day``, ``actor``, ``target`` and so on) are also
accepted.  Missing data is represented by ``None`` plus a sample/coverage
count rather than being silently scored as zero.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
import math
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


WOLF_ROLES = {
    "wolf",
    "werewolf",
    "wolves",
    "werewolves",
    "alpha_wolf",
    "wolf_king",
    "狼人",
    "狼王",
}
VILLAGE_ROLES = {
    "villager",
    "village",
    "seer",
    "prophet",
    "witch",
    "hunter",
    "guard",
    "idiot",
    "村民",
    "平民",
    "预言家",
    "女巫",
    "猎人",
    "守卫",
    "白痴",
}

_SPEECH_WORDS = ("speech", "speak", "statement", "message", "discussion", "发言", "陈述")
_VOTE_WORDS = ("vote", "ballot", "投票", "票型")
_CHECK_WORDS = ("seer_check", "investigate", "inspection", "divine", "查验", "验人", "预言")
_DEATH_WORDS = (
    "death",
    "dead",
    "eliminat",
    "execution",
    "execute",
    "lynch",
    "killed",
    "night_kill",
    "poison",
    "shot",
    "出局",
    "死亡",
    "处决",
    "放逐",
    "毒杀",
    "枪杀",
)


def _mapping(value: Any) -> dict[str, Any]:
    """Return a shallow dictionary for mappings/dataclasses/plain objects."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip()


def _norm(value: Any) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _contains_any(value: Any, needles: Iterable[str]) -> bool:
    normalized = _norm(value)
    return any(needle in normalized for needle in needles)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round_metric(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def normalize_faction(value: Any) -> str | None:
    """Normalize English/Chinese faction or role values."""

    token = _norm(value)
    if not token:
        return None
    if token in WOLF_ROLES or "werewolf" in token or token.startswith("wolf") or "狼人" in token:
        return "werewolves"
    if token in {"good", "good_team", "town", "townsfolk", "human", "humans", "好人", "村庄", "村庄阵营"}:
        return "village"
    if token in VILLAGE_ROLES:
        return "village"
    if token in {"draw", "tie", "平局", "none", "unknown"}:
        return "draw" if token in {"draw", "tie", "平局"} else None
    return token


def role_faction(role: Any) -> str | None:
    return normalize_faction(role)


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


def _canonical_event(raw_event: Any, index: int) -> _Event:
    raw = _mapping(raw_event)
    data = _mapping(raw.get("data") or raw.get("payload") or raw.get("metadata"))
    seq_value = _first(raw, "seq", "sequence", "index", default=index)
    try:
        seq = int(seq_value)
    except (TypeError, ValueError):
        seq = index

    round_value = _first(raw, "round", "day", "turn", "round_no", default=None)
    if round_value is None:
        round_value = _first(data, "round", "day", "turn", default=None)
    try:
        round_number = int(round_value) if round_value is not None else None
    except (TypeError, ValueError):
        match = re.search(r"\d+", _text(round_value))
        round_number = int(match.group()) if match else None

    event_type = _norm(_first(raw, "event_type", "type", "kind", "action", "name", default="unknown"))
    actor = _first(raw, "actor_id", "actor", "player_id", "source_id", "source", default=None)
    if actor is None:
        actor = _first(data, "actor_id", "actor", "voter_id", "player_id", "source_id", default=None)
    target = _first(raw, "target_id", "target", "victim_id", "candidate_id", default=None)
    if target is None:
        target = _first(data, "target_id", "target", "victim_id", "candidate_id", "vote_target", default=None)

    message = _text(_first(raw, "message", "text", "content", "utterance", default=""))
    if not message:
        message = _text(_first(data, "message", "text", "content", default=""))

    visibility_value = _first(raw, "visibility", "audience", "visible_to", default=None)
    if visibility_value is None:
        visibility_value = _first(data, "visibility", "audience", "visible_to", default=None)
    if isinstance(visibility_value, str):
        visibility = (_norm(visibility_value),)
    elif isinstance(visibility_value, Iterable) and not isinstance(visibility_value, (str, bytes, Mapping)):
        visibility = tuple(_norm(item) for item in visibility_value)
    else:
        # Most external logs omit visibility for public events.
        visibility = ("public",)

    return _Event(
        index=index,
        seq=seq,
        round=round_number,
        phase=_norm(_first(raw, "phase", "stage", default=_first(data, "phase", default=""))),
        event_type=event_type,
        actor=_text(actor) or None,
        target=_text(target) or None,
        message=message,
        data=data,
        visibility=visibility,
        raw=raw,
    )


def _canonical_events(events: Any) -> list[_Event]:
    if events is None:
        return []
    if isinstance(events, Mapping):
        events = events.get("events") or events.get("event_log") or events.get("log") or []
    if isinstance(events, (str, bytes)) or not isinstance(events, Iterable):
        return []
    return [_canonical_event(event, index) for index, event in enumerate(events)]


def _normalize_players(players: Any) -> dict[str, dict[str, Any]]:
    if players is None:
        return {}
    if isinstance(players, Mapping) and "players" in players:
        players = players.get("players")
    normalized: dict[str, dict[str, Any]] = {}
    if isinstance(players, Mapping):
        iterable = []
        for key, value in players.items():
            item = _mapping(value)
            item.setdefault("id", key)
            iterable.append(item)
    elif isinstance(players, Iterable) and not isinstance(players, (str, bytes)):
        iterable = [_mapping(value) for value in players]
    else:
        iterable = []

    for index, player in enumerate(iterable):
        player_id = _text(_first(player, "id", "player_id", "name", "seat", default=f"player_{index + 1}"))
        role = _first(player, "role", "identity", "character", default=None)
        team = normalize_faction(_first(player, "faction", "team", "camp", default=role))
        alive_value = _first(player, "alive", "is_alive", default=None)
        if alive_value is None:
            status = _norm(player.get("status"))
            alive = status not in {"dead", "eliminated", "out", "死亡", "出局"} if status else None
        else:
            alive = bool(alive_value)
        won_value = _first(player, "won", "is_winner", default=None)
        normalized[player_id] = {
            **player,
            "id": player_id,
            "name": _text(_first(player, "name", "display_name", default=player_id)),
            "role": _text(role) or None,
            "team": team,
            "alive": alive,
            "won": bool(won_value) if won_value is not None else None,
        }
    return normalized


def _event_is(event: _Event, words: Iterable[str]) -> bool:
    return _contains_any(event.event_type, words)


def _is_public(event: _Event) -> bool:
    return any(value in {"public", "all", "everyone", "global", "公开", "全体"} for value in event.visibility)


def _is_speech(event: _Event) -> bool:
    return _event_is(event, _SPEECH_WORDS)


def _is_vote(event: _Event) -> bool:
    if not _event_is(event, _VOTE_WORDS):
        return False
    # Tallies and result announcements do not represent an individual decision.
    if _contains_any(event.event_type, ("result", "tally", "summary", "结果", "统计")):
        return False
    return bool(event.actor and event.target)


def _is_check(event: _Event) -> bool:
    return _event_is(event, _CHECK_WORDS) and bool(event.actor and event.target)


def _is_death(event: _Event) -> bool:
    return _event_is(event, _DEATH_WORDS)


def _event_victim(event: _Event) -> str | None:
    victim = _first(event.data, "victim_id", "dead_player_id", "eliminated_id", "player_id", default=event.target)
    if victim is None and _is_death(event):
        victim = event.actor
    return _text(victim) or None


def _extract_winner(
    explicit_winner: Any,
    source: Mapping[str, Any],
    events: Sequence[_Event],
    players: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    winner = normalize_faction(explicit_winner)
    reason = _text(_first(source, "win_reason", "reason", default="")) or None
    if winner is None:
        winner = normalize_faction(_first(source, "winner", "winning_faction", "winning_team", default=None))
    for event in reversed(events):
        if winner is not None:
            break
        if _contains_any(event.event_type, ("game_over", "game_end", "finished", "winner", "游戏结束", "胜利")):
            winner = normalize_faction(
                _first(event.data, "winner", "winning_faction", "winning_team", "faction", default=event.target)
            )
            reason = reason or (_text(_first(event.data, "reason", "win_reason", default="")) or None)
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
    intent = _first(
        event.data,
        "vote_intent",
        "intended_vote",
        "vote_target",
        "candidate_id",
        "nominee",
        default=None,
    )
    if intent is not None:
        if isinstance(intent, Mapping):
            intent = _first(intent, "id", "target", "player_id", default=None)
        return _text(intent) or None

    claims = event.data.get("claims")
    if isinstance(claims, Iterable) and not isinstance(claims, (str, bytes, Mapping)):
        for claim_value in claims:
            claim = _mapping(claim_value)
            stance = _norm(_first(claim, "stance", "action", "type", default=""))
            if stance in {"vote", "vote_for", "eliminate", "投票", "放逐"}:
                target = _first(claim, "target", "target_id", "player_id", default=None)
                if target is not None:
                    return _text(target)

    patterns = (
        r"(?:vote(?:\s+for)?|投票给|投给|票给|归票)\s*[:：]?\s*@?([\w\-.\u4e00-\u9fff]+)",
        r"(?:intend(?:\s+to)?\s+vote)\s+@?([\w\-.]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, event.message, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_check_result(event: _Event, players: Mapping[str, Mapping[str, Any]]) -> str | None:
    value = _first(event.data, "result", "alignment", "faction", "team", "camp", "role", default=None)
    if value is None and "is_wolf" in event.data:
        value = "werewolves" if bool(event.data["is_wolf"]) else "village"
    result = normalize_faction(value)
    if result is None and event.target in players:
        result = players[event.target].get("team")
    return result


def _mentions_target(event: _Event, target: str, expected_faction: str | None) -> bool:
    target_lower = target.lower()
    if target_lower and target_lower in event.message.lower():
        return True
    for key in ("claims", "reveals", "checked_players", "suspects", "trusted", "accused"):
        value = event.data.get(key)
        if isinstance(value, Mapping):
            value = [value]
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for item in value:
                if isinstance(item, Mapping):
                    item_target = _text(_first(item, "target", "target_id", "player_id", "id", default=""))
                    item_result = normalize_faction(_first(item, "result", "alignment", "faction", "team", "role", default=None))
                    if item_target == target and (expected_faction is None or item_result in {None, expected_faction}):
                        return True
                elif _text(item) == target:
                    return True
    return False


def _information_leakage(events: Sequence[_Event]) -> dict[str, Any]:
    public_events = [event for event in events if _is_public(event)]
    leaks: list[dict[str, Any]] = []
    hidden_keys = {
        "beliefs",
        "private_info",
        "private_information",
        "private_facts",
        "hidden_info",
        "secret",
        "secret_role",
        "wolf_teammates",
        "memory",
        "long_term_memory",
        "planner_state",
        "internal_reasoning",
        "role_prompt",
        "system_prompt",
        "hidden_state",
    }
    authorised_types = ("reveal", "death", "game_over", "role_public", "公开身份", "遗言")
    for event in public_events:
        explicit = bool(
            _first(
                event.raw,
                "information_leakage",
                "privacy_violation",
                "leak",
                default=_first(event.data, "information_leakage", "privacy_violation", "leak", default=False),
            )
        )
        scope = _norm(_first(event.data, "information_scope", "source_visibility", "intended_visibility", default=""))
        leaked_keys = sorted(hidden_keys.intersection(event.data))
        suspicious_scope = scope in {"private", "secret", "hidden", "wolf_only", "role_only", "私密"}
        authorised = bool(_first(event.data, "authorized_reveal", "authorised_reveal", default=False)) or _event_is(
            event, authorised_types
        )
        if explicit or ((suspicious_scope or leaked_keys) and not authorised):
            leaks.append(
                {
                    "event_index": event.index,
                    "seq": event.seq,
                    "round": event.round,
                    "actor_id": event.actor,
                    "reason": "显式标记为信息泄露" if explicit else "私密字段出现在公开事件中",
                    "fields": leaked_keys,
                }
            )
    denominator = len(public_events)
    leak_rate = len(leaks) / denominator if denominator else None
    return {
        "leak_count": len(leaks),
        "public_event_count": denominator,
        "leak_rate": _round_metric(leak_rate),
        "isolation_score": _round_metric(1.0 - leak_rate) if leak_rate is not None else None,
        "events": leaks,
    }


def _speech_vote_consistency(events: Sequence[_Event]) -> dict[str, Any]:
    latest_intent: dict[tuple[int | None, str], tuple[str, int]] = {}
    comparisons: list[dict[str, Any]] = []
    by_player: dict[str, list[bool]] = defaultdict(list)
    for event in events:
        if _is_speech(event) and event.actor:
            intent = _extract_vote_intent(event)
            if intent:
                latest_intent[(event.round, event.actor)] = (intent, event.index)
        if _is_vote(event) and event.actor and event.target:
            key = (event.round, event.actor)
            stated = latest_intent.get(key)
            if stated is None and event.round is None:
                candidates = [
                    value
                    for (round_no, actor), value in latest_intent.items()
                    if actor == event.actor and value[1] < event.index
                ]
                stated = max(candidates, key=lambda value: value[1], default=None)
            if stated is not None and stated[1] < event.index:
                consistent = stated[0] == event.target
                by_player[event.actor].append(consistent)
                comparisons.append(
                    {
                        "event_index": event.index,
                        "round": event.round,
                        "player_id": event.actor,
                        "stated_target": stated[0],
                        "actual_target": event.target,
                        "consistent": consistent,
                    }
                )
    consistent_count = sum(item["consistent"] for item in comparisons)
    sample_size = len(comparisons)
    return {
        "sample_size": sample_size,
        "consistent_count": consistent_count,
        "rate": _round_metric(consistent_count / sample_size) if sample_size else None,
        "coverage": _round_metric(sample_size / max(1, sum(_is_vote(event) for event in events))),
        "by_player": {
            player_id: {
                "sample_size": len(values),
                "consistent_count": sum(values),
                "rate": _round_metric(sum(values) / len(values)),
            }
            for player_id, values in sorted(by_player.items())
        },
        "comparisons": comparisons,
    }


def _seer_check_utilization(
    events: Sequence[_Event], players: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    checks = [event for event in events if _is_check(event)]
    details: list[dict[str, Any]] = []
    by_player: dict[str, list[bool]] = defaultdict(list)
    for check in checks:
        result = _extract_check_result(check, players)
        used_by: str | None = None
        use_type: str | None = None
        for later in events[check.index + 1 :]:
            if later.actor != check.actor:
                continue
            if _is_public(later) and _is_speech(later) and _mentions_target(later, check.target or "", result):
                used_by, use_type = later.event_type, "public_claim"
                break
            if result == "werewolves" and _is_vote(later) and later.target == check.target:
                used_by, use_type = later.event_type, "vote"
                break
        utilized = used_by is not None
        if check.actor:
            by_player[check.actor].append(utilized)
        details.append(
            {
                "event_index": check.index,
                "round": check.round,
                "seer_id": check.actor,
                "target_id": check.target,
                "result": result,
                "utilized": utilized,
                "use_type": use_type,
                "use_event_type": used_by,
            }
        )
    used_count = sum(item["utilized"] for item in details)
    return {
        "check_count": len(details),
        "utilized_count": used_count,
        "rate": _round_metric(used_count / len(details)) if details else None,
        "by_player": {
            player_id: {
                "check_count": len(values),
                "utilized_count": sum(values),
                "rate": _round_metric(sum(values) / len(values)),
            }
            for player_id, values in sorted(by_player.items())
        },
        "checks": details,
    }


def _wolf_coordination(
    events: Sequence[_Event], players: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    wolf_ids = {player_id for player_id, player in players.items() if player.get("team") == "werewolves"}
    grouped: dict[tuple[int | None, str], list[_Event]] = defaultdict(list)
    for event in events:
        if _is_vote(event) and event.actor in wolf_ids:
            grouped[(event.round, event.phase or "unknown")].append(event)

    rounds: list[dict[str, Any]] = []
    for (round_no, phase), votes in sorted(grouped.items(), key=lambda item: ((item[0][0] or -1), item[0][1])):
        # Keep the last vote by each wolf in the round/phase.
        final_votes: dict[str, str] = {}
        for vote in votes:
            if vote.actor and vote.target:
                final_votes[vote.actor] = vote.target
        if len(final_votes) < 2:
            continue
        counts = Counter(final_votes.values())
        modal_target, modal_count = counts.most_common(1)[0]
        agreement = modal_count / len(final_votes)
        rounds.append(
            {
                "round": round_no,
                "phase": phase,
                "wolf_vote_count": len(final_votes),
                "modal_target": modal_target,
                "agreement": _round_metric(agreement),
                "unanimous": modal_count == len(final_votes),
                "votes": dict(sorted(final_votes.items())),
            }
        )
    mean_agreement = sum(item["agreement"] for item in rounds) / len(rounds) if rounds else None
    return {
        "wolf_count": len(wolf_ids),
        "rounds_evaluated": len(rounds),
        "mean_agreement": _round_metric(mean_agreement),
        "unanimous_rounds": sum(item["unanimous"] for item in rounds),
        "rounds": rounds,
    }


def _role_action_effectiveness(
    player_id: str,
    player: Mapping[str, Any],
    events: Sequence[_Event],
    players: Mapping[str, Mapping[str, Any]],
    seer_metric: Mapping[str, Any],
) -> tuple[float | None, int, list[str]]:
    role = _norm(player.get("role"))
    team = player.get("team")
    successes = 0.0
    attempts = 0
    notes: list[str] = []
    own_events = [event for event in events if event.actor == player_id]

    if role in {"seer", "prophet", "预言家"}:
        checks = [event for event in own_events if _is_check(event)]
        utilized = seer_metric.get("by_player", {}).get(player_id, {}).get("utilized_count", 0)
        wolves_found = sum(_extract_check_result(event, players) == "werewolves" for event in checks)
        attempts = len(checks) * 2
        successes = utilized + wolves_found
        if checks:
            notes.append(f"查验{len(checks)}次，发现狼人{wolves_found}次，利用信息{utilized}次")
    elif role in {"witch", "女巫"}:
        for event in own_events:
            if _contains_any(event.event_type, ("save", "antidote", "解药", "救")):
                attempts += 1
                target_team = players.get(event.target or "", {}).get("team")
                successes += target_team != "werewolves"
            elif _contains_any(event.event_type, ("poison", "毒")):
                attempts += 1
                successes += players.get(event.target or "", {}).get("team") == "werewolves"
        if attempts:
            notes.append(f"女巫技能有效{int(successes)}/{attempts}次")
    elif role in {"hunter", "猎人"}:
        shots = [event for event in own_events if _contains_any(event.event_type, ("shot", "shoot", "枪", "开枪"))]
        attempts = len(shots)
        successes = sum(players.get(event.target or "", {}).get("team") == "werewolves" for event in shots)
        if attempts:
            notes.append(f"猎人命中狼人{int(successes)}/{attempts}次")
    elif team == "werewolves":
        kills = [
            event
            for event in own_events
            if _contains_any(event.event_type, ("kill", "attack", "袭击", "刀")) and event.target
        ]
        attempts = len(kills)
        successes = sum(players.get(event.target or "", {}).get("team") == "village" for event in kills)
        if attempts:
            notes.append(f"狼人行动命中好人{int(successes)}/{attempts}次")
    return (successes / attempts if attempts else None), attempts, notes


def _player_metrics(
    events: Sequence[_Event],
    players: Mapping[str, Mapping[str, Any]],
    winner: str | None,
    seer_metric: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    speech_counts = Counter(event.actor for event in events if _is_speech(event) and event.actor)
    action_counts = Counter(event.actor for event in events if event.actor)
    max_actions = max(action_counts.values(), default=0)
    votes_by_player: dict[str, list[_Event]] = defaultdict(list)
    for event in events:
        if _is_vote(event) and event.actor:
            votes_by_player[event.actor].append(event)

    result: dict[str, dict[str, Any]] = {}
    for player_id, player in sorted(players.items()):
        team = player.get("team")
        votes = votes_by_player.get(player_id, [])
        aligned_votes = sum(
            players.get(vote.target or "", {}).get("team") not in {None, team}
            for vote in votes
        )
        vote_accuracy = aligned_votes / len(votes) if votes else None
        action_effectiveness, ability_attempts, notes = _role_action_effectiveness(
            player_id, player, events, players, seer_metric
        )
        won = winner in {"village", "werewolves"} and team == winner
        alive = player.get("alive")
        participation = action_counts[player_id] / max_actions if max_actions else 0.0
        # Neutral priors avoid punishing roles for having no legal opportunity.
        vote_component = 0.5 if vote_accuracy is None else vote_accuracy
        action_component = 0.5 if action_effectiveness is None else action_effectiveness
        contribution = (
            0.45 * float(won)
            + 0.15 * float(alive is True)
            + 0.15 * vote_component
            + 0.15 * action_component
            + 0.10 * participation
        )
        result[player_id] = {
            "name": player.get("name", player_id),
            "role": player.get("role"),
            "team": team,
            "alive": alive,
            "won": won if winner is not None else None,
            "speech_count": speech_counts[player_id],
            "action_count": action_counts[player_id],
            "vote_count": len(votes),
            "opponent_vote_count": aligned_votes,
            "vote_accuracy": _round_metric(vote_accuracy),
            "ability_attempts": ability_attempts,
            "role_action_effectiveness": _round_metric(action_effectiveness),
            "participation": _round_metric(participation),
            "win_contribution": 1.0 if won else (0.0 if winner is not None else None),
            "survival_contribution": 1.0 if alive is True else (0.0 if alive is False else None),
            "contribution_score": _round_metric(_clamp(contribution)),
            "notes": notes,
        }
    return result


def _turning_points(
    events: Sequence[_Event], players: Mapping[str, Mapping[str, Any]], winner: str | None
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for event in events:
        explicit = _first(event.data, "pivotal", "turning_point", default=False)
        victim = _event_victim(event) if _is_death(event) else None
        summary: str | None = None
        impact = 0.0
        category = "explicit"
        if victim:
            victim_info = players.get(victim, {})
            role = _text(victim_info.get("role")) or "未知角色"
            team = victim_info.get("team")
            category = "elimination"
            if team == "werewolves":
                impact = 1.0
                summary = f"{victim}（{role}）出局，村庄阵营取得关键进展。"
            elif _norm(role) in {"seer", "prophet", "预言家"}:
                impact = -1.0
                summary = f"预言家 {victim} 出局，村庄失去持续查验来源。"
            elif _norm(role) in {"witch", "女巫", "hunter", "猎人"}:
                impact = -0.7
                summary = f"关键神职 {victim}（{role}）出局，村庄行动空间收窄。"
            elif team == "village":
                impact = -0.4
                summary = f"好人 {victim} 出局，狼人推进人数优势。"
        elif _contains_any(event.event_type, ("save", "antidote", "解药", "救人")):
            category = "ability"
            target_team = players.get(event.target or "", {}).get("team")
            impact = 0.7 if target_team != "werewolves" else -0.5
            summary = f"女巫救下 {event.target or '目标'}，改变了当夜减员结果。"
        elif _contains_any(event.event_type, ("shot", "shoot", "开枪", "枪杀")):
            category = "ability"
            target_team = players.get(event.target or "", {}).get("team")
            impact = 0.8 if target_team == "werewolves" else -0.8
            summary = f"猎人对 {event.target or '目标'} 开枪，直接改变阵营人数。"
        elif _contains_any(event.event_type, ("game_over", "game_end", "finished", "游戏结束")):
            category = "outcome"
            impact = 1.2 if winner == "village" else -1.2 if winner == "werewolves" else 0.0
            winner_name = "村庄" if winner == "village" else "狼人" if winner == "werewolves" else "未知"
            summary = f"对局结束，{winner_name}阵营获胜。"
        elif explicit:
            category = "explicit"
            try:
                impact = float(_first(event.data, "impact", "impact_score", default=0.5))
            except (TypeError, ValueError):
                impact = 0.5
            summary = _text(_first(event.data, "summary", "reason", default=event.message)) or "日志标记的关键事件。"
        if summary:
            points.append(
                {
                    "event_index": event.index,
                    "seq": event.seq,
                    "round": event.round,
                    "phase": event.phase,
                    "event_type": event.event_type,
                    "actor_id": event.actor,
                    "target_id": event.target or victim,
                    "category": category,
                    "village_impact": round(impact, 3),
                    "importance": round(abs(impact), 3),
                    "summary": summary,
                }
            )
    # Retain the twelve most important events, displayed in match order.
    if len(points) > 12:
        points = sorted(points, key=lambda item: (-item["importance"], item["event_index"]))[:12]
        points.sort(key=lambda item: item["event_index"])
    return points


def _build_replay(
    result_metrics: Mapping[str, Any],
    process_metrics: Mapping[str, Any],
    player_metrics: Mapping[str, Mapping[str, Any]],
    turning_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    winner = result_metrics.get("winner")
    winner_name = "村庄阵营" if winner == "village" else "狼人阵营" if winner == "werewolves" else "胜负未确定"
    rounds = result_metrics.get("duration_rounds")
    event_count = result_metrics.get("event_count", 0)
    summary = f"{winner_name}{'获胜' if winner else ''}，对局记录了{event_count}个事件"
    if rounds:
        summary += f"，历经{rounds}轮"
    summary += "。"

    attribution: list[str] = []
    leakage = process_metrics["information_leakage"]
    if leakage["leak_count"]:
        attribution.append(f"检测到{leakage['leak_count']}次公开信息泄露，信息隔离是本局可信度的主要风险。")
    elif leakage["public_event_count"]:
        attribution.append("未检测到公开私密字段，信息隔离规则执行稳定。")

    consistency = process_metrics["speech_vote_consistency"]
    if consistency["rate"] is not None:
        attribution.append(
            f"有明确投票承诺的样本中，发言与投票一致率为{consistency['rate']:.1%}（{consistency['sample_size']}个样本）。"
        )
    seer = process_metrics["seer_check_utilization"]
    if seer["rate"] is not None:
        attribution.append(f"预言家查验信息利用率为{seer['rate']:.1%}（{seer['utilized_count']}/{seer['check_count']}）。")
    wolf = process_metrics["wolf_coordination"]
    if wolf["mean_agreement"] is not None:
        attribution.append(f"狼人可比较轮次的平均投票聚合度为{wolf['mean_agreement']:.1%}。")

    ranked = sorted(
        player_metrics.items(), key=lambda item: (item[1].get("contribution_score") or 0.0), reverse=True
    )
    if ranked:
        leaders = "、".join(f"{data.get('name', player_id)}({data.get('contribution_score', 0):.2f})" for player_id, data in ranked[:3])
        attribution.append(f"综合胜负、存活、投票、技能与参与度，贡献度领先者为：{leaders}。")

    role_reviews: dict[str, str] = {}
    for player_id, metric in player_metrics.items():
        outcome = "获胜" if metric.get("won") is True else "落败" if metric.get("won") is False else "胜负未知"
        survival = "存活" if metric.get("alive") is True else "出局" if metric.get("alive") is False else "生存状态未知"
        vote_text = (
            f"对立阵营投票率{metric['vote_accuracy']:.1%}"
            if metric.get("vote_accuracy") is not None
            else "无可评估投票"
        )
        note = "；".join(metric.get("notes") or [])
        role_reviews[player_id] = (
            f"{metric.get('name', player_id)}（{metric.get('role') or '身份未知'}）{outcome}、{survival}，"
            f"{vote_text}，贡献分{metric.get('contribution_score', 0):.2f}" + (f"；{note}" if note else "。")
        )

    observed = sum(
        value is not None
        for value in (
            leakage.get("isolation_score"),
            consistency.get("rate"),
            seer.get("rate"),
            wolf.get("mean_agreement"),
        )
    )
    confidence = _round_metric(min(1.0, 0.35 + 0.15 * observed + 0.02 * min(event_count, 10)))
    return {
        "summary": summary,
        "attribution": attribution,
        "key_turning_points": list(turning_points),
        "role_reviews": role_reviews,
        "confidence": confidence,
        "method": "基于结构化日志的确定性规则归因；相关性不等同于因果性。",
    }


class GameEvaluator:
    """Evaluate a single match without model calls or hidden global state."""

    def evaluate(
        self,
        game_log: Any,
        final_players: Any = None,
        winner: Any = None,
        *,
        game_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = _mapping(game_log) if isinstance(game_log, Mapping) or is_dataclass(game_log) else {}
        event_source = (
            _first(source, "events", "event_log", "log", default=[])
            if source
            else game_log
        )
        state_source = _mapping(final_players)
        if final_players is None:
            state_source = source
            final_players = source.get("players") or source.get("final_players") or source.get("state", {}).get("players")
        elif "players" in state_source:
            final_players = state_source.get("players")
        events = _canonical_events(event_source)
        players = _normalize_players(final_players)

        combined_source = {**source, **state_source, **dict(metadata or {})}
        resolved_winner, win_reason = _extract_winner(winner, combined_source, events, players)
        rounds = sorted({event.round for event in events if event.round is not None})
        survivors = [player_id for player_id, player in players.items() if player.get("alive") is True]
        winning_players = [
            player_id for player_id, player in players.items() if resolved_winner and player.get("team") == resolved_winner
        ]
        team_counts = Counter(player.get("team") or "unknown" for player in players.values())
        survivor_counts = Counter(
            (player.get("team") or "unknown") for player in players.values() if player.get("alive") is True
        )
        result_metrics = {
            "winner": resolved_winner,
            "win_reason": win_reason,
            "event_count": len(events),
            "duration_rounds": max(rounds) if rounds else None,
            "rounds_observed": rounds,
            "participant_count": len(players),
            "team_counts": dict(sorted(team_counts.items())),
            "survivor_count": len(survivors),
            "survivors": survivors,
            "survivor_counts": dict(sorted(survivor_counts.items())),
            "survival_rate": _round_metric(len(survivors) / len(players)) if players else None,
            "winning_players": winning_players,
        }

        leakage = _information_leakage(events)
        consistency = _speech_vote_consistency(events)
        seer_utilization = _seer_check_utilization(events, players)
        wolf_coordination = _wolf_coordination(events, players)
        player_metrics = _player_metrics(events, players, resolved_winner, seer_utilization)
        process_metrics = {
            "information_leakage": leakage,
            "speech_vote_consistency": consistency,
            "seer_check_utilization": seer_utilization,
            "wolf_coordination": wolf_coordination,
            "survival_and_contribution": {
                "survival_rate": result_metrics["survival_rate"],
                "by_player": {
                    player_id: {
                        "won": metric["won"],
                        "alive": metric["alive"],
                        "win_contribution": metric["win_contribution"],
                        "survival_contribution": metric["survival_contribution"],
                        "contribution_score": metric["contribution_score"],
                    }
                    for player_id, metric in player_metrics.items()
                },
            },
        }
        turning_points = _turning_points(events, players, resolved_winner)
        replay = _build_replay(result_metrics, process_metrics, player_metrics, turning_points)
        return {
            "schema_version": "1.0",
            "game_id": game_id or _text(_first(combined_source, "game_id", "id", default="")) or None,
            "result_metrics": result_metrics,
            "process_metrics": process_metrics,
            "player_metrics": player_metrics,
            "turning_points": turning_points,
            "replay": replay,
            "metadata": dict(metadata or {}),
        }


def evaluate_game(
    game_log: Any,
    final_players: Any = None,
    winner: Any = None,
    *,
    game_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience entry point for :class:`GameEvaluator`."""

    return GameEvaluator().evaluate(
        game_log,
        final_players,
        winner,
        game_id=game_id,
        metadata=metadata,
    )


def analyze_game(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias retained for API callers that use ``analyze`` terminology."""

    return evaluate_game(*args, **kwargs)


def _participant_key(participant: Mapping[str, Any]) -> str:
    explicit = _first(participant, "leaderboard_key", "agent_key", "entry_id", default=None)
    if explicit is not None:
        return _text(explicit)
    agent = _text(_first(participant, "agent", "agent_id", "controller", "id", "player_id", "name", default="agent"))
    model = _text(_first(participant, "model", "model_name", default=""))
    version = _text(_first(participant, "version", "agent_version", "model_version", default=""))
    qualifiers = ":".join(value for value in (model, version) if value)
    return f"{agent}@{qualifiers}" if qualifiers else agent


class EloLeaderboard:
    """Aggregate agent/model-version results with multiplayer team Elo.

    Ratings are updated from all cross-team pairs using a scaled, zero-sum
    pairwise update.  Teammates are never compared against one another.
    """

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0) -> None:
        if k_factor <= 0:
            raise ValueError("k_factor must be positive")
        self.initial_rating = float(initial_rating)
        self.k_factor = float(k_factor)
        self._entries: dict[str, dict[str, Any]] = {}
        self.games_recorded = 0
        self.game_ids: set[str] = set()

    def _entry(self, key: str, participant: Mapping[str, Any]) -> dict[str, Any]:
        if key not in self._entries:
            self._entries[key] = {
                "key": key,
                "agent": _text(_first(participant, "agent", "agent_id", "controller", "id", default=key)),
                "model": _text(_first(participant, "model", "model_name", default="")) or None,
                "version": _text(_first(participant, "version", "agent_version", "model_version", default="")) or None,
                "rating": self.initial_rating,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "roles": defaultdict(lambda: {"games": 0, "wins": 0}),
                "factions": defaultdict(lambda: {"games": 0, "wins": 0}),
            }
        return self._entries[key]

    def record_game(
        self,
        participants: Any,
        winner: Any = None,
        *,
        game_id: str | None = None,
    ) -> dict[str, float]:
        """Record a match and return per-entry rating deltas.

        ``participants`` may be a list of dictionaries, an evaluation returned
        by :func:`evaluate_game`, or a record containing ``participants`` and
        ``winner``.  Participant dictionaries should contain a faction/team (or
        role) plus agent/model/version identity fields.
        """

        source = _mapping(participants)
        if source:
            if "player_metrics" in source:
                evaluation = source
                winner = winner or evaluation.get("result_metrics", {}).get("winner")
                participant_values = []
                metadata_by_player = _mapping(evaluation.get("participant_metadata"))
                for player_id, metric_value in _mapping(evaluation.get("player_metrics")).items():
                    metric = _mapping(metric_value)
                    participant_values.append({"id": player_id, **_mapping(metadata_by_player.get(player_id)), **metric})
                participants = participant_values
                game_id = game_id or evaluation.get("game_id")
            else:
                winner = winner or _first(source, "winner", "winning_faction", default=None)
                participants = source.get("participants") or source.get("players") or []
                game_id = game_id or source.get("game_id") or source.get("id")
        if game_id and game_id in self.game_ids:
            raise ValueError(f"game_id already recorded: {game_id}")
        if isinstance(participants, Mapping):
            participant_iterable = []
            for player_id, value in participants.items():
                item = _mapping(value)
                item.setdefault("id", player_id)
                participant_iterable.append(item)
        elif isinstance(participants, Iterable) and not isinstance(participants, (str, bytes)):
            participant_iterable = [_mapping(value) for value in participants]
        else:
            participant_iterable = []

        canonical: list[dict[str, Any]] = []
        seen_in_game: set[str] = set()
        for participant in participant_iterable:
            key = _participant_key(participant)
            # One model/version entry may control several seats.  Each entry is
            # counted once per game; duplicate seats are merged by faction.
            team = normalize_faction(_first(participant, "team", "faction", "camp", default=participant.get("role")))
            if not key or key in seen_in_game:
                continue
            seen_in_game.add(key)
            item = {**participant, "key": key, "team": team}
            canonical.append(item)
            self._entry(key, participant)
        if not canonical:
            return {}

        resolved_winner = normalize_faction(winner)
        pre_ratings = {item["key"]: self._entries[item["key"]]["rating"] for item in canonical}
        opponent_counts = {
            item["key"]: sum(other["team"] != item["team"] for other in canonical if other["team"] is not None)
            for item in canonical
        }
        deltas = {item["key"]: 0.0 for item in canonical}
        for left_index, left in enumerate(canonical):
            if left["team"] is None:
                continue
            for right in canonical[left_index + 1 :]:
                if right["team"] is None or left["team"] == right["team"]:
                    continue
                expected_left = 1.0 / (1.0 + math.pow(10.0, (pre_ratings[right["key"]] - pre_ratings[left["key"]]) / 400.0))
                if resolved_winner == "draw" or resolved_winner not in {left["team"], right["team"]}:
                    actual_left = 0.5
                else:
                    actual_left = 1.0 if resolved_winner == left["team"] else 0.0
                scale = self.k_factor / max(opponent_counts[left["key"]], opponent_counts[right["key"]], 1)
                delta = scale * (actual_left - expected_left)
                deltas[left["key"]] += delta
                deltas[right["key"]] -= delta

        for item in canonical:
            entry = self._entries[item["key"]]
            entry["rating"] += deltas[item["key"]]
            entry["games"] += 1
            if resolved_winner == "draw" or resolved_winner is None:
                entry["draws"] += 1
            elif item["team"] == resolved_winner:
                entry["wins"] += 1
            else:
                entry["losses"] += 1
            role = _text(item.get("role")) or "unknown"
            faction = item["team"] or "unknown"
            entry["roles"][role]["games"] += 1
            entry["factions"][faction]["games"] += 1
            if item["team"] == resolved_winner:
                entry["roles"][role]["wins"] += 1
                entry["factions"][faction]["wins"] += 1

        self.games_recorded += 1
        if game_id:
            self.game_ids.add(_text(game_id))
        return {key: round(delta, 4) for key, delta in deltas.items()}

    add_game = record_game

    def rankings(self, min_games: int = 0) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for entry in self._entries.values():
            if entry["games"] < min_games:
                continue
            result.append(
                {
                    "key": entry["key"],
                    "agent": entry["agent"],
                    "model": entry["model"],
                    "version": entry["version"],
                    "rating": round(entry["rating"], 2),
                    "games": entry["games"],
                    "wins": entry["wins"],
                    "losses": entry["losses"],
                    "draws": entry["draws"],
                    "win_rate": round(entry["wins"] / entry["games"], 4) if entry["games"] else None,
                    "roles": {
                        role: {
                            **stats,
                            "win_rate": round(stats["wins"] / stats["games"], 4) if stats["games"] else None,
                        }
                        for role, stats in sorted(entry["roles"].items())
                    },
                    "factions": {
                        faction: {
                            **stats,
                            "win_rate": round(stats["wins"] / stats["games"], 4) if stats["games"] else None,
                        }
                        for faction, stats in sorted(entry["factions"].items())
                    },
                }
            )
        result.sort(key=lambda item: (-item["rating"], -item["win_rate"], item["key"]))
        for rank, item in enumerate(result, 1):
            item["rank"] = rank
        return result

    def to_dict(self, min_games: int = 0) -> dict[str, Any]:
        return {
            "rating_system": "multiplayer_pairwise_elo",
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "games_recorded": self.games_recorded,
            "rankings": self.rankings(min_games=min_games),
        }


Leaderboard = EloLeaderboard


def build_leaderboard(
    games: Iterable[Any], *, initial_rating: float = 1500.0, k_factor: float = 32.0, min_games: int = 0
) -> dict[str, Any]:
    """Aggregate evaluations or participant game records into a leaderboard."""

    board = EloLeaderboard(initial_rating=initial_rating, k_factor=k_factor)
    for game in games:
        board.record_game(game)
    return board.to_dict(min_games=min_games)


aggregate_leaderboard = build_leaderboard


__all__ = [
    "GameEvaluator",
    "EloLeaderboard",
    "Leaderboard",
    "evaluate_game",
    "analyze_game",
    "build_leaderboard",
    "aggregate_leaderboard",
    "normalize_faction",
    "role_faction",
]
