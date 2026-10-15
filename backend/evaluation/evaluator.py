"""单局评测入口：组合标准化、指标与复盘，产出 Schema 1.1。"""

from __future__ import annotations
from collections import Counter
from typing import Any, Mapping
from .common import (
    _canonical_events,
    _extract_winner,
    _mapping,
    _normalize_players,
    _rate,
    _text,
)
from .metrics import _exile_vote_accuracy, _information_leakage, _player_metrics, _seer_check_utilization, _sheriff_metrics, _speech_vote_consistency, _wolf_coordination
from .replay import _build_replay, _turning_points


def evaluate_game(
    game_log: Any,
    final_players: Any = None,
    winner: Any = None,
    *,
    game_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """只评测本项目日志；不修改输入，不调用大模型。"""
    # 第一阶段：把事件、玩家和终局字段标准化。这里保留缺失值，绝不根据角色
    # 常识“脑补”未记录的行动，否则旧回放会得到虚假的高分或低分。
    source = _mapping(game_log) if isinstance(game_log, Mapping) else {}
    if source and "events" not in source and not {"state", "players"}.intersection(source):
        raise ValueError("project replay requires events; external record formats are unsupported")
    event_source = source.get("events", []) if isinstance(game_log, Mapping) else game_log
    state_source = _mapping(final_players) if isinstance(final_players, Mapping) and "players" in final_players else {}
    if final_players is None:
        state_source = source
        final_players = source.get("players") or _mapping(source.get("state")).get("players")
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
        "survival_rate": _rate(len(survivors), len(players)),
        "winning_players": winning_players,
    }

    # 第二阶段：所有过程指标都从同一份规范事件派生。各辅助函数只负责一个
    # 维度，便于单独测试，也让新增指标不会悄悄改变已有综合口径。
    leakage = _information_leakage(events)
    consistency = _speech_vote_consistency(events)
    seer_utilization = _seer_check_utilization(events, players)
    wolf_coordination = _wolf_coordination(events, players)
    exile_accuracy = _exile_vote_accuracy(events, players)
    sheriff_metrics, sheriff_player_metrics = _sheriff_metrics(events, players)
    player_metrics = _player_metrics(events, players, resolved_winner, seer_utilization)
    for player_id, metric in player_metrics.items():
        metric.update(sheriff_player_metrics.get(player_id, {}))
    process_metrics = {
        "information_leakage": leakage,
        "speech_vote_consistency": consistency,
        "seer_check_utilization": seer_utilization,
        "wolf_coordination": wolf_coordination,
        "exile_vote_accuracy": exile_accuracy,
        "sheriff": sheriff_metrics,
        # Named aliases make the 1.1 additions easy for downstream
        # dashboards to consume without traversing the grouped object.
        "sheriff_election": sheriff_metrics["election"],
        "sheriff_leadership": sheriff_metrics["leadership"],
        "badge_flow_execution": sheriff_metrics["badge_flow"],
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
    # 第三阶段：指标与原始事件序号重新连接，生成可追溯的转折点和中文复盘。
    # 复盘属于规则归因，不把相关性包装成模型推断出的严格因果关系。
    turning_points = _turning_points(events, players, resolved_winner)
    replay = _build_replay(result_metrics, process_metrics, player_metrics, turning_points)
    return {
        "schema_version": "1.1",
        "game_id": game_id or _text(combined_source.get("game_id")) or None,
        "result_metrics": result_metrics,
        "process_metrics": process_metrics,
        "player_metrics": player_metrics,
        "turning_points": turning_points,
        "replay": replay,
        "metadata": dict(metadata or {}),
    }
