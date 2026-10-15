"""结构化日志的转折点识别和中文复盘归因。"""

from __future__ import annotations
from typing import Any, Mapping, Sequence
from .common import (
    _Event,
    _event_victim,
    _first,
    _is_death,
    _round_metric,
    _text,
)
from .metrics import _badge_flow_from_event


def _turning_points(
    events: Sequence[_Event], players: Mapping[str, Mapping[str, Any]], winner: str | None
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for event in events:
        explicit = event.data.get("pivotal", False)
        flow = _badge_flow_from_event(event)
        victim = _event_victim(event) if _is_death(event) else None
        summary: str | None = None
        impact = 0.0
        category = "explicit"
        if event.event_type == "sheriff_elected":
            elected = _text(
                event.data.get("holder_id", event.target)
            ) or "未知玩家"
            info = players.get(elected, {})
            role = _text(info.get("role")) or "未知角色"
            category = "sheriff_election"
            impact = -0.6 if info.get("team") == "werewolves" else 0.6
            label = "悍跳狼" if info.get("team") == "werewolves" else role
            summary = f"{elected}（{label}）当选警长，获得发言组织、归票与 1.5 票权。"
        elif event.event_type == "sheriff_election_void":
            category = "sheriff_election"
            impact = -0.25
            reason = _text(event.data.get("reason", event.message)) or "平票或无人当选"
            summary = f"警长竞选流局（{reason}），本局此后不再产生警长。"
        elif event.event_type == "sheriff_order":
            category = "sheriff_leadership"
            impact = 0.2
            direction = _text(event.data.get("direction", event.target)) or "既定"
            summary = f"警长选择{direction}发言顺序，并保留最后总结位置。"
        elif event.event_type == "sheriff_summary":
            category = "sheriff_nomination"
            impact = 0.3
            nominee = _text(
                _first(event.data, "sheriff_nomination", "vote_intent", default=event.target)
            ) or "未指定目标"
            summary = f"警长总结全场并归票 {nominee}。"
        elif event.event_type == "badge_transferred":
            category = "badge_disposition"
            impact = 0.35
            source = _text(event.actor) or "原警长"
            target = _text(event.target) or "未知玩家"
            summary = f"{source} 将警徽移交给 {target}，警长权力继续传承。"
        elif event.event_type == "badge_destroyed":
            category = "badge_disposition"
            impact = -0.25
            source = _text(event.actor) or "原警长"
            summary = f"{source} 撕毁警徽，本局此后不再有警长。"
        elif event.event_type == "day_vote_result" and bool(
            event.data.get("sheriff_vote_decisive", False)
        ):
            category = "sheriff_weight"
            impact = 0.55
            eliminated = _text(
                event.data.get("eliminated", event.target)
            ) or "最终目标"
            summary = f"警长的 1.5 票改变了票型结果，{eliminated} 成为最终放逐结果。"
        elif flow:
            category = "badge_flow"
            impact = 0.25
            targets = "、".join(flow.get("targets") or [])
            summary = f"{event.actor or '玩家'} 公开编码{flow.get('mode')}警徽流，计划查验 {targets}。"
        elif victim:
            victim_info = players.get(victim, {})
            role = _text(victim_info.get("role")) or "未知角色"
            team = victim_info.get("team")
            category = "elimination"
            if team == "werewolves":
                impact = 1.0
                summary = f"{victim}（{role}）出局，村庄阵营取得关键进展。"
            elif role == "seer":
                impact = -1.0
                summary = f"预言家 {victim} 出局，村庄失去持续查验来源。"
            elif role in {"witch", "hunter"}:
                impact = -0.7
                summary = f"关键神职 {victim}（{role}）出局，村庄行动空间收窄。"
            elif team == "village":
                impact = -0.4
                summary = f"好人 {victim} 出局，狼人推进人数优势。"
        elif event.event_type in {"witch_save", "witch_save_passed"}:
            category = "ability"
            target_team = players.get(event.target or "", {}).get("team")
            impact = 0.7 if target_team != "werewolves" else -0.5
            summary = f"女巫救下 {event.target or '目标'}，改变了当夜减员结果。"
        elif event.event_type == "hunter_shot":
            category = "ability"
            target_team = players.get(event.target or "", {}).get("team")
            impact = 0.8 if target_team == "werewolves" else -0.8
            summary = f"猎人对 {event.target or '目标'} 开枪，直接改变阵营人数。"
        elif event.event_type == "game_finished":
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
    sheriff = process_metrics.get("sheriff", {})
    election = sheriff.get("election", {})
    leadership = sheriff.get("leadership", {})
    badge_flow = sheriff.get("badge_flow", {})
    if election.get("held"):
        if election.get("winner_id"):
            role_label = election.get("winner_role") or "身份未知"
            attribution.append(f"{election['winner_id']}（{role_label}）赢得警长竞选。")
        elif election.get("void"):
            attribution.append("警长竞选流局，警徽未能形成持续领导链。")
    if leadership.get("nomination_response_rate") is not None:
        attribution.append(
            f"警长归票响应率为{leadership['nomination_response_rate']:.1%}"
            f"（{leadership['nomination_followed_count']}/{leadership['nomination_response_sample_size']}）。"
        )
    if leadership.get("decisive_weighted_vote_count"):
        attribution.append(f"警长 1.5 票有{leadership['decisive_weighted_vote_count']}次改变最终票型。")
    if leadership.get("order_decision_count"):
        directions = "、".join(
            item.get("direction") or "unknown"
            for item in leadership.get("orders", [])
        )
        attribution.append(
            f"警长共决定{leadership['order_decision_count']}次发言顺序"
            + (f"（{directions}）" if directions else "")
            + "。"
        )
    dispositions = [
        item
        for item in leadership.get("badge_transfer_history", [])
        if item.get("action") in {"transfer", "destroy"}
    ]
    if dispositions:
        chain = "；".join(
            (
                f"{item.get('from_id') or '原警长'}→{item.get('to_id') or '未知玩家'}"
                if item.get("action") == "transfer"
                else f"{item.get('from_id') or '原警长'}撕徽"
            )
            for item in dispositions
        )
        attribution.append(f"警徽处置链：{chain}。")
    if badge_flow.get("execution_rate") is not None:
        attribution.append(
            f"可验证的警徽流执行率为{badge_flow['execution_rate']:.1%}"
            f"（{badge_flow['compliant_count']}/{badge_flow['evaluable_count']}）。"
        )

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
