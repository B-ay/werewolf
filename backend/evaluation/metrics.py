"""结果与过程指标，包括警长、警徽流、言票一致性与角色贡献。"""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
from .common import (
    _Event,
    _EventIndex,
    _clamp,
    _rate,
    _event_victim,
    _extract_check_result,
    _extract_vote_intent,
    _first,
    _is_ballot_decision,
    _is_check,
    _is_consistency_speech,
    _is_death,
    _is_exile_vote,
    _is_primary_exile_vote,
    _is_public,
    _is_sheriff_ballot_event,
    _is_speech,
    _mapping,
    _mentions_target,
    _norm,
    _round_metric,
    _text,
)


def _information_leakage(events: _EventIndex) -> dict[str, Any]:
    public_events = events.groups["public"]
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
        authorised = bool(event.data.get("authorized_reveal")) or event.event_type in {"death", "death_detail"}
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


def _speech_vote_consistency(events: _EventIndex) -> dict[str, Any]:
    latest_intent: dict[tuple[int | None, str], tuple[str, int]] = {}
    comparisons: list[dict[str, Any]] = []
    by_player: dict[str, list[bool]] = defaultdict(list)
    for event in events:
        if _is_consistency_speech(event) and event.actor:
            intent = _extract_vote_intent(event)
            if intent:
                latest_intent[(event.round, event.actor)] = (intent, event.index)
        if _is_primary_exile_vote(event) and event.actor and event.target:
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
        "rate": _rate(consistent_count, sample_size),
        "coverage": _round_metric(sample_size / max(1, len(events.groups["primary_exile_vote"]))),
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
    events: _EventIndex, players: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    checks = events.groups["check"]
    details: list[dict[str, Any]] = []
    by_player: dict[str, list[bool]] = defaultdict(list)
    for check in checks:
        result = _extract_check_result(check, players)
        used_by: str | None = None
        use_type: str | None = None
        for later in events.by_actor.get(check.actor or "", []):
            if later.index <= check.index:
                continue
            if _is_public(later) and _is_speech(later) and _mentions_target(later, check.target or ""):
                used_by, use_type = later.event_type, "public_claim"
                break
            if result == "werewolves" and _is_exile_vote(later) and later.target == check.target:
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
        "rate": _rate(used_count, len(details)),
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


def _coordination_metric(events: Sequence[_Event]) -> dict[str, Any]:
    grouped: dict[tuple[int | None, str], list[_Event]] = defaultdict(list)
    for event in events:
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
        "rounds_evaluated": len(rounds),
        "mean_agreement": _round_metric(mean_agreement),
        "unanimous_rounds": sum(item["unanimous"] for item in rounds),
        "rounds": rounds,
    }


def _wolf_coordination(
    events: _EventIndex, players: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Measure each wolf coordination channel without mixing ballot types."""

    wolf_ids = {player_id for player_id, player in players.items() if player.get("team") == "werewolves"}
    night = _coordination_metric(
        [event for event in events.groups["wolf_vote"] if event.actor in wolf_ids]
    )
    exile = _coordination_metric(
        [event for event in events.groups["exile_vote"] if event.actor in wolf_ids]
    )
    sheriff = _coordination_metric(
        [event for event in events.groups["sheriff_ballot"] if event.actor in wolf_ids]
    )
    return {
        "wolf_count": len(wolf_ids),
        # Preserve the historical field as the private night-channel metric.
        "rounds_evaluated": night["rounds_evaluated"],
        "mean_agreement": night["mean_agreement"],
        "unanimous_rounds": night["unanimous_rounds"],
        "rounds": night["rounds"],
        "night_attack": night,
        "exile_vote": exile,
        "sheriff_election": sheriff,
    }


def _id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, (list, tuple)):
        raise TypeError("player references must be a list of ids")
    return list(dict.fromkeys(_text(item) for item in value if _text(item)))

def _event_ballot_choices(event: _Event) -> dict[str, str | None]:
    """统一读取封存单票和已公开批次；警长弃票必须保留为 None。"""
    if _is_sheriff_ballot_event(event):
        return {event.actor or "": None if event.data.get("abstained") or not event.target else event.target}
    if _is_ballot_decision(event):
        return {event.actor or "": event.target}
    votes = event.data.get("votes", {})
    if not isinstance(votes, Mapping):
        raise TypeError("votes must map voter ids to target ids or None")
    return {_text(voter): _text(target) or None for voter, target in votes.items()}

def _event_ballots(event: _Event) -> dict[str, str]:
    """Read valid individual choices from a ballot event or revealed batch."""

    return {
        voter: target
        for voter, target in _event_ballot_choices(event).items()
        if target is not None
    }


def _numeric_tally(data: Mapping[str, Any], *keys: str) -> dict[str, float]:
    for key in keys:
        value = data.get(key)
        if not isinstance(value, Mapping):
            continue
        result: dict[str, float] = {}
        for candidate_value, score_value in value.items():
            try:
                result[_text(candidate_value)] = float(score_value)
            except (TypeError, ValueError):
                continue
        if result:
            return result
    return {}


def _unique_tally_winner(tally: Mapping[str, float]) -> str | None:
    if not tally:
        return None
    top = max(tally.values())
    leaders = [candidate for candidate, score in tally.items() if score == top]
    return leaders[0] if len(leaders) == 1 else None


def _exile_vote_accuracy(
    events: _EventIndex, players: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Score only daytime exile/PK ballots, excluding wolf and sheriff votes."""

    ballots: dict[tuple[int | None, str, str], str] = {}
    for event in events:
        if _is_exile_vote(event) and event.actor and event.target:
            ballots[(event.round, event.phase, event.actor)] = event.target
        elif event.event_type == "day_vote_result":
            phase = "day_pk_vote" if bool(event.data.get("is_pk")) or event.phase == "day_pk_vote" else "day_vote"
            for voter, target in _event_ballots(event).items():
                ballots[(event.round, phase, voter)] = target

    by_player_values: dict[str, list[bool | None]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    for (round_no, phase, voter), target in sorted(
        ballots.items(), key=lambda item: ((item[0][0] or -1), item[0][1], item[0][2])
    ):
        voter_team = players.get(voter, {}).get("team")
        target_team = players.get(target, {}).get("team")
        correct = (
            voter_team != target_team
            if voter_team is not None and target_team is not None
            else None
        )
        by_player_values[voter].append(correct)
        details.append(
            {
                "round": round_no,
                "phase": phase,
                "voter_id": voter,
                "target_id": target,
                "correct": correct,
            }
        )
    evaluable = [item for item in details if item["correct"] is not None]
    correct_count = sum(item["correct"] is True for item in evaluable)
    return {
        "ballot_count": len(details),
        "sample_size": len(evaluable),
        "correct_count": correct_count,
        "rate": _rate(correct_count, len(evaluable)),
        "by_player": {
            player_id: {
                "ballot_count": len(values),
                "sample_size": len(known := [value for value in values if value is not None]),
                "correct_count": sum(value is True for value in known),
                "rate": _round_metric(sum(value is True for value in known) / len(known))
                if known
                else None,
            }
            for player_id, values in sorted(by_player_values.items())
        },
        "ballots": details,
    }


def _badge_flow_from_event(event: _Event) -> dict[str, Any] | None:
    flow = _mapping(event.data.get("badge_flow"))
    mode = _norm(flow.get("mode"))
    targets = _id_list(flow.get("targets"))[:2]
    if mode not in {"single", "double"} or not targets:
        return None
    return {"mode": mode, "targets": targets[:1] if mode == "single" else targets, "fallback_target": _text(flow.get("fallback_target")) or None}

def _sheriff_election(events: _EventIndex, players: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """竞选名单、封存选票、弃权和两轮票数只在此汇总。"""
    candidates: list[str] = []
    campaign_speech_order: list[str] = []
    withdrawn: list[str] = []
    pk_candidates: list[str] = []
    election_casts: dict[tuple[int, str], str | None] = {}
    election_ballots: dict[tuple[int, str], str] = {}
    election_tallies: dict[int, Counter[str]] = {}
    election_rounds: set[int] = set()
    winner = void_reason = None
    for event in events:
        data = event.data
        if event.event_type == "sheriff_candidacy_result":
            candidates.extend(player_id for player_id in _id_list(data.get("candidates")) if player_id not in candidates)
            campaign_speech_order = _id_list(data.get("speech_order"))
        elif event.event_type == "sheriff_speech" and event.phase == "sheriff_speech" and event.actor and event.actor not in candidates:
            candidates.append(event.actor)
        elif event.event_type == "sheriff_withdrawal_result":
            withdrawn.extend(player_id for player_id in _id_list(data.get("withdrawn")) if player_id not in withdrawn)
        elif event.event_type == "sheriff_pk_started":
            pk_candidates = _id_list(data.get("candidates"))
        elif _is_sheriff_ballot_event(event) or event.event_type == "sheriff_vote_result":
            vote_round = int(data.get("round", 2 if event.phase == "sheriff_pk_vote" else 1))
            election_rounds.add(vote_round)
            for voter_id, target_id in _event_ballot_choices(event).items():
                election_casts[(vote_round, voter_id)] = target_id
                if target_id is not None:
                    election_ballots[(vote_round, voter_id)] = target_id
            tally = _numeric_tally(data, "tally") if event.event_type == "sheriff_vote_result" else {}
            if tally:
                election_tallies[vote_round] = Counter({candidate: int(score) for candidate, score in tally.items()})
        elif event.event_type == "sheriff_elected":
            winner = _text(data.get("holder_id") or event.target) or None
        elif event.event_type == "sheriff_election_void":
            void_reason = _text(data.get("reason") or event.message) or "void"
    valid_ballots = list(election_ballots.values())
    if election_tallies:
        votes_received: Counter[str] = Counter()
        for tally in election_tallies.values():
            votes_received.update(tally)
        valid_ballot_count = sum(sum(tally.values()) for tally in election_tallies.values())
    else:
        votes_received = Counter(valid_ballots)
        valid_ballot_count = len(election_ballots)
    cast_ballot_count = len(election_casts)
    abstention_count = sum(target is None for target in election_casts.values())

    winner_role = players.get(winner or "", {}).get("role")
    election = {
        "held": any(event.event_type == "sheriff_election_started" for event in events),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "speech_order": campaign_speech_order,
        "withdrawn": withdrawn,
        "pk_occurred": bool(pk_candidates),
        "pk_candidates": pk_candidates,
        "vote_rounds": len(election_rounds),
        # ``ballot_count`` remains the 1.1-compatible valid-ballot count.
        "ballot_count": valid_ballot_count,
        "valid_ballot_count": valid_ballot_count,
        "cast_ballot_count": cast_ballot_count,
        "abstention_count": abstention_count,
        "winner_id": winner,
        "winner_role": winner_role,
        "seer_elected": winner_role == "seer",
        "wolf_bluffer_elected": players.get(winner or "", {}).get("team") == "werewolves",
        "void": winner is None and void_reason is not None,
        "void_reason": void_reason,
        "votes_received": dict(sorted(votes_received.items())),
    }

    return election


def _sheriff_leadership(events: _EventIndex) -> dict[str, Any]:
    """追踪持徽、发言顺序、归票及加权票的影响，保持事件时序。"""
    orders: list[dict[str, Any]] = []
    nominations: list[dict[str, Any]] = []
    badge_history: list[dict[str, Any]] = []
    sheriff_holders: list[str] = []
    decisive_votes: list[dict[str, Any]] = []
    current_holder = None
    for event in events:
        data, kind = event.data, event.event_type
        if kind in {"sheriff_elected", "badge_transferred", "badge_destroyed"}:
            source = event.actor if kind != "sheriff_elected" else None
            target = (_text(data.get("holder_id") or event.target) or None) if kind == "sheriff_elected" else event.target if kind == "badge_transferred" else None
            current_holder = target
            if target:
                sheriff_holders.append(target)
            badge_history.append({
                "event_index": event.index, "round": event.round,
                "action": {"sheriff_elected": "elected", "badge_transferred": "transfer", "badge_destroyed": "destroy"}[kind],
                "from_id": source, "to_id": target,
            })
        elif kind == "sheriff_order":
            orders.append({
                "event_index": event.index, "round": event.round,
                "sheriff_id": event.actor or None,
                "direction": _norm(data.get("direction") or event.target),
                "speech_order": _id_list(data.get("order")),
            })
        elif kind == "sheriff_summary":
            nominations.append({
                "event_index": event.index, "round": event.round,
                "sheriff_id": event.actor,
                "target_id": _text(_first(data, "sheriff_nomination", "vote_intent", default=event.target)) or None,
            })
        elif kind == "day_vote_result":
            weighted = _numeric_tally(data, "tally_units", "tally")
            base = _numeric_tally(data, "unweighted_tally_units")
            if not base:
                base = dict(Counter(_event_ballots(event).values()))
            explicit = data.get("sheriff_vote_decisive")
            decisive = bool(explicit) if explicit is not None else bool(weighted and base and _unique_tally_winner(weighted) != _unique_tally_winner(base))
            if decisive:
                decisive_votes.append({
                    "event_index": event.index, "round": event.round,
                    "sheriff_id": _text(data.get("sheriff_id") or current_holder) or None,
                    "weighted_winner": _unique_tally_winner(weighted) or _text(data.get("eliminated") or event.target) or None,
                    "unweighted_winner": _unique_tally_winner(base),
                })
    # Compare the final public nomination of each day to that day's first
    # exile ballot. Earlier discussion-cycle summaries and the constrained PK
    # ballot are separate decisions and must not duplicate/pollute the sample.
    final_nomination_index_by_round: dict[int | None, int] = {}
    for nomination in nominations:
        final_nomination_index_by_round[nomination["round"]] = nomination["event_index"]
    nomination_details: list[dict[str, Any]] = []
    for nomination in nominations:
        target = nomination["target_id"]
        ballots: dict[str, str] = {}
        is_final_for_round = (
            final_nomination_index_by_round.get(nomination["round"])
            == nomination["event_index"]
        )
        if target and is_final_for_round:
            round_events = events if nomination["round"] is None else sorted(
                events.by_round.get(nomination["round"], []) + events.by_round.get(None, []),
                key=lambda event: event.index,
            )
            for later in round_events:
                if later.index <= nomination["event_index"]:
                    continue
                if _is_primary_exile_vote(later) and later.actor and later.target:
                    ballots[later.actor] = later.target
                elif later.event_type == "day_vote_result" and later.phase not in {
                    "day_pk_vote",
                    "day_exile_pk_vote",
                }:
                    ballots.update(_event_ballots(later))
                    # The nomination belongs to the normal ballot.  A runoff
                    # is a new constrained decision and must not overwrite
                    # the first-round response sample.
                    break
        followed = sum(choice == target for choice in ballots.values())
        nomination_details.append(
            {
                **nomination,
                "evaluated": is_final_for_round,
                "ballot_count": len(ballots),
                "followed_count": followed,
                "response_rate": _rate(followed, len(ballots)),
            }
        )
    response_samples = sum(item["ballot_count"] for item in nomination_details)
    response_followed = sum(item["followed_count"] for item in nomination_details)


    leadership = {
        "holder_history": sheriff_holders,
        "order_decision_count": len(orders),
        "orders": orders,
        "nomination_count": len(nomination_details),
        "evaluated_nomination_count": sum(item["evaluated"] for item in nomination_details),
        "nomination_response_sample_size": response_samples,
        "nomination_followed_count": response_followed,
        "nomination_response_rate": _round_metric(response_followed / response_samples)
        if response_samples
        else None,
        "nominations": nomination_details,
        "decisive_weighted_vote_count": len(decisive_votes),
        "decisive_weighted_votes": decisive_votes,
        "badge_transfer_history": badge_history,
    }

    return leadership


def _badge_flow_metrics(events: _EventIndex, players: Mapping[str, Mapping[str, Any]], badge_history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """声明仅是战术承诺；真预言家可以评测，悍跳狼只记录公开声明。"""
    declarations: list[dict[str, Any]] = []
    declaration_signatures = set()
    for event in events.groups["speech"]:
        flow = _badge_flow_from_event(event)
        if flow and event.actor:
            signature = (event.actor, flow["mode"], tuple(flow["targets"]), flow.get("fallback_target"))
            if signature not in declaration_signatures:
                declaration_signatures.add(signature)
                declarations.append({"event_index": event.index, "round": event.round, "player_id": event.actor, **flow})
    # Evaluate truthful seer flows when their badge is later disposed.  Wolf
    # bluff declarations remain visible in the report but have no truth score.
    death_index: dict[str, int] = {}
    for event in events.groups["death"]:
        victim = _event_victim(event) if _is_death(event) else None
        if victim:
            death_index.setdefault(victim, event.index)
    dispositions = [item for item in badge_history if item["action"] in {"transfer", "destroy"}]
    flow_details: list[dict[str, Any]] = []
    for declaration in declarations:
        player_id = declaration["player_id"]
        role = _norm(players.get(player_id, {}).get("role"))
        is_seer = role == "seer"
        disposition = next(
            (
                item
                for item in dispositions
                if item["event_index"] > declaration["event_index"] and item.get("from_id") == player_id
            ),
            None,
        )
        expected_action: str | None = None
        expected_target: str | None = None
        expected_reason: str | None = None
        if is_seer and disposition is not None:
            results: list[str | None] = []
            for target in declaration["targets"]:
                check = next(
                    (
                        event
                        for event in events.groups["check"]
                        if declaration["event_index"] < event.index < disposition["event_index"]
                        and _is_check(event)
                        and event.actor == player_id
                        and event.target == target
                    ),
                    None,
                )
                # A first-night result can precede the public declaration.
                if check is None:
                    check = next(
                        (
                            event
                            for event in reversed(events.by_actor.get(player_id, []))
                            if event.index < disposition["event_index"] and _is_check(event) and event.target == target
                        ),
                        None,
                    )
                results.append(_extract_check_result(check, players) if check else None)

            if declaration["mode"] == "single" or len(declaration["targets"]) == 1 or results[1] is None:
                if results[0] == "village":
                    expected_action, expected_target = "transfer", declaration["targets"][0]
                    expected_reason = "single_good"
                else:
                    expected_action = "destroy"
                    expected_reason = "single_wolf_or_unchecked"
            elif results[0] == "village" and results[1] == "werewolves":
                expected_action, expected_target = "transfer", declaration["targets"][0]
                expected_reason = "double_one_good_one_wolf"
            elif results[0] == "werewolves" and results[1] == "village":
                expected_action, expected_target = "transfer", declaration["targets"][1]
                expected_reason = "double_one_wolf_one_good"
            elif results[0] == "village" and results[1] == "village":
                expected_action = "destroy"
                expected_reason = "double_both_good"
            elif results[0] == "werewolves" and results[1] == "werewolves":
                expected_target = declaration.get("fallback_target")
                if expected_target:
                    expected_action = "transfer"
                    expected_reason = "double_both_wolves_fallback"
                else:
                    expected_action = "destroy"
                    expected_reason = "double_both_wolves_without_fallback"
            else:
                expected_action = "destroy"
                expected_reason = "double_incomplete"

            if expected_target and death_index.get(expected_target, 10**9) <= disposition["event_index"]:
                expected_action, expected_target = "destroy", None
                expected_reason = "intended_receiver_dead"

        compliant: bool | None = None
        if expected_action and disposition:
            compliant = disposition["action"] == expected_action and (
                expected_action == "destroy" or disposition.get("to_id") == expected_target
            )
        flow_details.append(
            {
                **declaration,
                "declarer_role": players.get(player_id, {}).get("role"),
                "truth_evaluable": is_seer and disposition is not None,
                "expected_action": expected_action,
                "expected_target": expected_target,
                "expected_reason": expected_reason,
                "actual_action": disposition.get("action") if disposition else None,
                "actual_target": disposition.get("to_id") if disposition else None,
                "compliant": compliant,
            }
        )

    evaluable_flows = [item for item in flow_details if item["compliant"] is not None]
    compliant_flows = sum(item["compliant"] is True for item in evaluable_flows)

    badge_flow = {
        "declaration_count": len(flow_details),
        "evaluable_count": len(evaluable_flows),
        "compliant_count": compliant_flows,
        "execution_rate": _rate(compliant_flows, len(evaluable_flows)),
        "declarations": flow_details,
    }

    return badge_flow


def _sheriff_metrics(events: _EventIndex, players: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """组合三个独立步骤，再生成玩家维度补充，不改既有 Schema 1.1 字段。"""
    election = _sheriff_election(events, players)
    leadership = _sheriff_leadership(events)
    badge_flow = _badge_flow_metrics(events, players, leadership["badge_transfer_history"])
    candidates, withdrawn, winner = election["candidates"], election["withdrawn"], election["winner_id"]
    votes_received = election["votes_received"]
    sheriff_holders, orders = leadership["holder_history"], leadership["orders"]
    nomination_details, decisive_votes = leadership["nominations"], leadership["decisive_weighted_votes"]
    badge_history, flow_details = leadership["badge_transfer_history"], badge_flow["declarations"]
    evaluable_flows = [item for item in flow_details if item["compliant"] is not None]
    player_supplement: dict[str, dict[str, Any]] = {}
    for player_id in players:
        own_nominations = [item for item in nomination_details if item["sheriff_id"] == player_id]
        own_samples = sum(item["ballot_count"] for item in own_nominations)
        own_followed = sum(item["followed_count"] for item in own_nominations)
        own_flows = [item for item in evaluable_flows if item["player_id"] == player_id]
        player_supplement[player_id] = {
            "ran_for_sheriff": player_id in candidates,
            "withdrew_from_sheriff": player_id in withdrawn,
            "sheriff_election_won": player_id == winner,
            "sheriff_votes_received": votes_received.get(player_id, 0),
            "ever_sheriff": player_id in sheriff_holders,
            "sheriff_order_count": sum(item["sheriff_id"] == player_id for item in orders),
            "sheriff_nomination_count": len(own_nominations),
            "sheriff_nomination_response_rate": _rate(own_followed, own_samples),
            "sheriff_decisive_vote_count": sum(item["sheriff_id"] == player_id for item in decisive_votes),
            "badge_transfers_given": sum(
                item["action"] == "transfer" and item["from_id"] == player_id for item in badge_history
            ),
            "badge_transfers_received": sum(
                item["action"] == "transfer" and item["to_id"] == player_id for item in badge_history
            ),
            "badge_flow_declaration_count": sum(item["player_id"] == player_id for item in flow_details),
            "badge_flow_execution_rate": _round_metric(
                sum(item["compliant"] is True for item in own_flows) / len(own_flows)
            )
            if own_flows
            else None,
        }


    return {"election": election, "leadership": leadership, "badge_flow": badge_flow}, player_supplement


def _role_action_effectiveness(
    player_id: str,
    player: Mapping[str, Any],
    events: _EventIndex,
    players: Mapping[str, Mapping[str, Any]],
    seer_metric: Mapping[str, Any],
) -> tuple[float | None, int, list[str]]:
    role = _norm(player.get("role"))
    team = player.get("team")
    successes = 0.0
    attempts = 0
    notes: list[str] = []
    own_events = events.by_actor.get(player_id, [])

    if role == "seer":
        checks = [event for event in own_events if _is_check(event)]
        utilized = seer_metric.get("by_player", {}).get(player_id, {}).get("utilized_count", 0)
        wolves_found = sum(_extract_check_result(event, players) == "werewolves" for event in checks)
        attempts = len(checks) * 2
        successes = utilized + wolves_found
        if checks:
            notes.append(f"查验{len(checks)}次，发现狼人{wolves_found}次，利用信息{utilized}次")
    elif role == "witch":
        for event in own_events:
            if event.event_type in {"witch_save", "witch_save_passed"}:
                attempts += 1
                target_team = players.get(event.target or "", {}).get("team")
                successes += target_team != "werewolves"
            elif event.event_type in {"witch_poison", "witch_poison_passed"}:
                attempts += 1
                successes += players.get(event.target or "", {}).get("team") == "werewolves"
        if attempts:
            notes.append(f"女巫技能有效{int(successes)}/{attempts}次")
    elif role == "hunter":
        shots = [event for event in own_events if event.event_type == "hunter_shot"]
        attempts = len(shots)
        successes = sum(players.get(event.target or "", {}).get("team") == "werewolves" for event in shots)
        if attempts:
            notes.append(f"猎人命中狼人{int(successes)}/{attempts}次")
    elif team == "werewolves":
        kills = [
            event
            for event in own_events
            if event.event_type == "wolf_kill" and event.target
        ]
        attempts = len(kills)
        successes = sum(players.get(event.target or "", {}).get("team") == "village" for event in kills)
        if attempts:
            notes.append(f"狼人行动命中好人{int(successes)}/{attempts}次")
    return (successes / attempts if attempts else None), attempts, notes


def _player_metrics(
    events: _EventIndex,
    players: Mapping[str, Mapping[str, Any]],
    winner: str | None,
    seer_metric: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    speech_counts = Counter(event.actor for event in events.groups["speech"] if event.actor)
    action_counts = Counter({actor: len(items) for actor, items in events.by_actor.items()})
    max_actions = max(action_counts.values(), default=0)
    votes_by_player: dict[str, list[_Event]] = defaultdict(list)
    for event in events.groups["exile_vote"]:
        votes_by_player[event.actor or ""].append(event)

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
            "exile_vote_count": len(votes),
            "opponent_exile_vote_count": aligned_votes,
            "exile_vote_accuracy": _round_metric(vote_accuracy),
            "ability_attempts": ability_attempts,
            "role_action_effectiveness": _round_metric(action_effectiveness),
            "participation": _round_metric(participation),
            "win_contribution": 1.0 if won else (0.0 if winner is not None else None),
            "survival_contribution": 1.0 if alive is True else (0.0 if alive is False else None),
            "contribution_score": _round_metric(_clamp(contribution)),
            "notes": notes,
        }
    return result
