"""不同 Agent/模型版本的零和 Elo 与角色胜率排行榜。"""

from __future__ import annotations
from collections import defaultdict
import math
from typing import Any, Iterable, Mapping
from .common import _first, _mapping, _text, normalize_faction, _rate, _win_rates


def _participant_key(participant: Mapping[str, Any]) -> str:
    explicit = participant.get("leaderboard_key")
    if explicit is not None:
        return _text(explicit)
    agent = _text(_first(participant, "agent", "controller", "id", "name", default="agent"))
    model = _text(participant.get("model"))
    version = _text(participant.get("version"))
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
                "agent": _text(_first(participant, "agent", "controller", "id", default=key)),
                "model": _text(participant.get("model")) or None,
                "version": _text(participant.get("version")) or None,
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

        source = _mapping(participants) if isinstance(participants, Mapping) else {}
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
                winner = winner or source.get("winner")
                participants = source.get("participants") or source.get("players") or []
                game_id = game_id or source.get("game_id")
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
            raise TypeError("participants must be a list or a project evaluation")

        canonical: list[dict[str, Any]] = []
        seen_in_game: set[str] = set()
        for participant in participant_iterable:
            key = _participant_key(participant)
            # One model/version entry may control several seats.  Each entry is
            # counted once per game; duplicate seats are merged by faction.
            team = normalize_faction(_first(participant, "team", default=participant.get("role")))
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
                    "win_rate": _rate(entry["wins"], entry["games"]),
                    "roles": _win_rates(entry["roles"]),
                    "factions": _win_rates(entry["factions"]),
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


def build_leaderboard(
    games: Iterable[Any], *, initial_rating: float = 1500.0, k_factor: float = 32.0, min_games: int = 0
) -> dict[str, Any]:
    """Aggregate evaluations or participant game records into a leaderboard."""

    board = EloLeaderboard(initial_rating=initial_rating, k_factor=k_factor)
    for game in games:
        board.record_game(game)
    return board.to_dict(min_games=min_games)
