"""Core data models for the multi-agent Werewolf game.

The module deliberately contains no framework dependencies.  Every public model
can be converted to a JSON-friendly dictionary, which keeps the engine usable
from a CLI, a web API, or a tournament runner.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence


class Role(str, Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    HUNTER = "hunter"
    VILLAGER = "villager"

    @classmethod
    def coerce(cls, value: "Role | str") -> "Role":
        if isinstance(value, cls):
            return value
        aliases = {
            "wolf": cls.WEREWOLF,
            "狼人": cls.WEREWOLF,
            "预言家": cls.SEER,
            "女巫": cls.WITCH,
            "猎人": cls.HUNTER,
            "村民": cls.VILLAGER,
            "平民": cls.VILLAGER,
        }
        normalized = str(value).strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)

    @property
    def team(self) -> "Team":
        return Team.WEREWOLVES if self is Role.WEREWOLF else Team.VILLAGE


class Team(str, Enum):
    WEREWOLVES = "werewolves"
    VILLAGE = "village"
    DRAW = "draw"


class GameStatus(str, Enum):
    RUNNING = "running"
    WAITING_FOR_HUMAN = "waiting_for_human"
    FINISHED = "finished"


class Phase(str, Enum):
    NIGHT_WOLF = "night_wolf"
    NIGHT_SEER = "night_seer"
    NIGHT_WITCH_SAVE = "night_witch_save"
    NIGHT_WITCH_POISON = "night_witch_poison"
    NIGHT_RESOLVE = "night_resolve"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTE = "day_vote"
    HUNTER_SHOT = "hunter_shot"
    FINISHED = "finished"

    @property
    def public_name(self) -> str:
        if self.value.startswith("night_"):
            return "night"
        if self in (Phase.DAY_DISCUSSION, Phase.DAY_VOTE):
            return "day"
        return self.value


class ActionType(str, Enum):
    WOLF_KILL = "wolf_kill"
    SEER_CHECK = "seer_check"
    WITCH_SAVE = "witch_save"
    WITCH_POISON = "witch_poison"
    SPEAK = "speak"
    VOTE = "vote"
    HUNTER_SHOOT = "hunter_shoot"
    PASS = "pass"

    @classmethod
    def coerce(cls, value: "ActionType | str") -> "ActionType":
        if isinstance(value, cls):
            return value
        aliases = {
            "kill": cls.WOLF_KILL,
            "check": cls.SEER_CHECK,
            "save": cls.WITCH_SAVE,
            "poison": cls.WITCH_POISON,
            "shoot": cls.HUNTER_SHOOT,
            "skip": cls.PASS,
        }
        normalized = str(value).strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


@dataclass(slots=True)
class Action:
    """An action proposed by either an autonomous agent or a human player."""

    actor_id: str
    action_type: ActionType
    target_id: Optional[str] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, default_actor: Optional[str] = None
    ) -> "Action":
        action_name = value.get("action_type", value.get("type"))
        if action_name is None:
            raise ValueError("action_type is required")
        actor_id = value.get("actor_id", value.get("actor", default_actor))
        if actor_id is None:
            raise ValueError("actor_id is required")
        metadata = value.get("metadata", value.get("data", {})) or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            actor_id=str(actor_id),
            action_type=ActionType.coerce(action_name),
            target_id=(
                str(value.get("target_id", value.get("target")))
                if value.get("target_id", value.get("target")) is not None
                else None
            ),
            content=(str(value["content"]) if value.get("content") is not None else None),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "action_type": self.action_type.value,
            "target_id": self.target_id,
            "content": self.content,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(slots=True)
class PlayerState:
    id: str
    name: str
    role: Role
    alive: bool = True
    controller: str = "ai"
    death_reason: Optional[str] = None
    death_round: Optional[int] = None

    @property
    def team(self) -> Team:
        return self.role.team

    def to_private_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "team": self.team.value,
            "alive": self.alive,
            "controller": self.controller,
            "death_reason": self.death_reason,
            "death_round": self.death_round,
        }


@dataclass(slots=True)
class GameEvent:
    """Append-only structured event.

    ``visibility`` contains ``"public"`` for public events, concrete player ids
    for private events, or ``"observer"`` for referee-only events.  The engine
    never sends referee-only events to an agent observation.
    """

    seq: int
    round: int
    phase: str
    event_type: str
    message: str
    actor_id: Optional[str] = None
    target_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)
    visibility: tuple[str, ...] = ("public",)

    def visible_to(self, viewer_id: Optional[str]) -> bool:
        return "public" in self.visibility or (
            viewer_id is not None and viewer_id in self.visibility
        )

    def to_dict(self, *, include_visibility: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "seq": self.seq,
            "round": self.round,
            "phase": self.phase,
            "event_type": self.event_type,
            "actor_id": self.actor_id,
            "target_id": self.target_id,
            "message": self.message,
            # Event payloads may contain nested dictionaries/lists (for
            # example role_counts and vote tallies).  A shallow copy would
            # let an Agent mutate the referee's append-only source of truth
            # through an otherwise legitimate Observation.
            "data": deepcopy(self.data),
        }
        if include_visibility:
            result["visibility"] = list(self.visibility)
        return result


@dataclass(slots=True)
class LegalAction:
    action_type: ActionType
    target_ids: tuple[str, ...] = ()
    can_pass: bool = False
    content_required: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target_ids": list(self.target_ids),
            "can_pass": self.can_pass,
            "content_required": self.content_required,
            "description": self.description,
        }


@dataclass(slots=True)
class PendingAction:
    actor_id: str
    phase: Phase
    legal_actions: tuple[LegalAction, ...]
    prompt: str
    round: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "phase": self.phase.value,
            "round": self.round,
            "prompt": self.prompt,
            "legal_actions": [item.to_dict() for item in self.legal_actions],
        }


@dataclass(slots=True)
class Observation:
    """The complete and only information passed to an agent."""

    viewer_id: str
    round: int
    phase: Phase
    self_state: dict[str, Any]
    players: list[dict[str, Any]]
    events: list[dict[str, Any]]
    legal_actions: list[dict[str, Any]]
    private_facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Return a detached serialization.  Besides protecting the engine,
        # this lets provider adapters normalize or annotate their payload
        # without mutating the Observation seen by another policy layer.
        return deepcopy({
            "viewer_id": self.viewer_id,
            "round": self.round,
            "phase": self.phase.value,
            "self": self.self_state,
            "players": self.players,
            "events": self.events,
            "legal_actions": self.legal_actions,
            "private_facts": self.private_facts,
        })


def default_role_deck(player_count: int) -> list[Role]:
    """Return a reasonably balanced deck for five or more players."""

    if player_count < 5:
        raise ValueError("Werewolf requires at least 5 players")
    wolf_count = max(1, player_count // 3)
    # Leave room for the three power roles and at least one villager.
    wolf_count = min(wolf_count, player_count - 4)
    roles = [Role.WEREWOLF] * wolf_count
    roles.extend((Role.SEER, Role.WITCH, Role.HUNTER))
    roles.extend([Role.VILLAGER] * (player_count - len(roles)))
    return roles


@dataclass(slots=True)
class GameConfig:
    """Configuration accepted directly by :class:`WerewolfGame`.

    ``human_player`` is a convenience alias for a single entry in
    ``human_players``. References may be a player id (``p1``), a player name,
    or a one-based numeric seat.
    """

    player_count: int = 8
    player_names: list[str] = field(default_factory=list)
    roles: Optional[list[Role | str]] = None
    seed: int = 42
    human_player: Optional[str | int] = None
    human_players: list[str | int] = field(default_factory=list)
    reveal_roles_on_death: bool = True
    reveal_vote_choices: bool = True
    shuffle_roles: bool = True
    vote_tie_policy: str = "no_elimination"
    witch_one_potion_per_night: bool = True
    allow_witch_self_save: bool = True
    hunter_can_shoot_if_poisoned: bool = False
    discussion_rounds: int = 1
    max_rounds: int = 20

    def __post_init__(self) -> None:
        if self.player_names:
            self.player_names = [str(name) for name in self.player_names]
            self.player_count = len(self.player_names)
        else:
            self.player_names = [f"Player {i}" for i in range(1, self.player_count + 1)]

        if len(set(self.player_names)) != len(self.player_names):
            raise ValueError("player names must be unique")
        if self.player_count < 5:
            raise ValueError("player_count must be at least 5")
        if self.roles is None:
            self.roles = default_role_deck(self.player_count)
        else:
            self.roles = [Role.coerce(role) for role in self.roles]
        if len(self.roles) != self.player_count:
            raise ValueError("roles must contain exactly one role per player")
        if Role.WEREWOLF not in self.roles:
            raise ValueError("roles must contain at least one werewolf")
        if all(role is Role.WEREWOLF for role in self.roles):
            raise ValueError("roles must contain at least one village-aligned role")
        if self.vote_tie_policy not in {"no_elimination", "random"}:
            raise ValueError("vote_tie_policy must be 'no_elimination' or 'random'")
        if self.discussion_rounds < 1:
            raise ValueError("discussion_rounds must be positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")

        combined: list[str | int] = list(self.human_players or [])
        if self.human_player is not None and self.human_player not in combined:
            combined.append(self.human_player)
        self.human_players = combined

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GameConfig":
        known = {
            name
            for name in cls.__dataclass_fields__  # type: ignore[attr-defined]
            if not name.startswith("_")
        }
        return cls(**{key: val for key, val in value.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_count": self.player_count,
            "player_names": list(self.player_names),
            "roles": [Role.coerce(role).value for role in self.roles or ()],
            "seed": self.seed,
            "human_player": self.human_player,
            "human_players": list(self.human_players),
            "reveal_roles_on_death": self.reveal_roles_on_death,
            "reveal_vote_choices": self.reveal_vote_choices,
            "shuffle_roles": self.shuffle_roles,
            "vote_tie_policy": self.vote_tie_policy,
            "witch_one_potion_per_night": self.witch_one_potion_per_night,
            "allow_witch_self_save": self.allow_witch_self_save,
            "hunter_can_shoot_if_poisoned": self.hunter_can_shoot_if_poisoned,
            "discussion_rounds": self.discussion_rounds,
            "max_rounds": self.max_rounds,
        }


def ensure_action(
    value: Action | Mapping[str, Any], *, default_actor: Optional[str] = None
) -> Action:
    if isinstance(value, Action):
        return value
    return Action.from_dict(value, default_actor=default_actor)


def player_ids(players: Iterable[PlayerState]) -> tuple[str, ...]:
    return tuple(player.id for player in players)
