"""狼人杀多智能体系统的跨模块数据契约。

本模块故意不依赖 HTTP 框架、具体 Agent 或大模型 SDK。引擎、Agent、服务端、
评测器都只通过这里定义的对象交换数据，因此阅读代码时可以把本文件理解成整个
项目的“协议层”：

* :class:`Observation` 是裁判允许某个 Agent 看到的全部信息，Agent 不应直接接触
  ``GameState`` 或其他玩家的私有状态；
* :class:`LegalAction` 描述当前时点允许做什么，:class:`Action` 则是 Agent 最终
  提交的意图，真正执行前仍会由引擎二次校验；
* :class:`GameEvent` 是只追加的事实日志，同时承担前端展示、记忆写入、评测和复盘
  的共同数据源；
* 所有 ``to_dict`` 都返回可安全序列化的副本，避免调用方通过嵌套列表或字典反向
  修改裁判状态。
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
    """完整对局状态机的公开阶段枚举。

    枚举值足够细，便于引擎选择当前行动者与合法动作；``public_name`` 则把多个
    私有夜间阶段或警长子阶段折叠成适合普通观战者展示的粗粒度名称。
    """
    NIGHT_WOLF = "night_wolf"
    NIGHT_SEER = "night_seer"
    NIGHT_WITCH_SAVE = "night_witch_save"
    NIGHT_WITCH_POISON = "night_witch_poison"
    NIGHT_RESOLVE = "night_resolve"
    SHERIFF_RUN = "sheriff_run"
    SHERIFF_SPEECH = "sheriff_speech"
    SHERIFF_WITHDRAW = "sheriff_withdraw"
    SHERIFF_VOTE = "sheriff_vote"
    SHERIFF_PK_SPEECH = "sheriff_pk_speech"
    SHERIFF_PK_VOTE = "sheriff_pk_vote"
    DAWN_REVEAL = "dawn_reveal"
    SHERIFF_BADGE = "sheriff_badge"
    SHERIFF_ORDER = "sheriff_order"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTE = "day_vote"
    DAY_PK_SPEECH = "day_pk_speech"
    DAY_PK_VOTE = "day_pk_vote"
    HUNTER_SHOT = "hunter_shot"
    FINISHED = "finished"

    @property
    def public_name(self) -> str:
        if self.value.startswith("night_"):
            return "night"
        if self in (
            Phase.SHERIFF_RUN,
            Phase.SHERIFF_SPEECH,
            Phase.SHERIFF_WITHDRAW,
            Phase.SHERIFF_VOTE,
            Phase.SHERIFF_PK_SPEECH,
            Phase.SHERIFF_PK_VOTE,
        ):
            return "sheriff_election"
        if self in (
            Phase.SHERIFF_ORDER,
            Phase.DAY_DISCUSSION,
            Phase.DAY_VOTE,
            Phase.DAY_PK_SPEECH,
            Phase.DAY_PK_VOTE,
        ):
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
    SHERIFF_RUN = "sheriff_run"
    SHERIFF_WITHDRAW = "sheriff_withdraw"
    SHERIFF_VOTE = "sheriff_vote"
    SHERIFF_ORDER_CLOCKWISE = "sheriff_order_clockwise"
    SHERIFF_ORDER_COUNTERCLOCKWISE = "sheriff_order_counterclockwise"
    SHERIFF_SUMMARIZE = "sheriff_summarize"
    SHERIFF_TRANSFER = "sheriff_transfer"
    SHERIFF_DESTROY = "sheriff_destroy"
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
    """引擎为当前行动者生成的动作白名单。

    ``target_ids`` 已经完成存活、身份和阶段过滤。规划器以及 DeepSeek 只能在这个
    集合内选择目标；``can_pass`` 和 ``content_required`` 分别约束能否弃权、是否
    必须提供发言文本。该对象是“建议空间”，最终 :class:`Action` 仍由引擎校验。
    """

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
    """暂停等待人类输入时返回给前端的最小安全快照。

    它只包含当前玩家、公开阶段、提示和合法动作，不携带其他角色的私有事实。
    AI 玩家不会长期保存该对象，而是在 ``step`` 中立即完成一次决策。
    """

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
    """一次 Agent 决策可使用的完整且唯一的信息输入。

    ``self_state`` 保存本人的公开/私有状态，``players`` 是按观察者权限脱敏后的
    座位列表，``events`` 只包含该观察者可见的事件，``private_facts`` 则承载查验
    结果、狼队成员等合法私有事实。长期记忆也只能从这份对象吸收信息，不能回读
    引擎内部状态；这一约束是防止信息泄露的核心边界。
    """

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
    """直接传给 ``WerewolfGame`` 的对局配置。

    ``__post_init__`` 会完成默认牌组生成、角色数量校验、人类座位合并和规则参数
    校验，使引擎启动后不必反复处理不完整配置。``seed`` 只用于可复现的裁判随机
    流；各 Agent 仍持有相互独立的随机流。

    ``human_player`` 是单个人类座位的便捷别名，最终会合并进 ``human_players``。
    座位引用可以是玩家 ID（如 ``p1``）、玩家名或从 1 开始的座位号。
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
    enable_sheriff: bool = True
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
            "enable_sheriff": self.enable_sheriff,
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
