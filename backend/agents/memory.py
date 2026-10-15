"""独立角色目标与长期记忆；包括去重、窗口裁剪和快照恢复。"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from ..game.models import Action, Observation, Role
from .strategy import RoleStrategy


@dataclass(frozen=True, slots=True)
class RoleGoal:
    """角色 Agent 独占的不可变目标对象。

    ``primary_objective`` 描述最终胜利目标，``objectives`` 给出中间策略方向，
    ``constraints`` 则约束信息来源和角色行为。它会连同角色提示词一起发给模型。
    """

    role: Role
    primary_objective: str
    objectives: tuple[str, ...]
    constraints: tuple[str, ...]

    @property
    def team(self) -> str:
        return self.role.team.value

    @property
    def primary(self) -> str:
        return self.primary_objective

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "team": self.team,
            "primary_objective": self.primary_objective,
            "objectives": list(self.objectives),
            "constraints": list(self.constraints),
        }

    export = to_dict


@dataclass(slots=True)
class RoleMemory:
    """只由合法视角写入、可序列化、可跨对局保留的角色长期记忆。

    一次 ``decide`` 对应一次记忆摄取：当前自身状态、可见玩家和角色私有事实
    采用快照覆盖；公共事件按 ``seq`` 去重后追加；最终动作另存到 ``decisions``。
    记忆与 ``owner_id``、``role`` 强绑定，禁止同一对象被另一个座位或角色复用。

    容量控制采用可解释的分层裁剪，而不是让大模型生成不可审计的摘要：

    * 当前局只保留最近 ``max_events``（默认 256）条事件和最近
      ``max_decisions``（默认 128）个自身决策；超限时从头部裁掉最旧数据。
    * 复盘文本每条最多 1000 字符，只保留最近 64 条。
    * 检测到新对局时把当前局归档到 ``episodes``，最多保留最近
      ``max_episodes``（默认 16）局。
    * 真正送入大模型的 ``prompt_view`` 会进一步“压缩视图”：历史每局只取最近
      12 条可见事件和 8 个自身决策；当前局则使用已经过上述上限裁剪的内容。

    因此这里的“压缩”是确定性的窗口截断，不会臆造总结；被裁掉的旧内容不会
    自动恢复。需要更复杂的语义摘要时，应由外部评测器调用 ``add_reflection``
    写入明确标注的复盘文本，而不能把隐藏事实塞入记忆。
    """

    role: Optional[Role] = None
    owner_id: Optional[str] = None
    viewer_id: Optional[str] = None
    episode: int = 1
    turns: int = 0
    total_turns: int = 0
    last_round: int = 0
    last_event_seq: int = 0
    self_state: dict[str, Any] = field(default_factory=dict)
    visible_players: list[dict[str, Any]] = field(default_factory=list)
    private_facts: dict[str, Any] = field(default_factory=dict)
    strategy_state: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    experience: dict[str, Any] = field(default_factory=dict)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    _seen_event_seqs: set[int] = field(default_factory=set, repr=False)
    max_events: int = 256
    max_decisions: int = 128
    max_episodes: int = 16

    def ingest(self, observation: Observation) -> None:
        """摄取一次经过隔离的观察，并维护归属、新局检测和事件窗口。

        ``self_state``、``players``、``private_facts`` 都用深拷贝保存，避免后续
        引擎对象变化污染历史。事件只有携带整数 ``seq`` 且尚未见过时才会写入，
        从而处理“每次观察都重复携带历史事件”的常见情况。
        """

        observed_role = Role.coerce(observation.self_state["role"])
        observed_owner = observation.viewer_id
        if str(observation.self_state.get("id")) != observed_owner:
            raise ValueError("observation viewer_id does not match self_state.id")
        if self.owner_id is None:
            self.owner_id = observed_owner
        elif self.owner_id != observed_owner:
            raise ValueError(
                f"memory belongs to {self.owner_id}, not {observed_owner}"
            )
        if self.role is None:
            self.role = observed_role
        elif self.role is not observed_role:
            raise ValueError(
                f"memory belongs to role {self.role.value}, not {observed_role.value}"
            )
        # 用“回合倒退”或“事件序号回绕”识别复用同一 Agent 的下一局。
        # 两者都只依赖观察内可见字段，不需要访问全局 game_id。
        visible_seqs = [
            int(event["seq"])
            for event in observation.events
            if isinstance(event.get("seq"), int)
        ]
        max_visible_seq = max(visible_seqs, default=0)
        new_episode = bool(
            (self.last_round and observation.round < self.last_round)
            or (
                self.last_event_seq
                and max_visible_seq
                and max_visible_seq < self.last_event_seq
            )
        )
        if new_episode:
            self.begin_episode(viewer_id=observation.viewer_id, role=observed_role)

        self.viewer_id = observation.viewer_id
        self.turns += 1
        self.total_turns += 1
        self.last_round = observation.round
        self.last_event_seq = max(self.last_event_seq, max_visible_seq)
        # 状态类信息表示“当前认知”，所以覆盖旧快照；事件类信息才采用追加。
        self.self_state = deepcopy(observation.self_state)
        self.visible_players = deepcopy(observation.players)
        self.private_facts = deepcopy(observation.private_facts)
        sheriff_public = observation.private_facts.get("sheriff")
        if isinstance(sheriff_public, Mapping):
            # The engine exposes this exact object to every viewpoint.  Store
            # only that public overlay in strategy_state; role-private facts
            # such as checks, potions, and wolf teammates remain elsewhere.
            self.strategy_state["sheriff_public"] = deepcopy(dict(sheriff_public))

        # observation.events 可能在相邻决策中重复出现，seq 集合负责幂等去重。
        for event in observation.events:
            seq = event.get("seq")
            if not isinstance(seq, int) or seq in self._seen_event_seqs:
                continue
            self._seen_event_seqs.add(seq)
            self.events.append(deepcopy(event))
        # 超上限时只保留尾部窗口。这里没有隐式模型摘要，行为完全确定且可测试。
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def remember_decision(
        self,
        action: Action,
        observation: Observation,
        *,
        source: str,
    ) -> None:
        """记录最终实际执行的动作，并标明它来自模型、规划器或降级路径。"""

        self.decisions.append(
            {
                "episode": self.episode,
                "round": observation.round,
                "phase": observation.phase.value,
                "source": source,
                "action": action.to_dict(),
            }
        )
        # 决策历史与事件历史独立限长，避免长局持续扩大模型上下文和快照体积。
        if len(self.decisions) > self.max_decisions:
            self.decisions = self.decisions[-self.max_decisions :]

    def add_reflection(self, text: str) -> None:
        """保存外部评测器提供的复盘文字，而不是角色不可见的隐藏事实。

        单条先截到 1000 字符，再保留最近 64 条，防止复盘无限膨胀。
        """

        clean = str(text).strip()
        if clean:
            self.reflections.append(clean[:1000])
            self.reflections = self.reflections[-64:]

    def inherit_strategy(self, strategy: RoleStrategy) -> None:
        """新局只继承经验证的通用策略，不复制旧局事件、座位身份或技能结果。"""
        if self.role is not None and strategy.role is not self.role:
            raise ValueError("cannot inherit experience from another role")
        self.experience = strategy.to_dict()
        for lesson in strategy.lessons:
            if lesson not in self.reflections:
                self.add_reflection(lesson)

    def begin_episode(
        self,
        *,
        viewer_id: Optional[str] = None,
        role: Optional[Role | str] = None,
    ) -> None:
        """归档当前局并开始新局，同时保持座位和角色身份不可变。

        归档保留当前局完整的“已裁剪窗口”，之后 ``episodes`` 再按局数裁到最近
        ``max_episodes`` 局。当前局的临时状态会清空，但 ``total_turns`` 保留为
        跨局累计计数。
        """

        if self.turns or self.events or self.decisions:
            self.episodes.append(
                {
                    "episode": self.episode,
                    "owner_id": self.owner_id,
                    "role": self.role.value if self.role else None,
                    "turns": self.turns,
                    "last_round": self.last_round,
                    "events": deepcopy(self.events),
                    "decisions": deepcopy(self.decisions),
                    "reflections": list(self.reflections),
                    "strategy_state": deepcopy(self.strategy_state),
                }
            )
            # 历史局超限时淘汰最旧局；episode 序号继续单调增加。
            self.episodes = self.episodes[-self.max_episodes :]
            self.episode += 1
        requested_owner = viewer_id or self.owner_id
        requested_role = Role.coerce(role) if role is not None else self.role
        if self.owner_id is not None and requested_owner != self.owner_id:
            raise ValueError("cannot change memory owner between episodes")
        if self.role is not None and requested_role is not self.role:
            raise ValueError("cannot change memory role between episodes")
        self.owner_id = requested_owner
        self.viewer_id = requested_owner
        self.role = requested_role
        self.turns = 0
        self.last_round = 0
        self.last_event_seq = 0
        self.self_state = {}
        self.visible_players = []
        self.private_facts = {}
        self.strategy_state = {}
        self.events = []
        self.decisions = []
        self.reflections = []
        self._seen_event_seqs = set()
        if self.experience:
            self.inherit_strategy(RoleStrategy.from_dict(self.experience))

    def prompt_view(self) -> dict[str, Any]:
        """生成发送给 DeepSeek 等可选模型的紧凑记忆视图。

        历史局只暴露每局最近 12 条事件和 8 个决策，这是第二层上下文压缩；
        当前局事件/决策已分别受 256/128 的第一层窗口限制。返回值继续深拷贝，
        模型适配器无法反向修改 Agent 的持久记忆。
        """

        return {
            "episode": self.episode,
            "turns": self.turns,
            "prior_episodes": [
                {
                    "episode": item.get("episode"),
                    "role": item.get("role"),
                    "turns": item.get("turns"),
                    "last_round": item.get("last_round"),
                    "reflections": list(item.get("reflections") or []),
                    "recent_visible_events": deepcopy(
                        list(item.get("events") or [])[-12:]
                    ),
                    "recent_decisions": deepcopy(
                        list(item.get("decisions") or [])[-8:]
                    ),
                }
                for item in self.episodes
            ],
            "visible_events": deepcopy(self.events),
            "own_previous_decisions": deepcopy(self.decisions),
            "reflections": list(self.reflections),
            "strategy_state": deepcopy(self.strategy_state),
            **({"experience": deepcopy(self.experience)} if self.experience else {}),
        }

    def export(self) -> dict[str, Any]:
        """导出带版本号和容量配置的独立 JSON 快照，便于持久化或续局。"""

        return {
            "schema_version": 1,
            "version": 1,
            "role": self.role.value if self.role else None,
            "owner_id": self.owner_id,
            "viewer_id": self.owner_id,
            "episode": self.episode,
            "turns": self.turns,
            "total_turns": self.total_turns,
            "last_round": self.last_round,
            "last_event_seq": self.last_event_seq,
            "self_state": deepcopy(self.self_state),
            "visible_players": deepcopy(self.visible_players),
            "private_facts": deepcopy(self.private_facts),
            "strategy_state": deepcopy(self.strategy_state),
            **({"experience": deepcopy(self.experience)} if self.experience else {}),
            "events": deepcopy(self.events),
            "decisions": deepcopy(self.decisions),
            "reflections": list(self.reflections),
            "episodes": deepcopy(self.episodes),
            "seen_event_seqs": sorted(self._seen_event_seqs),
            "limits": {
                "max_events": self.max_events,
                "max_decisions": self.max_decisions,
                "max_episodes": self.max_episodes,
            },
        }

    snapshot = export

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "RoleMemory":
        """从 ``export`` 结果构造新的记忆对象。"""

        memory = cls()
        memory.restore(snapshot)
        return memory

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """用已导出的快照替换当前内容，并校验版本及座位字段一致性。

        ``strategy_state`` 在 schema v1 中是向后兼容的可选字段；恢复容量配置时
        会把三个上限约束为至少 1，避免产生永远无法保存任何历史的记忆对象。
        """

        if not isinstance(snapshot, Mapping):
            raise TypeError("memory snapshot must be a mapping")
        schema_version = int(snapshot.get("schema_version", snapshot.get("version", 1)))
        if schema_version != 1:
            raise ValueError(f"unsupported memory schema version: {schema_version}")
        role = snapshot.get("role")
        self.role = Role.coerce(role) if role is not None else None
        owner_id = snapshot.get("owner_id", snapshot.get("viewer_id"))
        legacy_viewer = snapshot.get("viewer_id")
        if (
            owner_id is not None
            and legacy_viewer is not None
            and str(owner_id) != str(legacy_viewer)
        ):
            raise ValueError("snapshot owner_id and viewer_id do not match")
        self.owner_id = str(owner_id) if owner_id is not None else None
        self.viewer_id = self.owner_id
        self.episode = max(1, int(snapshot.get("episode", 1)))
        self.turns = max(0, int(snapshot.get("turns", 0)))
        self.total_turns = max(0, int(snapshot.get("total_turns", self.turns)))
        self.last_round = max(0, int(snapshot.get("last_round", 0)))
        self.last_event_seq = max(0, int(snapshot.get("last_event_seq", 0)))
        self.self_state = deepcopy(dict(snapshot.get("self_state") or {}))
        self.visible_players = deepcopy(list(snapshot.get("visible_players") or []))
        self.private_facts = deepcopy(dict(snapshot.get("private_facts") or {}))
        # Optional in schema v1 so snapshots created before sheriff support
        # continue to restore without a migration.
        self.strategy_state = deepcopy(dict(snapshot.get("strategy_state") or {}))
        self.experience = {}
        if snapshot.get("experience"):
            strategy = RoleStrategy.from_dict(snapshot["experience"])
            if self.role is not None and strategy.role is not self.role:
                raise ValueError("memory experience role mismatch")
            self.experience = strategy.to_dict()
        self.events = deepcopy(list(snapshot.get("events") or []))
        self.decisions = deepcopy(list(snapshot.get("decisions") or []))
        self.reflections = [str(item) for item in snapshot.get("reflections") or []]
        self.episodes = deepcopy(list(snapshot.get("episodes") or []))
        self._seen_event_seqs = {
            int(item) for item in snapshot.get("seen_event_seqs") or []
        }
        limits = snapshot.get("limits") or {}
        self.max_events = max(1, int(limits.get("max_events", self.max_events)))
        self.max_decisions = max(
            1, int(limits.get("max_decisions", self.max_decisions))
        )
        self.max_episodes = max(1, int(limits.get("max_episodes", self.max_episodes)))
