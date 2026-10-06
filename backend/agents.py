"""Role-specialized autonomous agents for the Werewolf engine.

The engine passes an agent only a sanitized :class:`~backend.models.Observation`.
This module deliberately keeps that object as the sole source of game facts:
role memory, planners, and optional model calls never receive a game object or
the referee event stream.

Five concrete agents are provided -- :class:`WerewolfAgent`,
:class:`SeerAgent`, :class:`WitchAgent`, :class:`HunterAgent`, and
:class:`VillagerAgent`. Each owns a role goal, durable memory, a role prompt,
and a distinct planner. An LLM provider is optional; without one every agent
remains a complete, deterministic-under-seed baseline policy.
"""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .models import Action, ActionType, Observation, Role


@runtime_checkable
class Agent(Protocol):
    """Synchronous pluggable player policy."""

    def decide(self, observation: Observation, rng: random.Random) -> Action:
        """Choose one legal action from a private observation."""


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-neutral interface for plugging an LLM into an agent."""

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        """Return one text completion."""


def _legal(observation: Observation) -> dict[str, Any]:
    if not observation.legal_actions:
        raise ValueError("agent was asked to act without a legal action")
    return observation.legal_actions[0]


def _suspicion_from_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Build an explainable suspicion model from visible public statements."""

    scores: dict[str, float] = {}
    for event in events:
        if event.get("event_type") != "speech":
            continue
        data = event.get("data") or {}
        target = data.get("accuse_target")
        if not target:
            continue
        weight = 3.0 if data.get("claim_role") == Role.SEER.value else 1.0
        scores[str(target)] = scores.get(str(target), 0.0) + weight
    return scores


def _event_suspicion(observation: Observation) -> dict[str, float]:
    return _suspicion_from_events(observation.events)


def _best_target(
    candidates: Sequence[str], scores: Mapping[str, float], rng: random.Random
) -> Optional[str]:
    if not candidates:
        return None
    best_score = max((scores.get(item, 0.0) for item in candidates), default=0.0)
    tied = sorted(item for item in candidates if scores.get(item, 0.0) == best_score)
    return rng.choice(tied)


def _parse_json_object(text: str) -> Mapping[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, Mapping):
        raise ValueError("LLM response must be a JSON object")
    return value


def _action_is_legal(action: Action, observation: Observation) -> bool:
    """Validate a provider action before it can cross the agent boundary."""

    if action.actor_id != observation.viewer_id:
        return False
    if action.action_type is ActionType.PASS:
        return action.target_id is None and any(
            bool(rule.get("can_pass")) for rule in observation.legal_actions
        )

    for rule in observation.legal_actions:
        try:
            expected = ActionType.coerce(rule.get("action_type"))
        except (TypeError, ValueError):
            continue
        if action.action_type is not expected:
            continue
        targets = [str(item) for item in rule.get("target_ids", [])]
        if action.target_id is not None and action.target_id not in targets:
            return False
        if expected is not ActionType.SPEAK and targets and action.target_id is None:
            return False
        if bool(rule.get("content_required")) and not (action.content or "").strip():
            return False
        return True
    return False


@dataclass(frozen=True, slots=True)
class RoleGoal:
    """Immutable objective specification owned by one role agent."""

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
    """Durable, serializable memory populated only from visible observations.

    The memory survives repeated ``decide`` calls. If the observation appears
    to belong to a new game, current visible history is archived rather than
    discarded. Snapshots can be persisted to reuse an agent between games.
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
    events: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    _seen_event_seqs: set[int] = field(default_factory=set, repr=False)
    max_events: int = 256
    max_decisions: int = 128
    max_episodes: int = 16

    def ingest(self, observation: Observation) -> None:
        """Remember exactly the facts present in ``observation``."""

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
        self.self_state = deepcopy(observation.self_state)
        self.visible_players = deepcopy(observation.players)
        self.private_facts = deepcopy(observation.private_facts)

        for event in observation.events:
            seq = event.get("seq")
            if not isinstance(seq, int) or seq in self._seen_event_seqs:
                continue
            self._seen_event_seqs.add(seq)
            self.events.append(deepcopy(event))
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def remember_decision(
        self,
        action: Action,
        observation: Observation,
        *,
        source: str,
    ) -> None:
        self.decisions.append(
            {
                "episode": self.episode,
                "round": observation.round,
                "phase": observation.phase.value,
                "source": source,
                "action": action.to_dict(),
            }
        )
        if len(self.decisions) > self.max_decisions:
            self.decisions = self.decisions[-self.max_decisions :]

    def add_reflection(self, text: str) -> None:
        """Store a reflection supplied by an evaluator, not a hidden fact."""

        clean = str(text).strip()
        if clean:
            self.reflections.append(clean[:1000])
            self.reflections = self.reflections[-64:]

    def begin_episode(
        self,
        *,
        viewer_id: Optional[str] = None,
        role: Optional[Role | str] = None,
    ) -> None:
        """Archive current visible history and start a fresh game episode."""

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
                }
            )
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
        self.events = []
        self.decisions = []
        self.reflections = []
        self._seen_event_seqs = set()

    def prompt_view(self) -> dict[str, Any]:
        """Compact visible memory supplied to an optional model."""

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
        }

    def export(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable snapshot."""

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
        memory = cls()
        memory.restore(snapshot)
        return memory

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Replace memory with a previously exported snapshot."""

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


class RolePlanner(ABC):
    """Base interface for role-specific, dependency-free planners."""

    role: Role

    @abstractmethod
    def plan(
        self,
        observation: Observation,
        memory: RoleMemory,
        goal: RoleGoal,
        rng: random.Random,
    ) -> Action:
        """Return a legal baseline action for this role."""

    @staticmethod
    def _common_speech(
        observation: Observation,
        rng: random.Random,
        *,
        excluded: Sequence[str] = (),
    ) -> Action:
        actor = observation.viewer_id
        legal = _legal(observation)
        targets = [str(item) for item in legal.get("target_ids", [])]
        excluded_set = set(excluded)
        candidates = [
            target for target in targets if target != actor and target not in excluded_set
        ]
        candidates = candidates or [target for target in targets if target != actor]
        suspicion = _event_suspicion(observation)
        target = _best_target(candidates, suspicion, rng)
        names = {str(player["id"]): str(player["name"]) for player in observation.players}
        if target:
            content = (
                f"My current suspicion is {names.get(target, target)}. Their position "
                "needs more explanation; unless that changes, I recommend voting "
                "them out."
            )
            metadata = {"accuse_target": target, "vote_intent": target}
        else:
            content = "I have no firm suspect yet; I will compare statements and votes."
            metadata = {}
        return Action(
            actor,
            ActionType.SPEAK,
            target_id=target,
            content=content,
            metadata=metadata,
        )

    @staticmethod
    def _common_vote(
        observation: Observation,
        rng: random.Random,
        *,
        score_adjustments: Optional[Mapping[str, float]] = None,
        excluded: Sequence[str] = (),
    ) -> Action:
        actor = observation.viewer_id
        legal = _legal(observation)
        targets = [str(item) for item in legal.get("target_ids", [])]
        excluded_set = set(excluded)
        preferred = [target for target in targets if target not in excluded_set] or targets
        scores = _event_suspicion(observation)
        for target, amount in (score_adjustments or {}).items():
            scores[str(target)] = scores.get(str(target), 0.0) + float(amount)
        target = _best_target(preferred, scores, rng)
        return Action(actor, ActionType.VOTE, target_id=target)

    @staticmethod
    def _pass(actor: str) -> Action:
        return Action(actor, ActionType.PASS)


class WerewolfPlanner(RolePlanner):
    """Team-aware deception, night coordination, and threat removal."""

    role = Role.WEREWOLF

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del memory, goal
        legal = _legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        teammates = [str(item) for item in observation.private_facts.get("teammates", [])]
        if kind is ActionType.WOLF_KILL:
            threat = {target: 0.0 for target in targets}
            for event in observation.events:
                if event.get("event_type") == "wolf_vote" and event.get("round") == observation.round:
                    target = (event.get("data") or {}).get("target_id")
                    if target in threat:
                        threat[str(target)] += 12.0
                if event.get("event_type") == "speech":
                    data = event.get("data") or {}
                    speaker = event.get("actor_id")
                    if speaker in threat and data.get("claim_role") == Role.SEER.value:
                        threat[str(speaker)] += 8.0
            return Action(actor, kind, target_id=_best_target(targets, threat, rng))
        if kind is ActionType.SPEAK:
            return self._common_speech(observation, rng, excluded=teammates)
        if kind is ActionType.VOTE:
            adjustments: dict[str, float] = {item: -30.0 for item in teammates}
            for event in observation.events:
                data = event.get("data") or {}
                speaker = event.get("actor_id")
                if speaker in targets and data.get("claim_role") == Role.SEER.value:
                    adjustments[str(speaker)] = adjustments.get(str(speaker), 0.0) + 6.0
            return self._common_vote(observation, rng, score_adjustments=adjustments, excluded=teammates)
        return self._pass(actor)


class SeerPlanner(RolePlanner):
    """Information acquisition, result disclosure, and evidence-led voting."""

    role = Role.SEER

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del memory, goal
        legal = _legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        results = observation.private_facts.get("seer_results") or {}
        if kind is ActionType.SEER_CHECK:
            unchecked = [target for target in targets if target not in results]
            pool = sorted(unchecked or targets)
            return Action(actor, kind, target_id=rng.choice(pool) if pool else None)
        if kind is ActionType.SPEAK:
            names = {str(player["id"]): str(player["name"]) for player in observation.players}
            wolves = sorted(str(pid) for pid, result in results.items() if result == Role.WEREWOLF.value)
            if wolves:
                target = wolves[0]
                return Action(
                    actor,
                    kind,
                    target_id=target,
                    content=f"I am the Seer. I checked {names.get(target, target)} and they are a werewolf. I recommend voting them out.",
                    metadata={"claim_role": Role.SEER.value, "accuse_target": target, "vote_intent": target},
                )
            if results:
                target = sorted(str(pid) for pid in results)[-1]
                return Action(
                    actor,
                    kind,
                    target_id=target,
                    content=f"I am the Seer. My check on {names.get(target, target)} was not a werewolf result.",
                    metadata={"claim_role": Role.SEER.value, "defend_target": target},
                )
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            adjustments = {
                str(target): (20.0 if result == Role.WEREWOLF.value else -8.0)
                for target, result in results.items()
            }
            return self._common_vote(observation, rng, score_adjustments=adjustments)
        return self._pass(actor)


class WitchPlanner(RolePlanner):
    """Conservative potion management backed by visible social evidence."""

    role = Role.WITCH

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del memory, goal
        legal = _legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        if kind is ActionType.WITCH_SAVE:
            victim = observation.private_facts.get("night_victim")
            if victim and str(victim) in targets:
                return Action(actor, kind, target_id=str(victim))
            return self._pass(actor)
        if kind is ActionType.WITCH_POISON:
            suspicion = _event_suspicion(observation)
            target = _best_target(targets, suspicion, rng)
            score = suspicion.get(target or "", 0.0)
            if target and (score >= 2.0 or (observation.round >= 3 and rng.random() < 0.45)):
                return Action(actor, kind, target_id=target)
            return self._pass(actor)
        if kind is ActionType.SPEAK:
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            return self._common_vote(observation, rng)
        return self._pass(actor)


class HunterPlanner(RolePlanner):
    """Public deduction plus a final evidence-based retaliation decision."""

    role = Role.HUNTER

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del memory, goal
        legal = _legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        if kind is ActionType.HUNTER_SHOOT:
            target = _best_target(targets, _event_suspicion(observation), rng)
            return Action(actor, kind, target_id=target) if target else self._pass(actor)
        if kind is ActionType.SPEAK:
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            return self._common_vote(observation, rng)
        return self._pass(actor)


class VillagerPlanner(RolePlanner):
    """Pure public-information deduction with consistent speech and voting."""

    role = Role.VILLAGER

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del memory, goal
        kind = ActionType.coerce(_legal(observation)["action_type"])
        actor = observation.viewer_id
        if kind is ActionType.SPEAK:
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            return self._common_vote(observation, rng)
        return self._pass(actor)


class RoleAgent:
    """Shared safety shell; role strategy lives in concrete planners."""

    ROLE: Role

    def __init__(
        self,
        *,
        provider: Optional[LLMProvider] = None,
        memory: Optional[RoleMemory] = None,
        temperature: float = 0.2,
        owner_id: Optional[str] = None,
    ) -> None:
        self.role = self.ROLE
        self.owner_id = str(owner_id) if owner_id is not None else None
        self.provider = provider
        self.temperature = float(temperature)
        self.goal = self._make_goal()
        self.memory = memory if memory is not None else RoleMemory(
            role=self.role,
            owner_id=self.owner_id,
            viewer_id=self.owner_id,
        )
        if self.memory.role is None:
            self.memory.role = self.role
        elif self.memory.role is not self.role:
            raise ValueError(
                f"memory role {self.memory.role.value} cannot be used by {self.role.value}"
            )
        if self.memory.owner_id is not None:
            if self.owner_id is not None and self.owner_id != self.memory.owner_id:
                raise ValueError("agent owner_id conflicts with memory owner_id")
            self.owner_id = self.memory.owner_id
        elif self.owner_id is not None:
            self.memory.owner_id = self.owner_id
            self.memory.viewer_id = self.owner_id
        self.prompt = self._make_prompt()
        self.role_prompt = self.prompt
        self.planner = self._make_planner()

    @abstractmethod
    def _make_goal(self) -> RoleGoal:
        raise NotImplementedError

    @abstractmethod
    def _make_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _make_planner(self) -> RolePlanner:
        raise NotImplementedError

    def decide(self, observation: Observation, rng: random.Random) -> Action:
        if not observation.legal_actions:
            raise ValueError("agent was asked to act without a legal action")
        observed_role = Role.coerce(observation.self_state["role"])
        if observed_role is not self.role:
            raise ValueError(f"{type(self).__name__} cannot control role {observed_role.value}")
        if str(observation.self_state.get("id")) != observation.viewer_id:
            raise ValueError("observation viewer_id does not match self_state.id")
        if self.owner_id is None:
            self.owner_id = observation.viewer_id
            self.memory.owner_id = self.owner_id
            self.memory.viewer_id = self.owner_id
        elif self.owner_id != observation.viewer_id:
            raise ValueError(
                f"{type(self).__name__} belongs to {self.owner_id}, "
                f"not {observation.viewer_id}"
            )
        self.memory.ingest(observation)
        planned = self.planner.plan(observation, self.memory, self.goal, rng)
        if not _action_is_legal(planned, observation):
            planned = self._minimal_legal_action(observation, rng)
        action = planned
        source = "planner"
        if self.provider is not None:
            try:
                candidate = self._model_action(observation)
                if not _action_is_legal(candidate, observation):
                    raise ValueError("provider selected an action outside legal_actions")
                action = candidate
                source = "provider"
            except Exception:
                action = planned
                source = "planner_fallback"
        self.memory.remember_decision(action, observation, source=source)
        return action

    def _model_action(self, observation: Observation) -> Action:
        assert self.provider is not None
        system = (
            f"{self.prompt}\n\n"
            f"ROLE GOAL:\n{json.dumps(self.goal.to_dict(), ensure_ascii=False)}\n\n"
            "SECURITY AND OUTPUT RULES:\nUse only the supplied observation and memory. "
            "Never invent hidden roles, private events, or unavailable actions. Return "
            "exactly one JSON object with action_type, optional target_id, optional "
            "content, and optional metadata. Choose only from legal_actions. "
            'Example JSON: {"action_type":"vote","target_id":"p2",'
            '"content":null,"metadata":{}}.'
        )
        payload = {"observation": observation.to_dict(), "memory": self.memory.prompt_view()}
        raw = self.provider.complete(
            (
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ),
            temperature=self.temperature,
        )
        return Action.from_dict(_parse_json_object(raw), default_actor=observation.viewer_id)

    @staticmethod
    def _minimal_legal_action(observation: Observation, rng: random.Random) -> Action:
        actor = observation.viewer_id
        rule = _legal(observation)
        if rule.get("can_pass"):
            return Action(actor, ActionType.PASS)
        kind = ActionType.coerce(rule["action_type"])
        targets = [str(item) for item in rule.get("target_ids", [])]
        target = rng.choice(sorted(targets)) if targets else None
        content = "I will evaluate only the information currently available."
        return Action(actor, kind, target_id=target, content=content if rule.get("content_required") else None)


class WerewolfAgent(RoleAgent):
    ROLE = Role.WEREWOLF

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "Reach parity with the village while keeping the wolf team alive.",
            ("Coordinate night eliminations with known wolf teammates.", "Remove information roles that threaten the wolf team.", "Maintain a believable public position and avoid exposing teammates."),
            ("Use only visible public evidence and the supplied teammate list.", "Never target a known teammate at night."),
        )

    def _make_prompt(self) -> str:
        return "You are the WEREWOLF role. Collaborate with teammate identities in private_facts, converge on a night target, and use careful public deception. Do not reveal the wolf team or claim access to hidden facts."

    def _make_planner(self) -> RolePlanner:
        return WerewolfPlanner()


class SeerAgent(RoleAgent):
    ROLE = Role.SEER

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "Help the village eliminate every werewolf using verified inspections.",
            ("Inspect useful unchecked players each night.", "Preserve and communicate check results at strategically useful times.", "Align votes with confirmed wolf and non-wolf evidence."),
            ("Treat only seer_results as verified private checks.", "Do not fabricate an inspection result."),
        )

    def _make_prompt(self) -> str:
        return "You are the SEER role. Build a verified result ledger from seer_results, choose unchecked targets, and distinguish facts from social suspicion. Never present an unperformed check as a result."

    def _make_planner(self) -> RolePlanner:
        return SeerPlanner()


class WitchAgent(RoleAgent):
    ROLE = Role.WITCH

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "Use the antidote and poison to maximize the village's chance of survival.",
            ("Track the one-use potion resources across the game.", "Save high-value lives when the antidote is available.", "Poison only when visible evidence is strong enough."),
            ("Use only night_victim and potion facts supplied in private_facts.", "Do not disclose private night information as certain public fact."),
        )

    def _make_prompt(self) -> str:
        return "You are the WITCH role. Treat potions as scarce long-horizon resources. Use only the supplied night_victim and potions state, and base poison decisions on accumulated visible evidence rather than guesses."

    def _make_planner(self) -> RolePlanner:
        return WitchPlanner()


class HunterAgent(RoleAgent):
    ROLE = Role.HUNTER

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "Support village deduction and make a high-confidence final shot if killed.",
            ("Develop a consistent public read from statements and votes.", "Use the death shot against the strongest evidence-backed suspect.", "Avoid an unsupported shot when passing is safer."),
            ("Do not imply knowledge beyond public evidence.", "Choose a shot only from the current legal targets."),
        )

    def _make_prompt(self) -> str:
        return "You are the HUNTER role. Reason from public evidence while alive and keep a ranked suspect list for a possible final shot. Do not claim investigative powers or hidden information."

    def _make_planner(self) -> RolePlanner:
        return HunterPlanner()


class VillagerAgent(RoleAgent):
    ROLE = Role.VILLAGER

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "Help the village identify and eliminate every werewolf.",
            ("Compare public claims, accusations, and vote consistency.", "Explain suspicions clearly and revise them when evidence changes.", "Vote for the strongest public-evidence suspect."),
            ("Reason only from public information and your own role.", "Never invent a power-role result."),
        )

    def _make_prompt(self) -> str:
        return "You are the VILLAGER role with no private investigative power. Infer from public speech, claims, and voting behavior; keep statements consistent with what an ordinary villager can know."

    def _make_planner(self) -> RolePlanner:
        return VillagerPlanner()


class HeuristicAgent:
    """Backward-compatible facade dispatching to a specialized role agent."""

    def __init__(self, style: str = "balanced") -> None:
        self.style = style
        self._delegates: dict[Role, RoleAgent] = {}

    def decide(self, observation: Observation, rng: random.Random) -> Action:
        role = Role.coerce(observation.self_state["role"])
        delegate = self._delegates.get(role)
        if delegate is None:
            delegate = _agent_for_role(role)
            self._delegates[role] = delegate
        return delegate.decide(observation, rng)


@dataclass(slots=True)
class LLMAgent:
    """Generic provider policy retained for compatibility."""

    provider: LLMProvider
    fallback: Agent = field(default_factory=HeuristicAgent)
    temperature: float = 0.2

    def decide(self, observation: Observation, rng: random.Random) -> Action:
        system = (
            "You are a player in a game of Werewolf. Use only the supplied "
            "observation; never assume hidden roles or events. Return one JSON "
            "object with action_type, optional target_id, optional content, and "
            "optional metadata. Choose only from legal_actions."
        )
        payload = json.dumps(observation.to_dict(), ensure_ascii=False, sort_keys=True)
        try:
            raw = self.provider.complete(
                ({"role": "system", "content": system}, {"role": "user", "content": payload}),
                temperature=self.temperature,
            )
            candidate = Action.from_dict(_parse_json_object(raw), default_actor=observation.viewer_id)
            if not _action_is_legal(candidate, observation):
                raise ValueError("provider selected an action outside legal_actions")
            return candidate
        except Exception:
            return self.fallback.decide(observation, rng)

    @staticmethod
    def _parse_json(text: str) -> Mapping[str, Any]:
        return _parse_json_object(text)


@dataclass(slots=True)
class ScriptedAgent:
    """Testing/demo policy that consumes pre-built actions then falls back."""

    actions: deque[Action | Mapping[str, Any]] = field(default_factory=deque)
    fallback: Agent = field(default_factory=HeuristicAgent)

    def __init__(self, actions: Sequence[Action | Mapping[str, Any]], fallback: Optional[Agent] = None) -> None:
        self.actions = deque(actions)
        self.fallback = fallback or HeuristicAgent()

    def decide(self, observation: Observation, rng: random.Random) -> Action:
        if not self.actions:
            return self.fallback.decide(observation, rng)
        item = self.actions.popleft()
        if isinstance(item, Action):
            if not item.actor_id:
                item.actor_id = observation.viewer_id
            return item
        return Action.from_dict(item, default_actor=observation.viewer_id)


def _agent_for_role(
    role: Role | str,
    *,
    provider: Optional[LLMProvider] = None,
    memory: Optional[RoleMemory] = None,
    temperature: float = 0.2,
    owner_id: Optional[str] = None,
) -> RoleAgent:
    role = Role.coerce(role)
    classes: dict[Role, type[RoleAgent]] = {
        Role.WEREWOLF: WerewolfAgent,
        Role.SEER: SeerAgent,
        Role.WITCH: WitchAgent,
        Role.HUNTER: HunterAgent,
        Role.VILLAGER: VillagerAgent,
    }
    return classes[role](
        provider=provider,
        memory=memory,
        temperature=temperature,
        owner_id=owner_id,
    )


AgentFactory = Callable[[str, Role], Agent]


def default_agent_factory(
    player_id: str,
    role: Role,
    *,
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.2,
) -> Agent:
    """Create an independent, role-specialized baseline agent."""

    return _agent_for_role(
        role,
        provider=provider,
        temperature=temperature,
        owner_id=player_id,
    )


def role_agent_factory(
    provider: LLMProvider,
    *,
    temperature: float = 0.2,
) -> AgentFactory:
    """Bind one provider to fresh, owner-isolated role agents.

    The provider may be shared because it is only a transport adapter. Every
    factory call still creates a new goal, memory, prompt, and planner for the
    assigned seat.
    """

    def create(player_id: str, role: Role) -> Agent:
        return default_agent_factory(
            player_id,
            role,
            provider=provider,
            temperature=temperature,
        )

    return create


__all__ = [
    "Agent", "AgentFactory", "LLMProvider", "RoleGoal", "RoleMemory", "RolePlanner",
    "WerewolfPlanner", "SeerPlanner", "WitchPlanner", "HunterPlanner", "VillagerPlanner",
    "RoleAgent", "WerewolfAgent", "SeerAgent", "WitchAgent", "HunterAgent", "VillagerAgent",
    "HeuristicAgent", "LLMAgent", "ScriptedAgent", "default_agent_factory",
    "role_agent_factory",
]
