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

from .models import Action, ActionType, Observation, Phase, Role


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


def _legal_rules(observation: Observation) -> dict[ActionType, dict[str, Any]]:
    """Index all legal rules, including multi-choice sheriff decisions."""

    rules: dict[ActionType, dict[str, Any]] = {}
    for rule in observation.legal_actions:
        try:
            kind = ActionType.coerce(rule.get("action_type"))
        except (TypeError, ValueError):
            continue
        rules[kind] = rule
    return rules


def _seat_key(player_id: str) -> tuple[int, int | str]:
    """Sort conventional ``p12`` seats numerically and other ids stably."""

    match = re.fullmatch(r"p(\d+)", str(player_id), flags=re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    return (1, str(player_id))


def _visible_player_ids(
    observation: Observation, *, alive_only: bool = False
) -> list[str]:
    return [
        str(player["id"])
        for player in observation.players
        if player.get("id") is not None
        and (not alive_only or bool(player.get("alive")))
    ]


def _public_metadata_is_valid(action: Action, observation: Observation) -> bool:
    """Validate structured public claims without consulting hidden state.

    A badge-flow declaration communicates only future player references. It
    must never carry inspection results (real or fabricated) as hidden fields;
    actual claims belong in public speech content and established public claim
    metadata.
    """

    metadata = action.metadata
    if not isinstance(metadata, Mapping):
        return False
    player_ids = set(_visible_player_ids(observation))

    flow = metadata.get("badge_flow")
    if flow is not None:
        if not isinstance(flow, Mapping):
            return False
        if set(flow) != {"mode", "targets", "fallback_target"}:
            return False
        mode = flow.get("mode")
        if mode not in {"single", "double"}:
            return False
        targets = flow.get("targets")
        if not isinstance(targets, list):
            return False
        expected_count = 1 if mode == "single" else 2
        if len(targets) != expected_count:
            return False
        normalized_targets = [str(item) for item in targets]
        if len(set(normalized_targets)) != len(normalized_targets):
            return False
        if any(target not in player_ids for target in normalized_targets):
            return False
        if observation.viewer_id in normalized_targets:
            return False
        fallback = flow.get("fallback_target")
        if fallback is not None:
            fallback = str(fallback)
            if (
                fallback not in player_ids
                or fallback == observation.viewer_id
                or fallback in normalized_targets
            ):
                return False

    nomination = metadata.get("sheriff_nomination")
    if nomination is not None:
        nomination = str(nomination)
        if nomination not in player_ids or nomination == observation.viewer_id:
            return False
        if (
            action.action_type is ActionType.SHERIFF_SUMMARIZE
            and action.target_id != nomination
        ):
            return False

    # These pre-existing speech fields also carry player references.  Keep
    # validation at the agent boundary so a provider can never smuggle an
    # unknown seat into the public event stream.
    for key in ("accuse_target", "defend_target", "vote_intent"):
        referenced = metadata.get(key)
        if referenced is not None and str(referenced) not in player_ids:
            return False
    claimed_role = metadata.get("claim_role")
    if claimed_role is not None and str(claimed_role) not in {
        role.value for role in Role
    }:
        return False
    return True


def _suspicion_from_events(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Build an explainable suspicion model from visible public statements."""

    scores: dict[str, float] = {}
    public_speech_types = {
        "speech",
        "sheriff_speech",
        "sheriff_pk_speech",
        "sheriff_summary",
        "day_pk_speech",
    }
    for event in events:
        if event.get("event_type") not in public_speech_types:
            continue
        data = event.get("data") or {}
        if not isinstance(data, Mapping):
            continue
        claimed_seer = data.get("claim_role") == Role.SEER.value
        target = data.get("accuse_target")
        if target:
            weight = 3.0 if claimed_seer else 1.0
            scores[str(target)] = scores.get(str(target), 0.0) + weight

        defended = data.get("defend_target")
        if defended:
            weight = 2.0 if claimed_seer else 0.75
            scores[str(defended)] = scores.get(str(defended), 0.0) - weight

        # A Sheriff summary is the formal consolidation recommendation.  It
        # should influence baseline voters, while role-private evidence can
        # still override it through each planner's score adjustments.
        nomination = data.get("sheriff_nomination")
        if nomination:
            scores[str(nomination)] = scores.get(str(nomination), 0.0) + 4.0
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
    if not _public_metadata_is_valid(action, observation):
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
    strategy_state: dict[str, Any] = field(default_factory=dict)
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
        sheriff_public = observation.private_facts.get("sheriff")
        if isinstance(sheriff_public, Mapping):
            # The engine exposes this exact object to every viewpoint.  Store
            # only that public overlay in strategy_state; role-private facts
            # such as checks, potions, and wolf teammates remain elsewhere.
            self.strategy_state["sheriff_public"] = deepcopy(dict(sheriff_public))

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
                    "strategy_state": deepcopy(self.strategy_state),
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
        self.strategy_state = {}
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
            "strategy_state": deepcopy(self.strategy_state),
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
            "strategy_state": deepcopy(self.strategy_state),
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
        # Optional in schema v1 so snapshots created before sheriff support
        # continue to restore without a migration.
        self.strategy_state = deepcopy(dict(snapshot.get("strategy_state") or {}))
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

    @staticmethod
    def _ranked_target(
        observation: Observation,
        rule: Mapping[str, Any],
        rng: random.Random,
        *,
        score_adjustments: Optional[Mapping[str, float]] = None,
        excluded: Sequence[str] = (),
        prefer_low_score: bool = False,
    ) -> Optional[str]:
        targets = [str(item) for item in rule.get("target_ids", [])]
        excluded_set = {str(item) for item in excluded}
        pool = [target for target in targets if target not in excluded_set] or targets
        if not pool:
            return None
        scores = _event_suspicion(observation)
        for target, amount in (score_adjustments or {}).items():
            scores[str(target)] = scores.get(str(target), 0.0) + float(amount)
        extreme = (
            min(scores.get(target, 0.0) for target in pool)
            if prefer_low_score
            else max(scores.get(target, 0.0) for target in pool)
        )
        tied = sorted(
            target for target in pool if scores.get(target, 0.0) == extreme
        )
        return rng.choice(tied)

    @staticmethod
    def _sheriff_order(
        observation: Observation,
        *,
        score_adjustments: Optional[Mapping[str, float]] = None,
    ) -> Action:
        """Put the strongest public suspect as early as possible."""

        actor = observation.viewer_id
        rules = _legal_rules(observation)
        clockwise = ActionType.SHERIFF_ORDER_CLOCKWISE
        counterclockwise = ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE
        if clockwise not in rules:
            return Action(actor, counterclockwise)
        if counterclockwise not in rules:
            return Action(actor, clockwise)

        alive = sorted(
            _visible_player_ids(observation, alive_only=True), key=_seat_key
        )
        if actor not in alive or len(alive) <= 2:
            return Action(actor, clockwise)
        anchor = alive.index(actor)
        cw_order = alive[anchor + 1 :] + alive[:anchor]
        ccw_order = list(reversed(cw_order))
        suspicion = _event_suspicion(observation)
        for target, amount in (score_adjustments or {}).items():
            suspicion[str(target)] = suspicion.get(str(target), 0.0) + float(amount)

        def order_score(order: Sequence[str]) -> float:
            size = len(order)
            return sum(
                (size - index) * suspicion.get(player_id, 0.0)
                for index, player_id in enumerate(order)
            )

        selected = counterclockwise if order_score(ccw_order) > order_score(cw_order) else clockwise
        return Action(actor, selected)

    def _plan_sheriff_action(
        self,
        observation: Observation,
        memory: RoleMemory,
        rng: random.Random,
        *,
        run_for_sheriff: bool,
        stay_in_race: bool,
        score_adjustments: Optional[Mapping[str, float]] = None,
        excluded: Sequence[str] = (),
        preferred_badge_targets: Sequence[str] = (),
        destroy_badge: bool = False,
    ) -> Optional[Action]:
        """Handle role-independent sheriff powers and election mechanics."""

        del memory
        actor = observation.viewer_id
        rules = _legal_rules(observation)
        if ActionType.SHERIFF_RUN in rules:
            if run_for_sheriff:
                return Action(actor, ActionType.SHERIFF_RUN)
            return self._pass(actor)

        if ActionType.SHERIFF_WITHDRAW in rules:
            if stay_in_race:
                return self._pass(actor)
            return Action(actor, ActionType.SHERIFF_WITHDRAW)

        if ActionType.SHERIFF_VOTE in rules:
            rule = rules[ActionType.SHERIFF_VOTE]
            target = self._ranked_target(
                observation,
                rule,
                rng,
                score_adjustments=score_adjustments,
                excluded=excluded,
            )
            if target is None:
                return self._pass(actor)
            return Action(actor, ActionType.SHERIFF_VOTE, target_id=target)

        if (
            ActionType.SHERIFF_ORDER_CLOCKWISE in rules
            or ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE in rules
        ):
            return self._sheriff_order(
                observation,
                score_adjustments=score_adjustments,
            )

        if ActionType.SHERIFF_SUMMARIZE in rules:
            rule = rules[ActionType.SHERIFF_SUMMARIZE]
            target = self._ranked_target(
                observation,
                rule,
                rng,
                score_adjustments=score_adjustments,
                excluded=excluded,
            )
            if target is None:
                # An engine should always provide a nomination target. Keeping
                # this path legal makes standalone policies robust to an empty
                # target list in an interrupted game.
                return self._pass(actor)
            names = {
                str(player["id"]): str(player.get("name", player["id"]))
                for player in observation.players
            }
            return Action(
                actor,
                ActionType.SHERIFF_SUMMARIZE,
                target_id=target,
                content=(
                    f"After hearing everyone, I nominate {names.get(target, target)} "
                    "for today's elimination. Compare your reads, then consolidate "
                    "the vote there."
                ),
                metadata={"sheriff_nomination": target, "vote_intent": target},
            )

        badge_rules = {
            kind: rules[kind]
            for kind in (ActionType.SHERIFF_TRANSFER, ActionType.SHERIFF_DESTROY)
            if kind in rules
        }
        if badge_rules:
            transfer = badge_rules.get(ActionType.SHERIFF_TRANSFER)
            if transfer is not None:
                legal_targets = [str(item) for item in transfer.get("target_ids", [])]
                for target in preferred_badge_targets:
                    if str(target) in legal_targets:
                        return Action(
                            actor,
                            ActionType.SHERIFF_TRANSFER,
                            target_id=str(target),
                        )
            if destroy_badge and ActionType.SHERIFF_DESTROY in badge_rules:
                return Action(actor, ActionType.SHERIFF_DESTROY)
            if transfer is not None:
                target = self._ranked_target(
                    observation,
                    transfer,
                    rng,
                    score_adjustments=score_adjustments,
                    excluded=excluded,
                    prefer_low_score=True,
                )
                if target is not None:
                    return Action(
                        actor, ActionType.SHERIFF_TRANSFER, target_id=target
                    )
            if ActionType.SHERIFF_DESTROY in badge_rules:
                return Action(actor, ActionType.SHERIFF_DESTROY)
            return self._pass(actor)
        return None


class WerewolfPlanner(RolePlanner):
    """Team-aware deception, night coordination, and threat removal."""

    role = Role.WEREWOLF

    @staticmethod
    def _wolf_ids(observation: Observation) -> list[str]:
        actor = observation.viewer_id
        teammates = {
            str(item) for item in observation.private_facts.get("teammates", [])
        }
        teammates.add(actor)
        alive = set(_visible_player_ids(observation, alive_only=True))
        return sorted(teammates & alive, key=_seat_key)

    def _fake_seer_speech(
        self,
        observation: Observation,
        memory: RoleMemory,
    ) -> Action:
        actor = observation.viewer_id
        names = {
            str(player["id"]): str(player.get("name", player["id"]))
            for player in observation.players
        }
        wolf_ids = set(self._wolf_ids(observation))
        public_pool = [
            player_id
            for player_id in sorted(
                _visible_player_ids(observation, alive_only=True), key=_seat_key
            )
            if player_id not in wolf_ids
        ]
        state = memory.strategy_state.setdefault("werewolf_sheriff", {})
        ledger = state.setdefault("fake_seer_results", {})
        if not ledger and public_pool:
            # A stable fabricated gold-water result is safer than accidentally
            # accusing a teammate; later speeches reuse this exact ledger.
            ledger[public_pool[0]] = "not_werewolf"
        checked_target = next(iter(ledger), None)

        flow = state.get("badge_flow")
        if not isinstance(flow, Mapping):
            future = [target for target in public_pool if target not in ledger][:2]
            if future:
                fallback = next(
                    (
                        target
                        for target in public_pool
                        if target not in future and target not in ledger
                    ),
                    None,
                )
                flow = {
                    "mode": "double" if len(future) == 2 else "single",
                    "targets": future,
                    "fallback_target": fallback,
                }
                state["badge_flow"] = deepcopy(flow)

        metadata: dict[str, Any] = {"claim_role": Role.SEER.value}
        parts = ["I am the Seer and I am running for Sheriff."]
        if checked_target is not None:
            metadata["defend_target"] = checked_target
            parts.append(
                f"My first check was {names.get(checked_target, checked_target)}, "
                "who was not a werewolf."
            )
        if isinstance(flow, Mapping):
            metadata["badge_flow"] = deepcopy(dict(flow))
            labels = [names.get(str(item), str(item)) for item in flow["targets"]]
            parts.append("My badge flow is " + " then ".join(labels) + ".")
        return Action(
            actor,
            ActionType.SPEAK,
            target_id=checked_target,
            content=" ".join(parts),
            metadata=metadata,
        )

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del goal
        legal = _legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        teammates = [str(item) for item in observation.private_facts.get("teammates", [])]
        wolf_ids = self._wolf_ids(observation)
        bluffer = wolf_ids[0] if wolf_ids else actor
        is_bluffer = actor == bluffer
        memory.strategy_state.setdefault("werewolf_sheriff", {})[
            "bluffer_id"
        ] = bluffer

        rules = _legal_rules(observation)
        sheriff_adjustments: dict[str, float] = {
            teammate: -30.0 for teammate in teammates
        }
        sheriff_excluded: Sequence[str] = teammates
        if ActionType.SHERIFF_VOTE in rules:
            sheriff_adjustments[bluffer] = 100.0
            sheriff_excluded = ()
        sheriff_action = self._plan_sheriff_action(
            observation,
            memory,
            rng,
            run_for_sheriff=is_bluffer,
            stay_in_race=is_bluffer,
            score_adjustments=sheriff_adjustments,
            excluded=sheriff_excluded,
            preferred_badge_targets=sorted(teammates, key=_seat_key),
        )
        if sheriff_action is not None:
            return sheriff_action

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
            if observation.phase in (
                Phase.SHERIFF_SPEECH,
                Phase.SHERIFF_PK_SPEECH,
            ) and is_bluffer:
                return self._fake_seer_speech(observation, memory)
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

    @staticmethod
    def _badge_flow(
        observation: Observation,
        memory: RoleMemory,
    ) -> Optional[dict[str, Any]]:
        state = memory.strategy_state.setdefault("seer_sheriff", {})
        existing = state.get("badge_flow")
        if isinstance(existing, Mapping):
            return deepcopy(dict(existing))

        results = {
            str(target): str(result)
            for target, result in (
                observation.private_facts.get("seer_results") or {}
            ).items()
        }
        candidates = [
            player_id
            for player_id in sorted(
                _visible_player_ids(observation, alive_only=True), key=_seat_key
            )
            if player_id != observation.viewer_id and player_id not in results
        ]
        targets = candidates[:2]
        if not targets:
            return None
        fallback = next(
            (
                player_id
                for player_id in sorted(
                    _visible_player_ids(observation, alive_only=True), key=_seat_key
                )
                if player_id != observation.viewer_id
                and player_id not in targets
                and player_id not in results
            ),
            None,
        )
        flow = {
            "mode": "double" if len(targets) == 2 else "single",
            "targets": targets,
            "fallback_target": fallback,
        }
        state["badge_flow"] = deepcopy(flow)
        return flow

    @staticmethod
    def _badge_recipient(
        observation: Observation,
        memory: RoleMemory,
    ) -> Optional[str]:
        flow = memory.strategy_state.get("seer_sheriff", {}).get("badge_flow")
        if not isinstance(flow, Mapping):
            return None
        targets = [str(item) for item in flow.get("targets") or []]
        if not targets:
            return None
        results = {
            str(target): str(result)
            for target, result in (
                observation.private_facts.get("seer_results") or {}
            ).items()
        }

        first = targets[0]
        if first not in results:
            return None
        first_good = results[first] != Role.WEREWOLF.value
        if len(targets) == 1 or targets[1] not in results:
            return first if first_good else None

        second = targets[1]
        second_good = results[second] != Role.WEREWOLF.value
        if first_good != second_good:
            return first if first_good else second
        if first_good and second_good:
            return None
        fallback = flow.get("fallback_target")
        return str(fallback) if fallback is not None else None

    def _campaign_speech(
        self,
        observation: Observation,
        memory: RoleMemory,
    ) -> Action:
        actor = observation.viewer_id
        names = {
            str(player["id"]): str(player.get("name", player["id"]))
            for player in observation.players
        }
        results = {
            str(target): str(result)
            for target, result in (
                observation.private_facts.get("seer_results") or {}
            ).items()
        }
        metadata: dict[str, Any] = {"claim_role": Role.SEER.value}
        parts = ["I am the Seer and I am running for Sheriff."]
        target: Optional[str] = None
        if results:
            # Preserve engine insertion order so the first-night result is
            # reported before later checks.
            target, result = next(iter(results.items()))
            if result == Role.WEREWOLF.value:
                metadata.update(
                    {"accuse_target": target, "vote_intent": target}
                )
                parts.append(
                    f"I checked {names.get(target, target)} and found a werewolf."
                )
            else:
                metadata["defend_target"] = target
                parts.append(
                    f"I checked {names.get(target, target)} and found they are "
                    "not a werewolf."
                )
        flow = self._badge_flow(observation, memory)
        if flow is not None:
            metadata["badge_flow"] = deepcopy(flow)
            labels = [names.get(str(item), str(item)) for item in flow["targets"]]
            parts.append("My badge flow is " + " then ".join(labels) + ".")
        return Action(
            actor,
            ActionType.SPEAK,
            target_id=target,
            content=" ".join(parts),
            metadata=metadata,
        )

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del goal
        legal = _legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        results = observation.private_facts.get("seer_results") or {}
        adjustments = {
            str(target): (20.0 if result == Role.WEREWOLF.value else -8.0)
            for target, result in results.items()
        }

        rules = _legal_rules(observation)
        if (
            ActionType.SHERIFF_TRANSFER in rules
            or ActionType.SHERIFF_DESTROY in rules
        ):
            recipient = self._badge_recipient(observation, memory)
            sheriff_action = self._plan_sheriff_action(
                observation,
                memory,
                rng,
                run_for_sheriff=True,
                stay_in_race=True,
                score_adjustments=adjustments,
                preferred_badge_targets=([recipient] if recipient else ()),
                destroy_badge=True,
            )
        else:
            sheriff_action = self._plan_sheriff_action(
                observation,
                memory,
                rng,
                run_for_sheriff=True,
                stay_in_race=True,
                score_adjustments=adjustments,
            )
        if sheriff_action is not None:
            return sheriff_action

        if kind is ActionType.SEER_CHECK:
            unchecked = [target for target in targets if target not in results]
            flow = memory.strategy_state.get("seer_sheriff", {}).get("badge_flow")
            if isinstance(flow, Mapping):
                for planned_target in flow.get("targets") or ():
                    planned_target = str(planned_target)
                    if planned_target in unchecked:
                        return Action(actor, kind, target_id=planned_target)
            pool = sorted(unchecked or targets)
            return Action(actor, kind, target_id=rng.choice(pool) if pool else None)
        if kind is ActionType.SPEAK:
            if observation.phase in (
                Phase.SHERIFF_SPEECH,
                Phase.SHERIFF_PK_SPEECH,
            ):
                return self._campaign_speech(observation, memory)
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
            return self._common_vote(observation, rng, score_adjustments=adjustments)
        return self._pass(actor)


class WitchPlanner(RolePlanner):
    """Conservative potion management backed by visible social evidence."""

    role = Role.WITCH

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        del goal
        sheriff_action = self._plan_sheriff_action(
            observation,
            memory,
            rng,
            run_for_sheriff=False,
            stay_in_race=False,
        )
        if sheriff_action is not None:
            return sheriff_action
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
        del goal
        sheriff_action = self._plan_sheriff_action(
            observation,
            memory,
            rng,
            run_for_sheriff=False,
            stay_in_race=False,
        )
        if sheriff_action is not None:
            return sheriff_action
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
        del goal
        sheriff_action = self._plan_sheriff_action(
            observation,
            memory,
            rng,
            run_for_sheriff=False,
            stay_in_race=False,
        )
        if sheriff_action is not None:
            return sheriff_action
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
            "For a public badge-flow declaration, metadata.badge_flow must be exactly "
            '{"mode":"single|double","targets":["player_id"],'
            '"fallback_target":"player_id or null"}; use exactly one target for '
            "single and two for double, and never include inspection truth in it. "
            "A Sheriff summary must set metadata.sheriff_nomination equal to target_id. "
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
            ("Coordinate night eliminations with known wolf teammates.", "Remove information roles that threaten the wolf team.", "Maintain a believable public position and avoid exposing teammates.", "Use one consistent low-seat bluffing candidate to contest the Sheriff badge."),
            ("Use only visible public evidence and the supplied teammate list.", "Never target a known teammate at night.", "Keep fabricated checks and badge-flow declarations consistent."),
        )

    def _make_prompt(self) -> str:
        return "You are the WEREWOLF role. Collaborate with teammate identities in private_facts, converge on a night target, and use careful public deception. The lowest-seat living wolf should run for Sheriff and may bluff as Seer with a stable fabricated check ledger and badge flow; other wolves support that candidate. As Sheriff, choose speaking order, nominate a target, and transfer the badge strategically. Do not reveal the wolf team or claim access to hidden facts."

    def _make_planner(self) -> RolePlanner:
        return WerewolfPlanner()


class SeerAgent(RoleAgent):
    ROLE = Role.SEER

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "Help the village eliminate every werewolf using verified inspections.",
            ("Inspect useful unchecked players each night.", "Preserve and communicate check results at strategically useful times.", "Run for Sheriff, publish a one- or two-night badge flow, and encode completed checks in any final badge transfer.", "Align votes with confirmed wolf and non-wolf evidence."),
            ("Treat only seer_results as verified private checks.", "Do not fabricate an inspection result.", "A badge-flow declaration names future targets but never exposes private truth before a public claim."),
        )

    def _make_prompt(self) -> str:
        return "You are the SEER role. Build a verified result ledger from seer_results, choose unchecked targets, and distinguish facts from social suspicion. Always run for Sheriff, stay in the race, report genuine completed checks, and publish a stable single- or double-night badge flow. On death, transfer or destroy the badge according to completed checks. Never present an unperformed check as a result."

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
        return "You are the WITCH role. Treat potions as scarce long-horizon resources. Use only the supplied night_victim and potions state, and base poison decisions on accumulated visible evidence rather than guesses. You normally stay off the Sheriff ballot, but if elected you can choose speaking order, nominate a target, and transfer or destroy the badge using public evidence."

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
        return "You are the HUNTER role. Reason from public evidence while alive and keep a ranked suspect list for a possible final shot. You normally stay off the Sheriff ballot, but if elected use every Sheriff power from public evidence and handle the badge before taking a death shot. Do not claim investigative powers or hidden information."

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
        return "You are the VILLAGER role with no private investigative power. Infer from public speech, claims, and voting behavior; keep statements consistent with what an ordinary villager can know. You normally stay off the Sheriff ballot, but if elected choose speaking order, nominate a target, and transfer or destroy the badge from public evidence alone."

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
