"""Deterministic, information-isolated Werewolf game engine."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from .agents import Agent, AgentFactory, default_agent_factory
from .models import (
    Action,
    ActionType,
    GameConfig,
    GameEvent,
    GameStatus,
    LegalAction,
    Observation,
    PendingAction,
    Phase,
    PlayerState,
    Role,
    Team,
    ensure_action,
)


class InvalidAction(ValueError):
    """Raised when a human submits an action outside their legal action space."""


class WerewolfGame:
    """A complete turn-based Werewolf match.

    The engine is synchronous by design. ``step`` advances at most one player
    decision (or one automatic transition), while ``run`` advances until the
    game finishes or a human-controlled player needs input.

    Security boundary: autonomous agents only receive :class:`Observation`.
    They never receive this object, its player-role map, or referee events.
    """

    def __init__(
        self,
        config: GameConfig | Mapping[str, Any] | None = None,
        agents: Optional[Mapping[str, Agent]] = None,
        *,
        agent_factory: Optional[AgentFactory] = None,
    ) -> None:
        if config is None:
            config = GameConfig()
        elif isinstance(config, Mapping):
            config = GameConfig.from_dict(config)
        self.config = config
        self.rng = random.Random(config.seed)

        deck = [Role.coerce(role) for role in config.roles or ()]
        if config.shuffle_roles:
            self.rng.shuffle(deck)

        human_ids = self._resolve_player_references(config.human_players)
        self.players: dict[str, PlayerState] = {}
        for index, (name, role) in enumerate(zip(config.player_names, deck), start=1):
            player_id = f"p{index}"
            self.players[player_id] = PlayerState(
                id=player_id,
                name=name,
                role=role,
                controller="human" if player_id in human_ids else "ai",
            )

        supplied_agents = dict(agents or {})
        factory = agent_factory or default_agent_factory
        self.agents: dict[str, Agent] = {}
        for player in self.players.values():
            if player.controller == "human":
                continue
            candidate = supplied_agents.get(player.id, supplied_agents.get(player.name))
            self.agents[player.id] = candidate or factory(player.id, player.role)
        # Policies get independent deterministic streams.  An experimental
        # agent therefore cannot perturb referee tie-breaks by consuming extra
        # random numbers.
        self._agent_rngs: dict[str, random.Random] = {
            player_id: random.Random(config.seed + index * 1_000_003)
            for index, player_id in enumerate(self.players, start=1)
        }
        # Invalid-policy repair uses streams that policies never receive.
        # Otherwise an Agent could consume its provided RNG before returning
        # an invalid action and thereby steer the engine's safe fallback.
        self._fallback_rngs: dict[str, random.Random] = {
            player_id: random.Random(config.seed + 9_000_019 + index * 2_000_033)
            for index, player_id in enumerate(self.players, start=1)
        }

        self.events: list[GameEvent] = []
        self.round = 1
        self.phase = Phase.NIGHT_WOLF
        self.status = GameStatus.RUNNING
        self.winner: Optional[str] = None
        self.win_reason: Optional[str] = None
        self.pending_action: Optional[PendingAction] = None

        self._phase_queue: list[str] = []
        self._wolf_votes: dict[str, str] = {}
        self._day_votes: dict[str, str] = {}
        self._night_victim: Optional[str] = None
        self._night_saved = False
        self._night_poison_targets: list[str] = []
        self._witch_used_this_night: set[str] = set()
        self._witch_resources: dict[str, dict[str, bool]] = {
            player.id: {"antidote": True, "poison": True}
            for player in self.players.values()
            if player.role is Role.WITCH
        }
        self._seer_results: dict[str, dict[str, str]] = {
            player.id: {}
            for player in self.players.values()
            if player.role is Role.SEER
        }
        self._hunter_queue: list[str] = []
        self._after_hunter_phase: Optional[Phase] = None
        self._discussion_cycle = 1

        role_counts = Counter(role.value for role in deck)
        self._log(
            "game_started",
            f"Game started with {len(self.players)} players.",
            data={
                "player_count": len(self.players),
                "role_counts": dict(role_counts),
            },
        )
        wolf_ids = tuple(
            player.id for player in self.players.values() if player.role is Role.WEREWOLF
        )
        for player in self.players.values():
            self._log(
                "role_assigned",
                f"You are the {player.role.value}.",
                actor_id=player.id,
                data={"role": player.role.value, "team": player.team.value},
                visibility=(player.id,),
            )
            if player.role is Role.WEREWOLF:
                teammates = [pid for pid in wolf_ids if pid != player.id]
                self._log(
                    "wolf_team",
                    "Your werewolf teammates are known to you.",
                    actor_id=player.id,
                    data={"teammates": teammates},
                    visibility=(player.id,),
                )
        self._enter_phase(Phase.NIGHT_WOLF)

    # ------------------------------------------------------------------
    # Public API

    def step(
        self, action: Action | Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Advance one decision/transition and return a public snapshot."""

        if action is not None:
            return self.submit_action(action)
        if self.status is GameStatus.FINISHED or self.pending_action is not None:
            return self.public_state()

        handlers = {
            Phase.NIGHT_WOLF: self._step_wolf,
            Phase.NIGHT_SEER: self._step_seer,
            Phase.NIGHT_WITCH_SAVE: self._step_witch_save,
            Phase.NIGHT_WITCH_POISON: self._step_witch_poison,
            Phase.NIGHT_RESOLVE: self._resolve_night,
            Phase.DAY_DISCUSSION: self._step_discussion,
            Phase.DAY_VOTE: self._step_vote,
            Phase.HUNTER_SHOT: self._step_hunter,
        }
        handler = handlers.get(self.phase)
        if handler is not None:
            handler()
        return self.public_state()

    def run(self, max_steps: int = 10_000) -> dict[str, Any]:
        """Run autonomously until completion, a human prompt, or the step cap."""

        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        steps = 0
        while self.status is not GameStatus.FINISHED and self.pending_action is None:
            if steps >= max_steps:
                break
            self.step()
            steps += 1
        return self.public_state()

    def submit_action(
        self, value: Action | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Submit the currently pending human decision."""

        if self.pending_action is None:
            raise InvalidAction("the game is not waiting for a human action")
        action = ensure_action(value, default_actor=self.pending_action.actor_id)
        legal = self.pending_action.legal_actions
        self._validate_action(action, legal, self.pending_action.actor_id)
        self.pending_action = None
        self.status = GameStatus.RUNNING
        self._consume_action(action)
        return self.public_state(viewer_id=action.actor_id)

    def observe(
        self,
        viewer_id: str,
        legal_actions: Optional[Sequence[LegalAction]] = None,
    ) -> Observation:
        """Build an information-filtered observation for one player."""

        if viewer_id not in self.players:
            raise KeyError(f"unknown player: {viewer_id}")
        player = self.players[viewer_id]
        legal = list(legal_actions or ())
        private_facts: dict[str, Any] = {}
        if player.role is Role.WEREWOLF:
            private_facts["teammates"] = [
                other.id
                for other in self.players.values()
                if other.role is Role.WEREWOLF and other.id != viewer_id
            ]
        elif player.role is Role.SEER:
            private_facts["seer_results"] = dict(self._seer_results.get(viewer_id, {}))
        elif player.role is Role.WITCH:
            private_facts["potions"] = dict(self._witch_resources.get(viewer_id, {}))
            if self.phase in (Phase.NIGHT_WITCH_SAVE, Phase.NIGHT_WITCH_POISON):
                private_facts["night_victim"] = self._night_victim

        return Observation(
            viewer_id=viewer_id,
            round=self.round,
            phase=self.phase,
            self_state=player.to_private_dict(),
            players=self._players_view(viewer_id=viewer_id, omniscient=False),
            events=self.event_log(viewer_id=viewer_id),
            legal_actions=[item.to_dict() for item in legal],
            private_facts=private_facts,
        )

    def event_log(
        self, viewer_id: Optional[str] = None, *, omniscient: bool = False
    ) -> list[dict[str, Any]]:
        """Return the event stream filtered for a public/player/referee view."""

        if viewer_id is not None and viewer_id not in self.players:
            raise KeyError(f"unknown player: {viewer_id}")
        if omniscient:
            return [event.to_dict(include_visibility=True) for event in self.events]

        # Re-number a filtered stream so gaps in referee sequence numbers cannot
        # be used as a side channel to count another role's private actions.
        selected = [event for event in self.events if event.visible_to(viewer_id)]
        result: list[dict[str, Any]] = []
        for visible_seq, event in enumerate(selected, start=1):
            item = event.to_dict(include_visibility=False)
            item["seq"] = visible_seq
            result.append(item)
        return result

    def public_state(
        self, viewer_id: Optional[str] = None, *, omniscient: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-friendly state without leaking inaccessible facts."""

        if viewer_id is not None and viewer_id not in self.players:
            raise KeyError(f"unknown player: {viewer_id}")
        stage: str = self.phase.public_name
        if omniscient or self.phase in (
            Phase.DAY_DISCUSSION,
            Phase.DAY_VOTE,
            Phase.HUNTER_SHOT,
            Phase.FINISHED,
        ):
            stage = self.phase.value
        elif self.pending_action and self.pending_action.actor_id == viewer_id:
            stage = self.phase.value

        pending: Optional[dict[str, Any]] = None
        hidden_private_turn = bool(
            self.pending_action
            and self.phase.public_name == "night"
            and not omniscient
            and viewer_id != self.pending_action.actor_id
        )
        if self.pending_action:
            if omniscient or viewer_id == self.pending_action.actor_id:
                pending = self.pending_action.to_dict()
            elif not hidden_private_turn:
                is_public_turn = self.phase in (
                    Phase.DAY_DISCUSSION,
                    Phase.DAY_VOTE,
                    Phase.HUNTER_SHOT,
                )
                pending = {
                    "required": True,
                    "actor_id": self.pending_action.actor_id if is_public_turn else None,
                    "phase": self.phase.public_name,
                }

        visible_events = self.event_log(viewer_id=viewer_id, omniscient=omniscient)
        visible_status = self.status
        if self.status is GameStatus.WAITING_FOR_HUMAN and hidden_private_turn:
            # Whether a human-controlled seat was asked to act in a particular
            # night sub-phase reveals that seat's hidden role.  Unauthorized
            # viewers simply see an in-progress night until the action lands.
            visible_status = GameStatus.RUNNING
        return {
            "status": visible_status.value,
            "round": self.round,
            "phase": self.phase.public_name,
            "stage": stage,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "players": self._players_view(viewer_id=viewer_id, omniscient=omniscient),
            "pending_action": pending,
            "events": visible_events,
            "event_count": len(visible_events),
        }

    def result(self) -> dict[str, Any]:
        """Compact result payload for evaluators and tournament runners."""

        return {
            "status": self.status.value,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "rounds": self.round,
            "seed": self.config.seed,
            "players": self._players_view(omniscient=True),
            "event_count": len(self.events),
        }

    # ------------------------------------------------------------------
    # Phase handlers

    def _step_wolf(self) -> None:
        if not self._phase_queue:
            self._resolve_wolf_votes()
            return
        actor = self._phase_queue[0]
        targets = tuple(
            player.id
            for player in self.players.values()
            if player.alive and player.role is not Role.WEREWOLF
        )
        if not targets:
            self._phase_queue.clear()
            self._resolve_wolf_votes()
            return
        legal = (LegalAction(ActionType.WOLF_KILL, targets, description="Choose a night victim."),)
        self._request_or_act(actor, legal, "Choose the werewolves' night target.")

    def _step_seer(self) -> None:
        if not self._phase_queue:
            self._enter_phase(Phase.NIGHT_WITCH_SAVE)
            return
        actor = self._phase_queue[0]
        targets = tuple(
            player.id for player in self.players.values() if player.alive and player.id != actor
        )
        if not targets:
            self._phase_queue.pop(0)
            return
        legal = (LegalAction(ActionType.SEER_CHECK, targets, description="Inspect one player."),)
        self._request_or_act(actor, legal, "Choose a player to inspect.")

    def _step_witch_save(self) -> None:
        if not self._phase_queue:
            self._enter_phase(Phase.NIGHT_WITCH_POISON)
            return
        actor = self._phase_queue[0]
        targets: tuple[str, ...] = ()
        if self._night_victim is not None:
            if self.config.allow_witch_self_save or self._night_victim != actor:
                targets = (self._night_victim,)
        if not targets:
            self._phase_queue.pop(0)
            return
        legal = (
            LegalAction(
                ActionType.WITCH_SAVE,
                targets,
                can_pass=True,
                description="Use the antidote on tonight's victim, or pass.",
            ),
        )
        self._request_or_act(actor, legal, "Use the antidote or pass.")

    def _step_witch_poison(self) -> None:
        if not self._phase_queue:
            self._enter_phase(Phase.NIGHT_RESOLVE)
            return
        actor = self._phase_queue[0]
        targets = tuple(
            player.id for player in self.players.values() if player.alive and player.id != actor
        )
        if not targets:
            self._phase_queue.pop(0)
            return
        legal = (
            LegalAction(
                ActionType.WITCH_POISON,
                targets,
                can_pass=True,
                description="Use poison on one player, or pass.",
            ),
        )
        self._request_or_act(actor, legal, "Use poison or pass.")

    def _resolve_night(self) -> None:
        deaths: list[tuple[str, str]] = []
        if self._night_victim is not None and not self._night_saved:
            deaths.append((self._night_victim, "werewolf_attack"))
        deaths.extend((target, "witch_poison") for target in self._night_poison_targets)

        # Preserve seat order and collapse a player hit by multiple effects.
        reasons: dict[str, str] = {}
        for target, reason in deaths:
            reasons.setdefault(target, reason)
            if reason == "witch_poison":
                reasons[target] = reason
        killed: list[str] = []
        for player_id in self.players:
            if player_id in reasons and self._kill_player(player_id, reasons[player_id], public_cause="night"):
                killed.append(player_id)

        if killed:
            names = [self.players[player_id].name for player_id in killed]
            self._log(
                "night_result",
                "The night ended with casualties: " + ", ".join(names) + ".",
                data={"deaths": killed},
            )
        else:
            self._log("night_result", "The night ended peacefully; nobody died.", data={"deaths": []})

        self._hunter_queue = [
            player_id
            for player_id in killed
            if self.players[player_id].role is Role.HUNTER
            and (
                reasons[player_id] != "witch_poison"
                or self.config.hunter_can_shoot_if_poisoned
            )
        ]
        if self._hunter_queue:
            self._after_hunter_phase = Phase.DAY_DISCUSSION
            self._enter_phase(Phase.HUNTER_SHOT)
            return
        if self._check_victory():
            return
        self._enter_phase(Phase.DAY_DISCUSSION)

    def _step_discussion(self) -> None:
        if not self._phase_queue:
            if self._discussion_cycle < self.config.discussion_rounds:
                self._discussion_cycle += 1
                self._phase_queue = self._alive_ids()
                return
            self._enter_phase(Phase.DAY_VOTE)
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self._alive_ids() if pid != actor)
        legal = (
            LegalAction(
                ActionType.SPEAK,
                targets,
                content_required=True,
                description="Make one public statement.",
            ),
        )
        self._request_or_act(actor, legal, "Give your public statement.")

    def _step_vote(self) -> None:
        if not self._phase_queue:
            self._resolve_day_vote()
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self._alive_ids() if pid != actor)
        if not targets:
            self._phase_queue.pop(0)
            return
        legal = (LegalAction(ActionType.VOTE, targets, description="Vote to eliminate one player."),)
        self._request_or_act(actor, legal, "Cast your elimination vote.")

    def _step_hunter(self) -> None:
        if not self._phase_queue:
            self._finish_hunter_phase()
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self._alive_ids() if pid != actor)
        if not targets:
            self._phase_queue.pop(0)
            return
        legal = (
            LegalAction(
                ActionType.HUNTER_SHOOT,
                targets,
                can_pass=True,
                description="Shoot one living player, or pass.",
            ),
        )
        self._request_or_act(actor, legal, "Choose whether to fire the hunter's shot.")

    # ------------------------------------------------------------------
    # Decision handling

    def _request_or_act(
        self, actor_id: str, legal: Sequence[LegalAction], prompt: str
    ) -> None:
        player = self.players[actor_id]
        if player.controller == "human":
            self.pending_action = PendingAction(
                actor_id=actor_id,
                phase=self.phase,
                legal_actions=tuple(legal),
                prompt=prompt,
                round=self.round,
            )
            self.status = GameStatus.WAITING_FOR_HUMAN
            self._log(
                "action_required",
                prompt,
                actor_id=actor_id,
                data={"phase": self.phase.value},
                visibility=(actor_id,),
            )
            return

        observation = self.observe(actor_id, legal)
        try:
            action = self.agents[actor_id].decide(observation, self._agent_rngs[actor_id])
            if not isinstance(action, Action):
                action = ensure_action(action, default_actor=actor_id)  # type: ignore[arg-type]
            self._validate_action(action, legal, actor_id)
        except Exception as exc:
            self._log(
                "agent_action_repaired",
                "An invalid agent action was replaced with a safe legal action.",
                actor_id=actor_id,
                data={"error": f"{type(exc).__name__}: {exc}"},
                visibility=("observer",),
            )
            action = self._safe_fallback(actor_id, legal)
        self._consume_action(action)

    def _validate_action(
        self, action: Action, legal: Sequence[LegalAction], expected_actor: str
    ) -> None:
        if action.actor_id != expected_actor:
            raise InvalidAction(f"expected actor {expected_actor}, got {action.actor_id}")
        allowed = {item.action_type: item for item in legal}
        can_pass = any(item.can_pass for item in legal)
        if action.action_type is ActionType.PASS:
            if not can_pass:
                raise InvalidAction("passing is not legal in this phase")
            if action.target_id is not None:
                raise InvalidAction("a pass action cannot have a target")
            return
        if action.action_type not in allowed:
            raise InvalidAction(f"{action.action_type.value} is not legal in this phase")
        rule = allowed[action.action_type]
        if (
            rule.target_ids
            and action.action_type is not ActionType.SPEAK
            and action.target_id not in rule.target_ids
        ):
            raise InvalidAction("target is not in the legal target set")
        if (
            action.action_type is ActionType.SPEAK
            and action.target_id is not None
            and action.target_id not in rule.target_ids
        ):
            raise InvalidAction("target is not in the legal target set")
        if not rule.target_ids and action.target_id is not None:
            raise InvalidAction("this action does not accept a target")
        if rule.content_required and not (action.content and action.content.strip()):
            raise InvalidAction("this action requires non-empty content")

    def _safe_fallback(self, actor_id: str, legal: Sequence[LegalAction]) -> Action:
        rule = legal[0]
        if rule.action_type is ActionType.SPEAK:
            return Action(actor_id, ActionType.SPEAK, content="I am still evaluating the available evidence.")
        if rule.can_pass:
            return Action(actor_id, ActionType.PASS)
        target = (
            self._fallback_rngs[actor_id].choice(sorted(rule.target_ids))
            if rule.target_ids
            else None
        )
        return Action(actor_id, rule.action_type, target_id=target)

    def _consume_action(self, action: Action) -> None:
        if not self._phase_queue or self._phase_queue[0] != action.actor_id:
            raise RuntimeError("action does not match the active turn")
        self._phase_queue.pop(0)

        if self.phase is Phase.NIGHT_WOLF:
            assert action.target_id is not None
            self._wolf_votes[action.actor_id] = action.target_id
            wolves = tuple(self._alive_role_ids(Role.WEREWOLF))
            self._log(
                "wolf_vote",
                f"{self.players[action.actor_id].name} selected a night target.",
                actor_id=action.actor_id,
                target_id=action.target_id,
                data={"target_id": action.target_id},
                visibility=wolves,
            )
            if not self._phase_queue:
                self._resolve_wolf_votes()
            return

        if self.phase is Phase.NIGHT_SEER:
            assert action.target_id is not None
            result = (
                Role.WEREWOLF.value
                if self.players[action.target_id].role is Role.WEREWOLF
                else "not_werewolf"
            )
            self._seer_results[action.actor_id][action.target_id] = result
            self._log(
                "seer_check",
                f"Your inspection of {self.players[action.target_id].name} returned {result}.",
                actor_id=action.actor_id,
                target_id=action.target_id,
                data={"result": result},
                visibility=(action.actor_id,),
            )
            if not self._phase_queue:
                self._enter_phase(Phase.NIGHT_WITCH_SAVE)
            return

        if self.phase is Phase.NIGHT_WITCH_SAVE:
            if action.action_type is ActionType.WITCH_SAVE:
                self._night_saved = True
                self._witch_resources[action.actor_id]["antidote"] = False
                self._witch_used_this_night.add(action.actor_id)
                self._log(
                    "witch_save",
                    "You used the antidote.",
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                    visibility=(action.actor_id,),
                )
            else:
                self._log(
                    "witch_save_passed",
                    "You kept the antidote.",
                    actor_id=action.actor_id,
                    visibility=(action.actor_id,),
                )
            if not self._phase_queue:
                self._enter_phase(Phase.NIGHT_WITCH_POISON)
            return

        if self.phase is Phase.NIGHT_WITCH_POISON:
            if action.action_type is ActionType.WITCH_POISON:
                assert action.target_id is not None
                self._night_poison_targets.append(action.target_id)
                self._witch_resources[action.actor_id]["poison"] = False
                self._witch_used_this_night.add(action.actor_id)
                self._log(
                    "witch_poison",
                    f"You poisoned {self.players[action.target_id].name}.",
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                    visibility=(action.actor_id,),
                )
            else:
                self._log(
                    "witch_poison_passed",
                    "You kept the poison.",
                    actor_id=action.actor_id,
                    visibility=(action.actor_id,),
                )
            if not self._phase_queue:
                self._enter_phase(Phase.NIGHT_RESOLVE)
            return

        if self.phase is Phase.DAY_DISCUSSION:
            metadata = self._sanitize_speech_metadata(action.metadata)
            content = (action.content or "").strip()[:800]
            self._log(
                "speech",
                content,
                actor_id=action.actor_id,
                data=metadata,
            )
            if not self._phase_queue and self._discussion_cycle >= self.config.discussion_rounds:
                self._enter_phase(Phase.DAY_VOTE)
            return

        if self.phase is Phase.DAY_VOTE:
            assert action.target_id is not None
            self._day_votes[action.actor_id] = action.target_id
            self._log(
                "vote_cast",
                f"{self.players[action.actor_id].name} cast a vote.",
                actor_id=action.actor_id,
                target_id=action.target_id,
                data={"target_id": action.target_id},
                visibility=("observer",),
            )
            if not self._phase_queue:
                self._resolve_day_vote()
            return

        if self.phase is Phase.HUNTER_SHOT:
            if action.action_type is ActionType.HUNTER_SHOOT:
                assert action.target_id is not None
                target_name = self.players[action.target_id].name
                self._log(
                    "hunter_shot",
                    f"{self.players[action.actor_id].name} shot {target_name}.",
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                )
                if self._kill_player(action.target_id, "hunter_shot", public_cause="hunter_shot"):
                    target = self.players[action.target_id]
                    if target.role is Role.HUNTER and action.target_id not in self._phase_queue:
                        self._phase_queue.append(action.target_id)
            else:
                self._log(
                    "hunter_passed",
                    f"{self.players[action.actor_id].name} declined to shoot.",
                    actor_id=action.actor_id,
                )
            if not self._phase_queue:
                self._finish_hunter_phase()
            return

        raise RuntimeError(f"cannot consume action in phase {self.phase.value}")

    # ------------------------------------------------------------------
    # Resolution and transitions

    def _resolve_wolf_votes(self) -> None:
        if self._wolf_votes:
            tally = Counter(self._wolf_votes.values())
            highest = max(tally.values())
            tied = sorted(target for target, count in tally.items() if count == highest)
            self._night_victim = self.rng.choice(tied)
            wolves = tuple(self._alive_role_ids(Role.WEREWOLF))
            self._log(
                "wolf_target",
                f"The pack selected {self.players[self._night_victim].name}.",
                target_id=self._night_victim,
                data={"tally": dict(tally)},
                visibility=wolves,
            )
        self._enter_phase(Phase.NIGHT_SEER)

    def _resolve_day_vote(self) -> None:
        tally = Counter(self._day_votes.values())
        eliminated: Optional[str] = None
        tied: list[str] = []
        if tally:
            highest = max(tally.values())
            tied = sorted(target for target, count in tally.items() if count == highest)
            if len(tied) == 1:
                eliminated = tied[0]
            elif self.config.vote_tie_policy == "random":
                eliminated = self.rng.choice(tied)

        vote_data: dict[str, Any] = {"tally": dict(tally), "tied": tied, "eliminated": eliminated}
        if self.config.reveal_vote_choices:
            vote_data["votes"] = dict(self._day_votes)
        if eliminated is None:
            self._log("vote_result", "The vote was tied; nobody was eliminated.", data=vote_data)
            self._advance_after_day()
            return

        self._log(
            "vote_result",
            f"{self.players[eliminated].name} was selected for elimination.",
            target_id=eliminated,
            data=vote_data,
        )
        killed = self._kill_player(eliminated, "execution", public_cause="execution")
        if killed and self.players[eliminated].role is Role.HUNTER:
            self._hunter_queue = [eliminated]
            self._after_hunter_phase = Phase.NIGHT_WOLF
            self._enter_phase(Phase.HUNTER_SHOT)
            return
        self._advance_after_day()

    def _advance_after_day(self) -> None:
        if self._check_victory():
            return
        if self.round >= self.config.max_rounds:
            self._finish(Team.DRAW, f"maximum of {self.config.max_rounds} rounds reached")
            return
        self.round += 1
        self._enter_phase(Phase.NIGHT_WOLF)

    def _finish_hunter_phase(self) -> None:
        destination = self._after_hunter_phase
        self._after_hunter_phase = None
        if self._check_victory():
            return
        if destination is Phase.NIGHT_WOLF:
            if self.round >= self.config.max_rounds:
                self._finish(Team.DRAW, f"maximum of {self.config.max_rounds} rounds reached")
                return
            self.round += 1
            self._enter_phase(Phase.NIGHT_WOLF)
        else:
            self._enter_phase(destination or Phase.DAY_DISCUSSION)

    def _enter_phase(self, phase: Phase) -> None:
        if self.status is GameStatus.FINISHED:
            return
        self.phase = phase
        self.pending_action = None
        self.status = GameStatus.RUNNING

        if phase is Phase.NIGHT_WOLF:
            self._wolf_votes = {}
            self._day_votes = {}
            self._night_victim = None
            self._night_saved = False
            self._night_poison_targets = []
            self._witch_used_this_night = set()
            self._phase_queue = self._alive_role_ids(Role.WEREWOLF)
            self._log("night_started", f"Night {self.round} began.")
        elif phase is Phase.NIGHT_SEER:
            self._phase_queue = self._alive_role_ids(Role.SEER)
            self._log_secret_phase(phase)
        elif phase is Phase.NIGHT_WITCH_SAVE:
            self._phase_queue = [
                pid
                for pid in self._alive_role_ids(Role.WITCH)
                if self._witch_resources[pid]["antidote"] and self._night_victim is not None
            ]
            self._log_secret_phase(phase)
        elif phase is Phase.NIGHT_WITCH_POISON:
            self._phase_queue = [
                pid
                for pid in self._alive_role_ids(Role.WITCH)
                if self._witch_resources[pid]["poison"]
                and not (
                    self.config.witch_one_potion_per_night
                    and pid in self._witch_used_this_night
                )
            ]
            self._log_secret_phase(phase)
        elif phase is Phase.NIGHT_RESOLVE:
            self._phase_queue = []
            self._log_secret_phase(phase)
        elif phase is Phase.DAY_DISCUSSION:
            self._discussion_cycle = 1
            self._phase_queue = self._alive_ids()
            self._log("day_started", f"Day {self.round} discussion began.")
        elif phase is Phase.DAY_VOTE:
            self._day_votes = {}
            self._phase_queue = self._alive_ids()
            self._log("vote_started", f"Day {self.round} voting began.")
        elif phase is Phase.HUNTER_SHOT:
            self._phase_queue = list(self._hunter_queue)
            self._hunter_queue = []
            for hunter_id in self._phase_queue:
                self._log(
                    "hunter_triggered",
                    f"{self.players[hunter_id].name} may take a final shot.",
                    actor_id=hunter_id,
                )

    def _kill_player(self, player_id: str, reason: str, *, public_cause: str) -> bool:
        player = self.players[player_id]
        if not player.alive:
            return False
        player.alive = False
        player.death_reason = reason
        player.death_round = self.round
        role_data: dict[str, Any] = {}
        if self.config.reveal_roles_on_death:
            role_data["role"] = player.role.value
        if public_cause == "night":
            message = f"{player.name} died during the night."
        elif public_cause == "execution":
            message = f"{player.name} was eliminated by vote."
        else:
            message = f"{player.name} died from the hunter's shot."
        self._log(
            "death",
            message,
            target_id=player_id,
            data={"cause": public_cause, **role_data},
        )
        self._log(
            "death_detail",
            f"Referee cause of death: {reason}.",
            target_id=player_id,
            data={"cause": reason, "role": player.role.value},
            visibility=("observer",),
        )
        return True

    def _check_victory(self) -> bool:
        wolves = sum(
            player.alive and player.role is Role.WEREWOLF for player in self.players.values()
        )
        village = sum(
            player.alive and player.role is not Role.WEREWOLF for player in self.players.values()
        )
        if wolves == 0:
            self._finish(Team.VILLAGE, "all werewolves were eliminated")
            return True
        if wolves >= village:
            self._finish(Team.WEREWOLVES, "werewolves reached parity with the village")
            return True
        return False

    def _finish(self, winner: Team, reason: str) -> None:
        self.winner = winner.value
        self.win_reason = reason
        self.status = GameStatus.FINISHED
        self.phase = Phase.FINISHED
        self.pending_action = None
        self._phase_queue = []
        self._log(
            "game_finished",
            f"Game over: {winner.value} won ({reason}).",
            data={
                "winner": winner.value,
                "reason": reason,
                "roles": {pid: player.role.value for pid, player in self.players.items()},
            },
        )

    # ------------------------------------------------------------------
    # Views and utility helpers

    def _players_view(
        self, viewer_id: Optional[str] = None, *, omniscient: bool = False
    ) -> list[dict[str, Any]]:
        viewer = self.players.get(viewer_id) if viewer_id else None
        result: list[dict[str, Any]] = []
        for player in self.players.values():
            role_visible = (
                omniscient
                or self.status is GameStatus.FINISHED
                or player.id == viewer_id
                or (self.config.reveal_roles_on_death and not player.alive)
                or (
                    viewer is not None
                    and viewer.role is Role.WEREWOLF
                    and player.role is Role.WEREWOLF
                )
            )
            item: dict[str, Any] = {
                "id": player.id,
                "name": player.name,
                "alive": player.alive,
                "controller": player.controller,
                "role": player.role.value if role_visible else None,
                "team": player.team.value if role_visible else None,
                "death_round": player.death_round if not player.alive else None,
            }
            if omniscient:
                item["death_reason"] = player.death_reason
            result.append(item)
        return result

    def _resolve_player_references(self, references: Sequence[str | int]) -> set[str]:
        result: set[str] = set()
        names = {name: f"p{index}" for index, name in enumerate(self.config.player_names, 1)}
        count = len(self.config.player_names)
        for ref in references:
            if isinstance(ref, int):
                seat = 1 if ref == 0 else ref
                if not 1 <= seat <= count:
                    raise ValueError(f"human player seat out of range: {ref}")
                result.add(f"p{seat}")
                continue
            text = str(ref)
            if text in names:
                result.add(names[text])
            elif text.startswith("p") and text[1:].isdigit() and 1 <= int(text[1:]) <= count:
                result.add(text)
            elif text.isdigit() and 1 <= int(text) <= count:
                result.add(f"p{int(text)}")
            else:
                raise ValueError(f"unknown human player reference: {ref}")
        return result

    def _alive_ids(self) -> list[str]:
        return [player.id for player in self.players.values() if player.alive]

    def _alive_role_ids(self, role: Role) -> list[str]:
        return [
            player.id
            for player in self.players.values()
            if player.alive and player.role is role
        ]

    def _sanitize_speech_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        allowed: dict[str, Any] = {}
        for key in (
            "claim_role",
            "accuse_target",
            "defend_target",
            "vote_intent",
        ):
            value = metadata.get(key)
            if value is not None:
                allowed[key] = str(value)[:80]
        return allowed

    def _log_secret_phase(self, phase: Phase) -> None:
        self._log(
            "phase_started",
            f"Referee entered {phase.value}.",
            data={"phase": phase.value},
            visibility=("observer",),
        )

    def _log(
        self,
        event_type: str,
        message: str,
        *,
        actor_id: Optional[str] = None,
        target_id: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None,
        visibility: Sequence[str] = ("public",),
    ) -> GameEvent:
        event = GameEvent(
            seq=len(self.events) + 1,
            round=self.round,
            phase=self.phase.value,
            event_type=event_type,
            actor_id=actor_id,
            target_id=target_id,
            message=message,
            data=dict(data or {}),
            visibility=tuple(visibility),
        )
        self.events.append(event)
        return event


# Concise aliases useful to callers that prefer a generic engine name.
GameEngine = WerewolfGame
