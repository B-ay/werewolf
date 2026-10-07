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

        # Sheriff is a public badge layered on top of a player's secret role.
        # Election choices are deliberately buffered without events until each
        # batch closes, so a later actor cannot condition their choice on an
        # earlier actor's sealed decision.
        self._sheriff_badge_status = "unassigned" if config.enable_sheriff else "void"
        self._sheriff_holder: Optional[str] = None
        self._sheriff_candidates: list[str] = []
        self._sheriff_original_candidates: list[str] = []
        self._sheriff_withdrawn: list[str] = []
        self._sheriff_pk_candidates: list[str] = []
        self._sheriff_speech_order: list[str] = []
        self._sheriff_voters: list[str] = []
        self._sheriff_vote_round = 0
        self._sheriff_sealed_choices: dict[str, Action] = {}
        self._sheriff_day_direction: Optional[str] = None
        self._day_speech_order: list[str] = []
        self._sheriff_nomination_target: Optional[str] = None
        self._sheriff_election_complete = not config.enable_sheriff
        self._badge_owner_awaiting_action: Optional[str] = None

        self._pending_dawn_deaths: dict[str, str] = {}
        self._day_vote_round = 0
        self._day_pk_candidates: list[str] = []
        # Destination after badge/hunter death abilities finish.  Keeping one
        # continuation makes badge-before-gun ordering identical for every
        # source of death, including a hunter shooting the new sheriff.
        self._after_hunter_phase = None

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
            Phase.SHERIFF_RUN: self._step_sheriff_run,
            Phase.SHERIFF_SPEECH: self._step_sheriff_speech,
            Phase.SHERIFF_WITHDRAW: self._step_sheriff_withdraw,
            Phase.SHERIFF_VOTE: self._step_sheriff_vote,
            Phase.SHERIFF_PK_SPEECH: self._step_sheriff_pk_speech,
            Phase.SHERIFF_PK_VOTE: self._step_sheriff_vote,
            Phase.DAWN_REVEAL: self._resolve_dawn,
            Phase.SHERIFF_BADGE: self._step_sheriff_badge,
            Phase.SHERIFF_ORDER: self._step_sheriff_order,
            Phase.DAY_DISCUSSION: self._step_discussion,
            Phase.DAY_VOTE: self._step_vote,
            Phase.DAY_PK_SPEECH: self._step_day_pk_speech,
            Phase.DAY_PK_VOTE: self._step_vote,
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

        self_state = player.to_private_dict()
        self_state["is_sheriff"] = player.id == self._sheriff_holder
        private_facts["sheriff"] = self._sheriff_public_state()
        return Observation(
            viewer_id=viewer_id,
            round=self.round,
            phase=self.phase,
            self_state=self_state,
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
            Phase.SHERIFF_RUN,
            Phase.SHERIFF_SPEECH,
            Phase.SHERIFF_WITHDRAW,
            Phase.SHERIFF_VOTE,
            Phase.SHERIFF_PK_SPEECH,
            Phase.SHERIFF_PK_VOTE,
            Phase.DAWN_REVEAL,
            Phase.SHERIFF_BADGE,
            Phase.SHERIFF_ORDER,
            Phase.DAY_DISCUSSION,
            Phase.DAY_VOTE,
            Phase.DAY_PK_SPEECH,
            Phase.DAY_PK_VOTE,
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
                    Phase.SHERIFF_RUN,
                    Phase.SHERIFF_SPEECH,
                    Phase.SHERIFF_WITHDRAW,
                    Phase.SHERIFF_VOTE,
                    Phase.SHERIFF_PK_SPEECH,
                    Phase.SHERIFF_PK_VOTE,
                    Phase.SHERIFF_BADGE,
                    Phase.SHERIFF_ORDER,
                    Phase.DAY_DISCUSSION,
                    Phase.DAY_VOTE,
                    Phase.DAY_PK_SPEECH,
                    Phase.DAY_PK_VOTE,
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
            "sheriff": self._sheriff_public_state(),
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
            "sheriff": self._sheriff_public_state(),
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
        self._pending_dawn_deaths = reasons
        if (
            self.round == 1
            and self.config.enable_sheriff
            and not self._sheriff_election_complete
        ):
            self._enter_phase(Phase.SHERIFF_RUN)
        else:
            self._enter_phase(Phase.DAWN_REVEAL)

    def _resolve_dawn(self) -> None:
        reasons = dict(self._pending_dawn_deaths)
        self._pending_dawn_deaths = {}
        killed: list[str] = []
        for player_id in self.players:
            if player_id in reasons and self._kill_player(
                player_id, reasons[player_id], public_cause="night"
            ):
                killed.append(player_id)

        if killed:
            names = [self.players[player_id].name for player_id in killed]
            self._log(
                "night_result",
                "The night ended with casualties: " + ", ".join(names) + ".",
                data={"deaths": killed},
            )
        else:
            self._log(
                "night_result",
                "The night ended peacefully; nobody died.",
                data={"deaths": []},
            )

        self._hunter_queue = [
            player_id
            for player_id in killed
            if self.players[player_id].role is Role.HUNTER
            and (
                reasons[player_id] != "witch_poison"
                or self.config.hunter_can_shoot_if_poisoned
            )
        ]
        self._after_hunter_phase = Phase.SHERIFF_ORDER
        dead_sheriff = next(
            (pid for pid in killed if pid == self._sheriff_holder), None
        )
        if dead_sheriff is not None:
            self._begin_badge_transfer(dead_sheriff)
            return
        if self._hunter_queue:
            self._enter_phase(Phase.HUNTER_SHOT)
            return
        self._complete_death_sequence()

    def _step_sheriff_run(self) -> None:
        if not self._phase_queue:
            self._resolve_sheriff_run()
            return
        actor = self._phase_queue[0]
        legal = (
            LegalAction(
                ActionType.SHERIFF_RUN,
                can_pass=True,
                description="Run for sheriff, or remain off the ballot.",
            ),
        )
        self._request_or_act(actor, legal, "Choose whether to run for sheriff.")

    def _step_sheriff_speech(self) -> None:
        if not self._phase_queue:
            self._enter_phase(Phase.SHERIFF_WITHDRAW)
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self.players if pid != actor)
        legal = (
            LegalAction(
                ActionType.SPEAK,
                targets,
                content_required=True,
                description="Give a public sheriff campaign speech.",
            ),
        )
        self._request_or_act(actor, legal, "Give your sheriff campaign speech.")

    def _step_sheriff_withdraw(self) -> None:
        if not self._phase_queue:
            self._resolve_sheriff_withdrawals()
            return
        actor = self._phase_queue[0]
        legal = (
            LegalAction(
                ActionType.SHERIFF_WITHDRAW,
                can_pass=True,
                description="Withdraw from the sheriff election, or stay in.",
            ),
        )
        self._request_or_act(actor, legal, "Choose whether to withdraw.")

    def _step_sheriff_vote(self) -> None:
        if not self._phase_queue:
            self._resolve_sheriff_vote()
            return
        actor = self._phase_queue[0]
        targets = tuple(self._sheriff_pk_candidates or self._sheriff_candidates)
        legal = (
            LegalAction(
                ActionType.SHERIFF_VOTE,
                targets,
                can_pass=True,
                description="Vote for sheriff, or abstain.",
            ),
        )
        self._request_or_act(actor, legal, "Cast your sheriff ballot or abstain.")

    def _step_sheriff_pk_speech(self) -> None:
        if not self._phase_queue:
            self._enter_phase(Phase.SHERIFF_PK_VOTE)
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self.players if pid != actor)
        legal = (
            LegalAction(
                ActionType.SPEAK,
                targets,
                content_required=True,
                description="Give a sheriff runoff speech.",
            ),
        )
        self._request_or_act(actor, legal, "Give your sheriff runoff speech.")

    def _step_sheriff_badge(self) -> None:
        if not self._phase_queue:
            self._finish_badge_transfer()
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self._alive_ids() if pid != actor)
        legal_actions: list[LegalAction] = []
        if targets:
            legal_actions.append(
                LegalAction(
                    ActionType.SHERIFF_TRANSFER,
                    targets,
                    description="Transfer the sheriff badge to a living player.",
                )
            )
        legal_actions.append(
            LegalAction(
                ActionType.SHERIFF_DESTROY,
                description="Destroy the sheriff badge permanently.",
            )
        )
        self._request_or_act(actor, tuple(legal_actions), "Transfer or destroy the sheriff badge.")

    def _step_sheriff_order(self) -> None:
        if not self._phase_queue:
            self._enter_phase(Phase.DAY_DISCUSSION)
            return
        actor = self._phase_queue[0]
        legal = (
            LegalAction(
                ActionType.SHERIFF_ORDER_CLOCKWISE,
                description="Use clockwise speaking order.",
            ),
            LegalAction(
                ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE,
                description="Use counterclockwise speaking order.",
            ),
        )
        self._request_or_act(actor, legal, "Choose today's speaking direction.")

    def _step_discussion(self) -> None:
        if not self._phase_queue:
            if self._discussion_cycle < self.config.discussion_rounds:
                self._discussion_cycle += 1
                self._phase_queue = [
                    pid for pid in self._day_speech_order if self.players[pid].alive
                ]
                return
            self._enter_phase(Phase.DAY_VOTE)
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self._alive_ids() if pid != actor)
        if actor == self._sheriff_holder:
            legal = (
                LegalAction(
                    ActionType.SHERIFF_SUMMARIZE,
                    targets,
                    content_required=True,
                    description="Summarize the discussion and nominate an elimination target.",
                ),
            )
            self._request_or_act(actor, legal, "Summarize and nominate today's target.")
            return
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
        if self.phase is Phase.DAY_PK_VOTE:
            targets = tuple(self._day_pk_candidates)
        else:
            targets = tuple(pid for pid in self._alive_ids() if pid != actor)
        if not targets:
            self._phase_queue.pop(0)
            return
        legal = (LegalAction(ActionType.VOTE, targets, description="Vote to eliminate one player."),)
        self._request_or_act(actor, legal, "Cast your elimination vote.")

    def _step_day_pk_speech(self) -> None:
        if not self._phase_queue:
            self._enter_phase(Phase.DAY_PK_VOTE)
            return
        actor = self._phase_queue[0]
        targets = tuple(pid for pid in self._alive_ids() if pid != actor)
        legal = (
            LegalAction(
                ActionType.SPEAK,
                targets,
                content_required=True,
                description="Give a public elimination runoff speech.",
            ),
        )
        self._request_or_act(actor, legal, "Give your elimination runoff speech.")

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
        content = (
            "I am summarizing the public evidence and nominating this target."
            if rule.content_required
            else None
        )
        return Action(actor_id, rule.action_type, target_id=target, content=content)

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

        if self.phase in (Phase.SHERIFF_RUN, Phase.SHERIFF_WITHDRAW):
            if self.phase is Phase.SHERIFF_RUN:
                self._log(
                    "sheriff_run_choice",
                    "A sheriff candidacy choice was sealed.",
                    actor_id=action.actor_id,
                    data={"running": action.action_type is ActionType.SHERIFF_RUN},
                    visibility=("observer",),
                )
            else:
                self._log(
                    "sheriff_withdraw_choice",
                    "A sheriff withdrawal choice was sealed.",
                    actor_id=action.actor_id,
                    data={"withdrew": action.action_type is ActionType.SHERIFF_WITHDRAW},
                    visibility=("observer",),
                )
            self._sheriff_sealed_choices[action.actor_id] = action
            if not self._phase_queue:
                if self.phase is Phase.SHERIFF_RUN:
                    self._resolve_sheriff_run()
                else:
                    self._resolve_sheriff_withdrawals()
            return

        if self.phase in (Phase.SHERIFF_VOTE, Phase.SHERIFF_PK_VOTE):
            self._log(
                "sheriff_vote_cast",
                "A sheriff ballot was sealed.",
                actor_id=action.actor_id,
                target_id=action.target_id,
                data={
                    "round": self._sheriff_vote_round,
                    "abstained": action.action_type is ActionType.PASS,
                },
                visibility=("observer",),
            )
            self._sheriff_sealed_choices[action.actor_id] = action
            if not self._phase_queue:
                self._resolve_sheriff_vote()
            return

        if self.phase in (Phase.SHERIFF_SPEECH, Phase.SHERIFF_PK_SPEECH):
            metadata = self._sanitize_speech_metadata(action.metadata)
            content = (action.content or "").strip()[:800]
            event_type = (
                "sheriff_pk_speech"
                if self.phase is Phase.SHERIFF_PK_SPEECH
                else "sheriff_speech"
            )
            self._log(event_type, content, actor_id=action.actor_id, data=metadata)
            if not self._phase_queue:
                next_phase = (
                    Phase.SHERIFF_PK_VOTE
                    if self.phase is Phase.SHERIFF_PK_SPEECH
                    else Phase.SHERIFF_WITHDRAW
                )
                self._enter_phase(next_phase)
            return

        if self.phase is Phase.SHERIFF_BADGE:
            previous = self._badge_owner_awaiting_action or action.actor_id
            if action.action_type is ActionType.SHERIFF_TRANSFER:
                assert action.target_id is not None
                self._sheriff_holder = action.target_id
                self._sheriff_badge_status = "holder"
                self._log(
                    "badge_transferred",
                    f"{self.players[previous].name} transferred the sheriff badge to "
                    f"{self.players[action.target_id].name}.",
                    actor_id=previous,
                    target_id=action.target_id,
                )
            else:
                self._sheriff_holder = None
                self._sheriff_badge_status = "destroyed"
                self._log(
                    "badge_destroyed",
                    f"{self.players[previous].name} destroyed the sheriff badge.",
                    actor_id=previous,
                )
            self._badge_owner_awaiting_action = None
            self._finish_badge_transfer()
            return

        if self.phase is Phase.SHERIFF_ORDER:
            direction = (
                "clockwise"
                if action.action_type is ActionType.SHERIFF_ORDER_CLOCKWISE
                else "counterclockwise"
            )
            self._sheriff_day_direction = direction
            self._day_speech_order = self._speaking_order(action.actor_id, direction)
            self._log(
                "sheriff_order",
                f"{self.players[action.actor_id].name} chose {direction} speaking order.",
                actor_id=action.actor_id,
                data={"direction": direction, "order": list(self._day_speech_order)},
            )
            self._enter_phase(Phase.DAY_DISCUSSION)
            return

        if self.phase is Phase.DAY_DISCUSSION:
            metadata = self._sanitize_speech_metadata(action.metadata)
            content = (action.content or "").strip()[:800]
            if action.action_type is ActionType.SHERIFF_SUMMARIZE:
                assert action.target_id is not None
                self._sheriff_nomination_target = action.target_id
                metadata["sheriff_nomination"] = action.target_id
                self._log(
                    "sheriff_summary",
                    content,
                    actor_id=action.actor_id,
                    target_id=action.target_id,
                    data=metadata,
                )
            else:
                self._log("speech", content, actor_id=action.actor_id, data=metadata)
            if not self._phase_queue and self._discussion_cycle >= self.config.discussion_rounds:
                self._enter_phase(Phase.DAY_VOTE)
            return

        if self.phase is Phase.DAY_PK_SPEECH:
            metadata = self._sanitize_speech_metadata(action.metadata)
            content = (action.content or "").strip()[:800]
            self._log(
                "day_pk_speech",
                content,
                actor_id=action.actor_id,
                data=metadata,
            )
            if not self._phase_queue:
                self._enter_phase(Phase.DAY_PK_VOTE)
            return

        if self.phase in (Phase.DAY_VOTE, Phase.DAY_PK_VOTE):
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
                    if action.target_id == self._sheriff_holder:
                        self._hunter_queue = list(self._phase_queue)
                        self._begin_badge_transfer(action.target_id)
                        return
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

    def _resolve_sheriff_run(self) -> None:
        candidates = [
            pid
            for pid in self.players
            if (choice := self._sheriff_sealed_choices.get(pid)) is not None
            and choice.action_type is ActionType.SHERIFF_RUN
        ]
        self._sheriff_sealed_choices = {}
        self._sheriff_original_candidates = list(candidates)
        self._sheriff_candidates = list(candidates)
        self._sheriff_pk_candidates = []
        self._sheriff_voters = [pid for pid in self.players if pid not in candidates]
        if candidates:
            start = self.rng.randrange(len(candidates))
            self._sheriff_speech_order = candidates[start:] + candidates[:start]
        else:
            self._sheriff_speech_order = []
        self._log(
            "sheriff_candidacy_result",
            "Sheriff candidates were announced.",
            data={
                "candidates": list(candidates),
                "speech_order": list(self._sheriff_speech_order),
            },
        )
        if not candidates:
            self._void_sheriff_election("no_candidates", "Nobody ran for sheriff.")
            return
        self._enter_phase(Phase.SHERIFF_SPEECH)

    def _resolve_sheriff_withdrawals(self) -> None:
        withdrawn = [
            pid
            for pid in self._sheriff_original_candidates
            if (choice := self._sheriff_sealed_choices.get(pid)) is not None
            and choice.action_type is ActionType.SHERIFF_WITHDRAW
        ]
        self._sheriff_sealed_choices = {}
        self._sheriff_withdrawn = withdrawn
        self._sheriff_candidates = [
            pid for pid in self._sheriff_original_candidates if pid not in withdrawn
        ]
        self._log(
            "sheriff_withdrawal_result",
            "Sheriff withdrawals were announced.",
            data={
                "withdrawn": list(withdrawn),
                "candidates": list(self._sheriff_candidates),
            },
        )
        if not self._sheriff_candidates:
            self._void_sheriff_election("all_withdrew", "All candidates withdrew.")
        elif len(self._sheriff_candidates) == 1:
            self._elect_sheriff(self._sheriff_candidates[0], automatic=True)
        else:
            self._sheriff_vote_round = 1
            self._enter_phase(Phase.SHERIFF_VOTE)

    def _resolve_sheriff_vote(self) -> None:
        candidates = list(self._sheriff_pk_candidates or self._sheriff_candidates)
        ballots = {
            pid: (
                choice.target_id
                if choice.action_type is ActionType.SHERIFF_VOTE
                and choice.target_id in candidates
                else None
            )
            for pid, choice in self._sheriff_sealed_choices.items()
        }
        votes = {
            pid: target_id
            for pid, target_id in ballots.items()
            if target_id is not None
        }
        self._sheriff_sealed_choices = {}
        tally = Counter(target for target in votes.values() if target is not None)
        highest = max(tally.values()) if tally else 0
        tied = [pid for pid in candidates if tally.get(pid, 0) == highest] if highest else []
        data: dict[str, Any] = {
            "round": self._sheriff_vote_round,
            "tally": dict(tally),
            "tied": tied,
        }
        if self.config.reveal_vote_choices:
            # Preserve explicit abstentions so the revealed batch is a complete
            # ballot record rather than only a list of valid votes.
            data["votes"] = ballots
        self._log(
            "sheriff_vote_result",
            "Sheriff ballots were counted.",
            data=data,
        )
        if not tally:
            self._void_sheriff_election(
                "no_valid_ballots", "No valid sheriff ballots were cast."
            )
            return
        if len(tied) == 1:
            self._elect_sheriff(tied[0])
            return
        if self._sheriff_vote_round >= 2:
            self._void_sheriff_election(
                "runoff_tied", "The sheriff runoff remained tied."
            )
            return
        self._sheriff_vote_round = 2
        speech_positions = {pid: index for index, pid in enumerate(self._sheriff_speech_order)}
        self._sheriff_pk_candidates = sorted(
            tied,
            key=lambda pid: speech_positions.get(pid, -1),
            reverse=True,
        )
        self._log(
            "sheriff_pk_started",
            "A sheriff runoff began.",
            data={"candidates": list(self._sheriff_pk_candidates)},
        )
        self._enter_phase(Phase.SHERIFF_PK_SPEECH)

    def _elect_sheriff(self, player_id: str, *, automatic: bool = False) -> None:
        self._sheriff_holder = player_id
        self._sheriff_badge_status = "holder"
        self._sheriff_election_complete = True
        self._sheriff_pk_candidates = []
        self._log(
            "sheriff_elected",
            f"{self.players[player_id].name} became sheriff.",
            actor_id=player_id,
            data={"holder_id": player_id, "automatic": automatic},
        )
        self._enter_phase(Phase.DAWN_REVEAL)

    def _void_sheriff_election(self, reason: str, message: str) -> None:
        self._sheriff_holder = None
        self._sheriff_badge_status = "void"
        self._sheriff_election_complete = True
        self._sheriff_pk_candidates = []
        self._log(
            "sheriff_election_void",
            message,
            data={"reason": message, "reason_code": reason},
        )
        self._enter_phase(Phase.DAWN_REVEAL)

    def _resolve_day_vote(self) -> None:
        tally_units: Counter[str] = Counter()
        unweighted_tally_units: Counter[str] = Counter()
        for voter_id, target_id in self._day_votes.items():
            weight = 3 if voter_id == self._sheriff_holder else 2
            tally_units[target_id] += weight
            unweighted_tally_units[target_id] += 2
        eliminated: Optional[str] = None
        tied: list[str] = []
        if tally_units:
            highest = max(tally_units.values())
            tied = sorted(target for target, count in tally_units.items() if count == highest)
            if len(tied) == 1:
                eliminated = tied[0]
            elif not self.config.enable_sheriff and self.config.vote_tie_policy == "random":
                eliminated = self.rng.choice(tied)

        unweighted_winners: list[str] = []
        if unweighted_tally_units:
            unweighted_highest = max(unweighted_tally_units.values())
            unweighted_winners = sorted(
                pid
                for pid, units in unweighted_tally_units.items()
                if units == unweighted_highest
            )
        weighted_unique = tied[0] if len(tied) == 1 else None
        unweighted_unique = (
            unweighted_winners[0] if len(unweighted_winners) == 1 else None
        )
        sheriff_vote_decisive = bool(
            self._sheriff_holder
            and self._sheriff_holder in self._day_votes
            and weighted_unique is not None
            and weighted_unique != unweighted_unique
        )
        vote_data: dict[str, Any] = {
            "round": self._day_vote_round,
            "tally_units": dict(tally_units),
            "unweighted_tally_units": dict(unweighted_tally_units),
            "tally": {pid: units / 2 for pid, units in tally_units.items()},
            "tied": tied,
            "eliminated": eliminated,
            "sheriff_id": self._sheriff_holder,
            "sheriff_vote_decisive": sheriff_vote_decisive,
        }
        if self.config.reveal_vote_choices:
            vote_data["votes"] = dict(self._day_votes)
            vote_data["vote_weights"] = {
                pid: 1.5 if pid == self._sheriff_holder else 1.0
                for pid in self._day_votes
            }
        if eliminated is None:
            self._log(
                "day_vote_result",
                "The elimination vote did not produce a sole leader.",
                data=vote_data,
            )
            if (
                self.config.enable_sheriff
                and self._day_vote_round == 1
                and len(tied) > 1
            ):
                positions = {pid: idx for idx, pid in enumerate(self._day_speech_order)}
                self._day_pk_candidates = sorted(
                    tied,
                    key=lambda pid: positions.get(pid, -1),
                    reverse=True,
                )
                self._log(
                    "day_pk_started",
                    "An elimination runoff began.",
                    data={"candidates": list(self._day_pk_candidates)},
                )
                self._enter_phase(Phase.DAY_PK_SPEECH)
                return
            self._advance_after_day()
            return

        self._log(
            "day_vote_result",
            f"{self.players[eliminated].name} was selected for elimination.",
            target_id=eliminated,
            data=vote_data,
        )
        killed = self._kill_player(eliminated, "execution", public_cause="execution")
        if not killed:
            self._advance_after_day()
            return
        self._hunter_queue = (
            [eliminated] if self.players[eliminated].role is Role.HUNTER else []
        )
        self._after_hunter_phase = Phase.NIGHT_WOLF
        if eliminated == self._sheriff_holder:
            self._begin_badge_transfer(eliminated)
        elif self._hunter_queue:
            self._enter_phase(Phase.HUNTER_SHOT)
        else:
            self._complete_death_sequence()

    def _advance_after_day(self) -> None:
        if self._check_victory():
            return
        if self.round >= self.config.max_rounds:
            self._finish(Team.DRAW, f"maximum of {self.config.max_rounds} rounds reached")
            return
        self.round += 1
        self._enter_phase(Phase.NIGHT_WOLF)

    def _finish_hunter_phase(self) -> None:
        self._complete_death_sequence()

    def _complete_death_sequence(self) -> None:
        destination = self._after_hunter_phase or Phase.SHERIFF_ORDER
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
            if destination is Phase.SHERIFF_ORDER:
                self._start_day()
            else:
                self._enter_phase(destination)

    def _begin_badge_transfer(self, dead_holder: str) -> None:
        self._badge_owner_awaiting_action = dead_holder
        self._log(
            "badge_transfer_required",
            f"{self.players[dead_holder].name} must transfer or destroy the sheriff badge.",
            actor_id=dead_holder,
        )
        self._enter_phase(Phase.SHERIFF_BADGE)

    def _finish_badge_transfer(self) -> None:
        if self._hunter_queue:
            self._enter_phase(Phase.HUNTER_SHOT)
        else:
            self._complete_death_sequence()

    def _start_day(self) -> None:
        # Direction and speaking order are daily decisions.  Clear yesterday's
        # public values before the sheriff is prompted so spectators and later
        # agents cannot mistake stale order data for today's choice.
        self._sheriff_day_direction = None
        self._day_speech_order = []
        if (
            self.config.enable_sheriff
            and self._sheriff_badge_status == "holder"
            and self._sheriff_holder is not None
            and self.players[self._sheriff_holder].alive
        ):
            self._enter_phase(Phase.SHERIFF_ORDER)
        else:
            self._day_speech_order = self._alive_ids()
            self._enter_phase(Phase.DAY_DISCUSSION)

    def _speaking_order(self, sheriff_id: str, direction: str) -> list[str]:
        seats = list(self.players)
        anchor = seats.index(sheriff_id)
        if direction == "clockwise":
            traversal = seats[anchor + 1 :] + seats[:anchor]
        else:
            traversal = list(reversed(seats[:anchor])) + list(reversed(seats[anchor + 1 :]))
        return [pid for pid in traversal if self.players[pid].alive] + [sheriff_id]

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
            self._day_pk_candidates = []
            self._sheriff_nomination_target = None
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
        elif phase is Phase.SHERIFF_RUN:
            self._sheriff_sealed_choices = {}
            self._phase_queue = self._alive_ids()
            self._log(
                "sheriff_election_started",
                "The first-day sheriff election began before dawn was announced.",
            )
        elif phase is Phase.SHERIFF_SPEECH:
            self._phase_queue = list(self._sheriff_speech_order)
        elif phase is Phase.SHERIFF_WITHDRAW:
            self._sheriff_sealed_choices = {}
            self._phase_queue = list(self._sheriff_original_candidates)
        elif phase in (Phase.SHERIFF_VOTE, Phase.SHERIFF_PK_VOTE):
            self._sheriff_sealed_choices = {}
            self._phase_queue = [
                pid for pid in self._sheriff_voters if self.players[pid].alive
            ]
            self._log(
                "sheriff_vote_started",
                "Sheriff voting began.",
                data={
                    "round": self._sheriff_vote_round,
                    "candidates": list(
                        self._sheriff_pk_candidates or self._sheriff_candidates
                    ),
                    "voters": list(self._phase_queue),
                },
            )
        elif phase is Phase.SHERIFF_PK_SPEECH:
            self._phase_queue = list(self._sheriff_pk_candidates)
        elif phase is Phase.DAWN_REVEAL:
            self._phase_queue = []
            self._log("dawn_reveal_started", f"Dawn {self.round} results will be announced.")
        elif phase is Phase.SHERIFF_BADGE:
            self._phase_queue = (
                [self._badge_owner_awaiting_action]
                if self._badge_owner_awaiting_action is not None
                else []
            )
        elif phase is Phase.SHERIFF_ORDER:
            holder = self._sheriff_holder
            self._phase_queue = (
                [holder]
                if holder is not None and self.players[holder].alive
                else []
            )
        elif phase is Phase.DAY_DISCUSSION:
            self._discussion_cycle = 1
            if not self._day_speech_order:
                self._day_speech_order = self._alive_ids()
            self._phase_queue = [
                pid for pid in self._day_speech_order if self.players[pid].alive
            ]
            self._log("day_started", f"Day {self.round} discussion began.")
        elif phase is Phase.DAY_VOTE:
            self._day_votes = {}
            self._day_vote_round = 1
            self._day_pk_candidates = []
            self._phase_queue = self._alive_ids()
            self._log("vote_started", f"Day {self.round} voting began.")
        elif phase is Phase.DAY_PK_SPEECH:
            self._phase_queue = list(self._day_pk_candidates)
        elif phase is Phase.DAY_PK_VOTE:
            self._day_votes = {}
            self._day_vote_round = 2
            self._phase_queue = [
                pid for pid in self._alive_ids() if pid not in self._day_pk_candidates
            ]
            self._log(
                "vote_started",
                f"Day {self.round} runoff voting began.",
                data={"round": 2, "candidates": list(self._day_pk_candidates)},
            )
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
                "is_sheriff": player.id == self._sheriff_holder,
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
        claim_role = metadata.get("claim_role")
        if claim_role is not None:
            allowed["claim_role"] = str(claim_role)[:80]
        # These fields are machine-readable player references.  Invalid ids
        # are dropped rather than being allowed to create phantom seats in the
        # UI, replay, or evaluation pipeline.
        for key in ("accuse_target", "defend_target", "vote_intent"):
            value = metadata.get(key)
            if value is not None and str(value) in self.players:
                allowed[key] = str(value)
        nomination = metadata.get("sheriff_nomination")
        if nomination is not None and str(nomination) in self.players:
            allowed["sheriff_nomination"] = str(nomination)
        badge_flow = metadata.get("badge_flow")
        if isinstance(badge_flow, Mapping):
            mode = str(badge_flow.get("mode", ""))
            raw_targets = badge_flow.get("targets", ())
            if isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes)):
                expected = 1 if mode == "single" else 2 if mode == "double" else 0
                targets = [str(pid) for pid in raw_targets]
                if (
                    expected
                    and len(targets) == expected
                    and len(set(targets)) == len(targets)
                    and all(pid in self.players for pid in targets)
                ):
                    sanitized_flow: dict[str, Any] = {"mode": mode, "targets": targets}
                    fallback = badge_flow.get("fallback_target")
                    if fallback is not None and str(fallback) in self.players:
                        sanitized_flow["fallback_target"] = str(fallback)
                    allowed["badge_flow"] = sanitized_flow
        return allowed

    def _sheriff_public_state(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enable_sheriff,
            "badge_status": self._sheriff_badge_status,
            "holder_id": self._sheriff_holder,
            "candidates": list(self._sheriff_candidates),
            "withdrawn": list(self._sheriff_withdrawn),
            # This object describes the sheriff election.  Daytime exile-PK
            # candidates are published by ``day_pk_started`` and must not
            # overwrite or resurrect an already-finished election runoff.
            "pk_candidates": list(self._sheriff_pk_candidates),
            "vote_round": self._sheriff_vote_round,
            "speech_order": list(self._sheriff_speech_order),
            "day_direction": self._sheriff_day_direction,
            "day_speech_order": list(self._day_speech_order),
            "nomination_target": self._sheriff_nomination_target,
        }

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
