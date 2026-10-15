"""Architecture contracts for the five independent role agents.

These tests deliberately exercise the public agent boundary rather than
implementation details.  A role policy may change its heuristics without
breaking the guarantees that each seat owns its goal, long-term memory,
prompt, and planner, and that no malformed model output reaches the engine.
"""

from __future__ import annotations

import json
import random
import unittest
from typing import Any, Mapping, Sequence

from backend.game.models import GameConfig, Role
from backend.game.engine import WerewolfGame
from backend.agents.hunter import HunterAgent
from backend.agents.memory import RoleMemory
from backend.agents.seer import SeerAgent
from backend.agents.villager import VillagerAgent
from backend.agents.werewolf import WerewolfAgent
from backend.agents.witch import WitchAgent
from backend.agents.factory import create_agent
from backend.game.models import ActionType, Observation, Phase


ROLE_AGENT_TYPES = {
    Role.WEREWOLF: WerewolfAgent,
    Role.SEER: SeerAgent,
    Role.WITCH: WitchAgent,
    Role.HUNTER: HunterAgent,
    Role.VILLAGER: VillagerAgent,
}


def make_observation(
    role: Role,
    action_type: ActionType,
    *,
    viewer_id: str = "p1",
    round_number: int = 1,
    phase: Phase = Phase.DAY_VOTE,
    targets: Sequence[str] = ("p2", "p3"),
    can_pass: bool = False,
    content_required: bool = False,
    events: Sequence[Mapping[str, Any]] = (),
    private_facts: Mapping[str, Any] | None = None,
    players: Sequence[Mapping[str, Any]] | None = None,
) -> Observation:
    """Build the same sanitized value object that the engine gives a policy."""

    visible_players = [
        dict(player)
        for player in (
            players
            or (
                {"id": "p1", "name": "Agent One", "alive": True, "role": None},
                {"id": "p2", "name": "Agent Two", "alive": True, "role": None},
                {"id": "p3", "name": "Agent Three", "alive": True, "role": None},
            )
        )
    ]
    for player in visible_players:
        if str(player.get("id")) == viewer_id:
            player["role"] = role.value
            viewer_name = str(player.get("name", viewer_id))
            break
    else:
        raise ValueError("viewer must appear in players")

    return Observation(
        viewer_id=viewer_id,
        round=round_number,
        phase=phase,
        self_state={
            "id": viewer_id,
            "name": viewer_name,
            "role": role.value,
            "team": role.team.value,
            "alive": True,
            "controller": "ai",
            "death_reason": None,
            "death_round": None,
        },
        players=visible_players,
        events=[dict(event) for event in events],
        legal_actions=[
            {
                "action_type": action_type.value,
                "target_ids": list(targets),
                "can_pass": can_pass,
                "content_required": content_required,
                "description": "test action",
            }
        ],
        private_facts=dict(private_facts or {}),
    )


def assert_legal(test: unittest.TestCase, action: Any, observation: Observation) -> None:
    """Assert that a role agent has already normalized its proposed action."""

    test.assertEqual(action.actor_id, observation.viewer_id)
    if action.action_type is ActionType.PASS:
        test.assertTrue(any(rule.get("can_pass") for rule in observation.legal_actions))
        return
    matching = [
        rule
        for rule in observation.legal_actions
        if ActionType.coerce(rule["action_type"]) is action.action_type
    ]
    test.assertTrue(matching, f"{action.action_type.value} is not a legal action")
    rule = matching[0]
    targets = rule.get("target_ids") or []
    if targets and action.action_type is not ActionType.SPEAK:
        test.assertIn(action.target_id, targets)
    if rule.get("content_required"):
        test.assertTrue(action.content and action.content.strip())


class RecordingProvider:
    """Provider double that records prompts and returns a configured response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], float]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        copied = [dict(message) for message in messages]
        self.calls.append((copied, temperature))
        return self.response


class RoleAgentArchitectureTests(unittest.TestCase):
    @staticmethod
    def five_players() -> list[dict[str, Any]]:
        return [
            {
                "id": f"p{seat}",
                "name": f"Agent {seat}",
                "alive": True,
                "role": None,
            }
            for seat in range(1, 6)
        ]

    def test_default_factory_returns_five_distinct_role_classes(self) -> None:
        for role, expected_type in ROLE_AGENT_TYPES.items():
            with self.subTest(role=role.value):
                agent = create_agent("p1", role)
                self.assertIs(type(agent), expected_type)
                self.assertEqual(agent.role, role)

        self.assertEqual(len(set(ROLE_AGENT_TYPES.values())), len(Role))

    def test_every_role_owns_goal_memory_prompt_and_planner(self) -> None:
        prompts: set[str] = set()
        for role, agent_type in ROLE_AGENT_TYPES.items():
            with self.subTest(role=role.value):
                agent = agent_type()
                self.assertEqual(agent.role, role)
                self.assertIsNotNone(agent.goal)
                self.assertIsInstance(agent.memory, RoleMemory)
                self.assertIsNotNone(agent.planner)
                self.assertIsInstance(agent.prompt, str)
                self.assertGreater(len(agent.prompt.strip()), 20)
                self.assertFalse(hasattr(agent, "role_prompt"))
                prompts.add(agent.prompt)

                # The goal must itself be JSON-observable for evaluation and
                # must identify the role it serves, rather than being a string
                # selected ad hoc inside decide().
                goal_payload = (
                    agent.goal.to_dict()
                    if hasattr(agent.goal, "to_dict")
                    else vars(agent.goal)
                )
                self.assertIn(role.value, json.dumps(goal_payload, ensure_ascii=False))

        self.assertEqual(len(prompts), len(Role), "role prompts must be role-specific")

    def test_same_role_instances_do_not_share_mutable_components(self) -> None:
        first = VillagerAgent()
        second = VillagerAgent()

        self.assertIsNot(first.goal, second.goal)
        self.assertIsNot(first.memory, second.memory)
        self.assertIsNot(first.planner, second.planner)

        observation = make_observation(Role.VILLAGER, ActionType.VOTE)
        first.decide(observation, random.Random(1))
        self.assertNotEqual(first.memory.export(), second.memory.export())

    def test_memory_accumulates_observations_and_can_round_trip(self) -> None:
        agent = VillagerAgent()
        initial = agent.memory.export()
        marker_one = "PUBLIC-EVIDENCE-ALPHA"
        marker_two = "PUBLIC-EVIDENCE-BETA"

        first = make_observation(
            Role.VILLAGER,
            ActionType.SPEAK,
            phase=Phase.DAY_DISCUSSION,
            content_required=True,
            events=(
                {
                    "seq": 4,
                    "round": 1,
                    "phase": "day_discussion",
                    "event_type": "speech",
                    "actor_id": "p2",
                    "target_id": None,
                    "message": marker_one,
                    "data": {"accuse_target": "p3"},
                },
            ),
        )
        action = agent.decide(first, random.Random(2))
        assert_legal(self, action, first)
        after_first = agent.memory.export()
        self.assertNotEqual(initial, after_first)

        second = make_observation(
            Role.VILLAGER,
            ActionType.VOTE,
            round_number=2,
            events=(
                {
                    "seq": 9,
                    "round": 2,
                    "phase": "day_discussion",
                    "event_type": "speech",
                    "actor_id": "p3",
                    "target_id": None,
                    "message": marker_two,
                    "data": {"accuse_target": "p2"},
                },
            ),
        )
        action = agent.decide(second, random.Random(3))
        assert_legal(self, action, second)
        after_second = agent.memory.export()
        self.assertNotEqual(after_first, after_second)

        encoded = json.dumps(after_second, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("hidden_referee_secret", encoded)
        restored = RoleMemory.from_snapshot(json.loads(encoded))
        self.assertEqual(restored.export(), after_second)

    def test_role_planners_produce_legal_primary_actions(self) -> None:
        cases = (
            (
                WerewolfAgent(),
                make_observation(
                    Role.WEREWOLF,
                    ActionType.WOLF_KILL,
                    phase=Phase.NIGHT_WOLF,
                    private_facts={"teammates": []},
                ),
            ),
            (
                SeerAgent(),
                make_observation(
                    Role.SEER,
                    ActionType.SEER_CHECK,
                    phase=Phase.NIGHT_SEER,
                    private_facts={"seer_results": {}},
                ),
            ),
            (
                WitchAgent(),
                make_observation(
                    Role.WITCH,
                    ActionType.WITCH_SAVE,
                    phase=Phase.NIGHT_WITCH_SAVE,
                    targets=("p2",),
                    can_pass=True,
                    private_facts={"night_victim": "p2", "antidote": True, "poison": True},
                ),
            ),
            (
                WitchAgent(),
                make_observation(
                    Role.WITCH,
                    ActionType.WITCH_POISON,
                    phase=Phase.NIGHT_WITCH_POISON,
                    can_pass=True,
                    private_facts={"night_victim": "p2", "antidote": True, "poison": True},
                ),
            ),
            (
                HunterAgent(),
                make_observation(
                    Role.HUNTER,
                    ActionType.HUNTER_SHOOT,
                    phase=Phase.HUNTER_SHOT,
                    can_pass=True,
                ),
            ),
            (
                VillagerAgent(),
                make_observation(
                    Role.VILLAGER,
                    ActionType.SPEAK,
                    phase=Phase.DAY_DISCUSSION,
                    content_required=True,
                ),
            ),
        )

        for agent, observation in cases:
            with self.subTest(role=agent.role.value, action=observation.legal_actions[0]):
                assert_legal(self, agent.decide(observation, random.Random(11)), observation)

    def test_sheriff_run_and_withdraw_defaults_are_role_specific(self) -> None:
        for role, agent_type in ROLE_AGENT_TYPES.items():
            with self.subTest(role=role.value, phase="run"):
                observation = make_observation(
                    role,
                    ActionType.SHERIFF_RUN,
                    phase=Phase.SHERIFF_RUN,
                    targets=(),
                    can_pass=True,
                    private_facts={"teammates": ["p2"]}
                    if role is Role.WEREWOLF
                    else {},
                )
                action = agent_type().decide(observation, random.Random(31))
                assert_legal(self, action, observation)
                expected = (
                    ActionType.SHERIFF_RUN
                    if role in {Role.WEREWOLF, Role.SEER}
                    else ActionType.PASS
                )
                self.assertIs(action.action_type, expected)

        # Only the lowest-seat living wolf is the team's default bluffer.
        second_wolf = make_observation(
            Role.WEREWOLF,
            ActionType.SHERIFF_RUN,
            viewer_id="p2",
            phase=Phase.SHERIFF_RUN,
            can_pass=True,
            targets=(),
            players=self.five_players(),
            private_facts={"teammates": ["p1"]},
        )
        self.assertIs(
            WerewolfAgent().decide(second_wolf, random.Random(32)).action_type,
            ActionType.PASS,
        )

        support_vote = make_observation(
            Role.WEREWOLF,
            ActionType.SHERIFF_VOTE,
            viewer_id="p2",
            phase=Phase.SHERIFF_VOTE,
            targets=("p1", "p3"),
            can_pass=True,
            players=self.five_players(),
            private_facts={"teammates": ["p1"]},
        )
        support = WerewolfAgent().decide(support_vote, random.Random(320))
        self.assertIs(support.action_type, ActionType.SHERIFF_VOTE)
        self.assertEqual(support.target_id, "p1")

        for role, agent_type in ROLE_AGENT_TYPES.items():
            with self.subTest(role=role.value, phase="withdraw"):
                observation = make_observation(
                    role,
                    ActionType.SHERIFF_WITHDRAW,
                    phase=Phase.SHERIFF_WITHDRAW,
                    targets=(),
                    can_pass=True,
                    private_facts={"teammates": ["p2"]}
                    if role is Role.WEREWOLF
                    else {},
                )
                action = agent_type().decide(observation, random.Random(33))
                expected = (
                    ActionType.PASS
                    if role in {Role.WEREWOLF, Role.SEER}
                    else ActionType.SHERIFF_WITHDRAW
                )
                self.assertIs(action.action_type, expected)

    def test_seer_campaign_reports_real_check_and_stable_badge_flow(self) -> None:
        agent = SeerAgent()
        observation = make_observation(
            Role.SEER,
            ActionType.SPEAK,
            phase=Phase.SHERIFF_SPEECH,
            targets=("p2", "p3", "p4", "p5"),
            content_required=True,
            players=self.five_players(),
            private_facts={"seer_results": {"p2": Role.WEREWOLF.value}},
        )
        first = agent.decide(observation, random.Random(41))
        second = agent.decide(observation, random.Random(99))

        for action in (first, second):
            assert_legal(self, action, observation)
            self.assertEqual(action.metadata["claim_role"], Role.SEER.value)
            self.assertEqual(action.metadata["accuse_target"], "p2")
            self.assertIn("狼人", action.content)
            flow = action.metadata["badge_flow"]
            self.assertEqual(flow["mode"], "double")
            self.assertEqual(len(flow["targets"]), 2)
            self.assertEqual(set(flow), {"mode", "targets", "fallback_target"})
            self.assertNotIn("result", json.dumps(flow))
        self.assertEqual(
            first.metadata["badge_flow"], second.metadata["badge_flow"]
        )
        self.assertEqual(
            agent.memory.strategy_state["seer_sheriff"]["badge_flow"],
            first.metadata["badge_flow"],
        )

    def test_werewolf_bluff_uses_consistent_fake_check_and_badge_flow(self) -> None:
        agent = WerewolfAgent()
        observation = make_observation(
            Role.WEREWOLF,
            ActionType.SPEAK,
            phase=Phase.SHERIFF_SPEECH,
            targets=("p2", "p3", "p4", "p5"),
            content_required=True,
            players=self.five_players(),
            private_facts={"teammates": ["p4"]},
        )
        first = agent.decide(observation, random.Random(42))
        runoff = make_observation(
            Role.WEREWOLF,
            ActionType.SPEAK,
            phase=Phase.SHERIFF_PK_SPEECH,
            targets=("p2", "p3", "p4", "p5"),
            content_required=True,
            players=self.five_players(),
            private_facts={"teammates": ["p4"]},
        )
        second = agent.decide(runoff, random.Random(101))

        for action, current in ((first, observation), (second, runoff)):
            assert_legal(self, action, current)
            self.assertEqual(action.metadata["claim_role"], Role.SEER.value)
            self.assertNotIn("p4", action.metadata["badge_flow"]["targets"])
            self.assertNotIn("result", json.dumps(action.metadata["badge_flow"]))
        self.assertEqual(first.content, second.content)
        self.assertEqual(first.metadata, second.metadata)
        wolf_strategy = agent.memory.strategy_state["werewolf_sheriff"]
        self.assertEqual(wolf_strategy["bluffer_id"], "p1")
        self.assertEqual(len(wolf_strategy["fake_seer_results"]), 1)

    def test_seer_checks_declared_double_badge_flow_in_order(self) -> None:
        agent = SeerAgent()
        agent.memory.strategy_state["seer_sheriff"] = {
            "badge_flow": {
                "mode": "double",
                "targets": ["p3", "p4"],
                "fallback_target": "p5",
            }
        }

        first_night = make_observation(
            Role.SEER,
            ActionType.SEER_CHECK,
            phase=Phase.NIGHT_SEER,
            targets=("p2", "p3", "p4", "p5"),
            players=self.five_players(),
            private_facts={"seer_results": {"p2": "not_werewolf"}},
        )
        first = agent.decide(first_night, random.Random(201))
        self.assertEqual(first.target_id, "p3")
        assert_legal(self, first, first_night)

        second_night = make_observation(
            Role.SEER,
            ActionType.SEER_CHECK,
            round_number=2,
            phase=Phase.NIGHT_SEER,
            targets=("p2", "p4", "p5"),
            players=self.five_players(),
            private_facts={
                "seer_results": {
                    "p2": "not_werewolf",
                    "p3": Role.WEREWOLF.value,
                }
            },
        )
        second = agent.decide(second_night, random.Random(202))
        self.assertEqual(second.target_id, "p4")
        assert_legal(self, second, second_night)

    def test_seer_badge_transfer_encodes_completed_flow_results(self) -> None:
        cases = (
            ("single-good", ["p2"], {"p2": "not_werewolf"}, "p4", ActionType.SHERIFF_TRANSFER, "p2"),
            ("single-wolf", ["p2"], {"p2": Role.WEREWOLF.value}, "p4", ActionType.SHERIFF_DESTROY, None),
            ("one-good-one-wolf", ["p2", "p3"], {"p2": "not_werewolf", "p3": Role.WEREWOLF.value}, "p4", ActionType.SHERIFF_TRANSFER, "p2"),
            ("two-good", ["p2", "p3"], {"p2": "not_werewolf", "p3": "not_werewolf"}, "p4", ActionType.SHERIFF_DESTROY, None),
            ("two-wolves", ["p2", "p3"], {"p2": Role.WEREWOLF.value, "p3": Role.WEREWOLF.value}, "p4", ActionType.SHERIFF_TRANSFER, "p4"),
        )
        for label, targets, results, fallback, expected_kind, expected_target in cases:
            with self.subTest(case=label):
                agent = SeerAgent()
                agent.memory.strategy_state["seer_sheriff"] = {
                    "badge_flow": {
                        "mode": "single" if len(targets) == 1 else "double",
                        "targets": targets,
                        "fallback_target": fallback,
                    }
                }
                observation = make_observation(
                    Role.SEER,
                    ActionType.SHERIFF_TRANSFER,
                    phase=Phase.SHERIFF_BADGE,
                    targets=("p2", "p3", "p4", "p5"),
                    players=self.five_players(),
                    private_facts={"seer_results": results},
                )
                observation.legal_actions.append(
                    {
                        "action_type": ActionType.SHERIFF_DESTROY.value,
                        "target_ids": [],
                        "can_pass": False,
                        "content_required": False,
                        "description": "destroy",
                    }
                )
                action = agent.decide(observation, random.Random(43))
                assert_legal(self, action, observation)
                self.assertIs(action.action_type, expected_kind)
                self.assertEqual(action.target_id, expected_target)

        no_result = SeerAgent()
        no_result.memory.strategy_state["seer_sheriff"] = {
            "badge_flow": {
                "mode": "double",
                "targets": ["p2", "p3"],
                "fallback_target": "p4",
            }
        }
        observation = make_observation(
            Role.SEER,
            ActionType.SHERIFF_TRANSFER,
            phase=Phase.SHERIFF_BADGE,
            targets=("p2", "p3", "p4"),
            players=self.five_players(),
            private_facts={"seer_results": {}},
        )
        observation.legal_actions.append(
            {
                "action_type": ActionType.SHERIFF_DESTROY.value,
                "target_ids": [],
                "can_pass": False,
                "content_required": False,
                "description": "destroy",
            }
        )
        self.assertIs(
            no_result.decide(observation, random.Random(44)).action_type,
            ActionType.SHERIFF_DESTROY,
        )

        dead_recipient = SeerAgent()
        dead_recipient.memory.strategy_state["seer_sheriff"] = {
            "badge_flow": {
                "mode": "single",
                "targets": ["p2"],
                "fallback_target": None,
            }
        }
        observation = make_observation(
            Role.SEER,
            ActionType.SHERIFF_TRANSFER,
            phase=Phase.SHERIFF_BADGE,
            targets=("p3", "p4"),
            players=self.five_players(),
            private_facts={"seer_results": {"p2": "not_werewolf"}},
        )
        observation.legal_actions.append(
            {
                "action_type": ActionType.SHERIFF_DESTROY.value,
                "target_ids": [],
                "can_pass": False,
                "content_required": False,
                "description": "destroy",
            }
        )
        self.assertIs(
            dead_recipient.decide(observation, random.Random(45)).action_type,
            ActionType.SHERIFF_DESTROY,
        )

    def test_all_role_agents_support_shared_sheriff_powers(self) -> None:
        for role, agent_type in ROLE_AGENT_TYPES.items():
            private = {"teammates": ["p2"]} if role is Role.WEREWOLF else {}
            if role is Role.SEER:
                private = {"seer_results": {}}

            for speech_phase in (Phase.SHERIFF_SPEECH, Phase.SHERIFF_PK_SPEECH):
                with self.subTest(role=role.value, action=speech_phase.value):
                    observation = make_observation(
                        role,
                        ActionType.SPEAK,
                        phase=speech_phase,
                        targets=("p2", "p3"),
                        content_required=True,
                        private_facts=private,
                    )
                    assert_legal(
                        self,
                        agent_type().decide(observation, random.Random(50)),
                        observation,
                    )

            with self.subTest(role=role.value, action="vote"):
                observation = make_observation(
                    role,
                    ActionType.SHERIFF_VOTE,
                    phase=Phase.SHERIFF_VOTE,
                    targets=("p2", "p3"),
                    can_pass=True,
                    private_facts=private,
                )
                assert_legal(
                    self,
                    agent_type().decide(observation, random.Random(51)),
                    observation,
                )

            with self.subTest(role=role.value, action="order"):
                observation = make_observation(
                    role,
                    ActionType.SHERIFF_ORDER_CLOCKWISE,
                    phase=Phase.SHERIFF_ORDER,
                    targets=(),
                    private_facts=private,
                )
                observation.legal_actions.append(
                    {
                        "action_type": ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE.value,
                        "target_ids": [],
                        "can_pass": False,
                        "content_required": False,
                        "description": "counterclockwise",
                    }
                )
                assert_legal(
                    self,
                    agent_type().decide(observation, random.Random(52)),
                    observation,
                )

            with self.subTest(role=role.value, action="summarize"):
                observation = make_observation(
                    role,
                    ActionType.SHERIFF_SUMMARIZE,
                    phase=Phase.DAY_DISCUSSION,
                    targets=("p2", "p3"),
                    content_required=True,
                    private_facts=private,
                )
                action = agent_type().decide(observation, random.Random(53))
                assert_legal(self, action, observation)
                self.assertEqual(
                    action.metadata["sheriff_nomination"], action.target_id
                )

            with self.subTest(role=role.value, action="transfer"):
                observation = make_observation(
                    role,
                    ActionType.SHERIFF_TRANSFER,
                    phase=Phase.SHERIFF_BADGE,
                    targets=("p2", "p3"),
                    private_facts=private,
                )
                observation.legal_actions.append(
                    {
                        "action_type": ActionType.SHERIFF_DESTROY.value,
                        "target_ids": [],
                        "can_pass": False,
                        "content_required": False,
                        "description": "destroy",
                    }
                )
                assert_legal(
                    self,
                    agent_type().decide(observation, random.Random(54)),
                    observation,
                )

            with self.subTest(role=role.value, action="destroy-only"):
                observation = make_observation(
                    role,
                    ActionType.SHERIFF_DESTROY,
                    phase=Phase.SHERIFF_BADGE,
                    targets=(),
                    private_facts=private,
                )
                action = agent_type().decide(observation, random.Random(55))
                self.assertIs(action.action_type, ActionType.SHERIFF_DESTROY)
                assert_legal(self, action, observation)

    def test_sheriff_order_and_votes_use_public_leadership_evidence(self) -> None:
        players = self.five_players()
        order_observation = make_observation(
            Role.VILLAGER,
            ActionType.SHERIFF_ORDER_CLOCKWISE,
            viewer_id="p3",
            phase=Phase.SHERIFF_ORDER,
            targets=(),
            players=players,
            events=(
                {
                    "seq": 11,
                    "round": 1,
                    "phase": Phase.SHERIFF_SPEECH.value,
                    "event_type": "sheriff_speech",
                    "actor_id": "p1",
                    "target_id": "p2",
                    "message": "I checked p2 as a wolf.",
                    "data": {
                        "claim_role": Role.SEER.value,
                        "accuse_target": "p2",
                    },
                },
            ),
        )
        order_observation.legal_actions.append(
            {
                "action_type": ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE.value,
                "target_ids": [],
                "can_pass": False,
                "content_required": False,
                "description": "counterclockwise",
            }
        )
        order = VillagerAgent().decide(order_observation, random.Random(71))
        # From seat p3, counterclockwise puts p2 first; clockwise puts p2 last.
        self.assertIs(order.action_type, ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE)

        vote_observation = make_observation(
            Role.VILLAGER,
            ActionType.VOTE,
            phase=Phase.DAY_VOTE,
            targets=("p2", "p4"),
            players=players,
            events=(
                {
                    "seq": 21,
                    "round": 1,
                    "phase": Phase.DAY_DISCUSSION.value,
                    "event_type": "sheriff_summary",
                    "actor_id": "p3",
                    "target_id": "p4",
                    "message": "Consolidate on p4.",
                    "data": {"sheriff_nomination": "p4", "vote_intent": "p4"},
                },
            ),
        )
        vote = VillagerAgent().decide(vote_observation, random.Random(72))
        self.assertEqual(vote.target_id, "p4")

    def test_seer_order_uses_verified_checks(self) -> None:
        observation = make_observation(
            Role.SEER,
            ActionType.SHERIFF_ORDER_CLOCKWISE,
            viewer_id="p3",
            phase=Phase.SHERIFF_ORDER,
            targets=(),
            players=self.five_players(),
            private_facts={
                "seer_results": {
                    "p2": Role.WEREWOLF.value,
                    "p4": "not_werewolf",
                }
            },
        )
        observation.legal_actions.append(
            {
                "action_type": ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE.value,
                "target_ids": [],
                "can_pass": False,
                "content_required": False,
                "description": "counterclockwise",
            }
        )
        action = SeerAgent().decide(observation, random.Random(73))
        self.assertIs(action.action_type, ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE)

    def test_provider_badge_flow_is_strictly_validated_and_repaired(self) -> None:
        provider = RecordingProvider(
            json.dumps(
                {
                    "action_type": "speak",
                    "target_id": "p2",
                    "content": "I declare an invalid flow.",
                    "metadata": {
                        "badge_flow": {
                            "mode": "double",
                            "targets": ["p2", "not-a-player"],
                            "fallback_target": "p3",
                            "secret_result": "werewolf",
                        }
                    },
                }
            )
        )
        agent = VillagerAgent(provider=provider)
        observation = make_observation(
            Role.VILLAGER,
            ActionType.SPEAK,
            phase=Phase.SHERIFF_SPEECH,
            targets=("p2", "p3"),
            content_required=True,
        )
        action = agent.decide(observation, random.Random(61))
        assert_legal(self, action, observation)
        self.assertNotIn("badge_flow", action.metadata)
        self.assertEqual(agent.memory.decisions[-1]["source"], "planner_fallback")

    def test_strategy_state_is_optional_and_round_trips_in_legacy_schema(self) -> None:
        memory = RoleMemory(role=Role.SEER, owner_id="p1", viewer_id="p1")
        memory.strategy_state["seer_sheriff"] = {
            "badge_flow": {
                "mode": "single",
                "targets": ["p2"],
                "fallback_target": None,
            }
        }
        snapshot = memory.export()
        self.assertEqual(RoleMemory.from_snapshot(snapshot).export(), snapshot)

        legacy = dict(snapshot)
        legacy.pop("strategy_state")
        restored = RoleMemory.from_snapshot(legacy)
        self.assertEqual(restored.strategy_state, {})

    def test_memory_persists_only_public_sheriff_overlay_in_strategy_state(self) -> None:
        sheriff = {
            "enabled": True,
            "badge_status": "holder",
            "holder_id": "p2",
            "candidates": ["p2", "p3"],
            "withdrawn": [],
            "pk_candidates": [],
            "vote_round": 1,
            "speech_order": ["p3", "p2"],
            "day_direction": "clockwise",
            "day_speech_order": ["p3", "p1", "p2"],
            "nomination_target": None,
        }
        observation = make_observation(
            Role.SEER,
            ActionType.VOTE,
            private_facts={
                "sheriff": sheriff,
                "seer_results": {"p3": Role.WEREWOLF.value},
            },
        )
        agent = SeerAgent()
        agent.decide(observation, random.Random(81))
        self.assertEqual(agent.memory.strategy_state["sheriff_public"], sheriff)
        encoded = json.dumps(agent.memory.strategy_state, sort_keys=True)
        self.assertNotIn("seer_results", encoded)

    def test_provider_rejects_unknown_player_in_public_metadata(self) -> None:
        provider = RecordingProvider(
            json.dumps(
                {
                    "action_type": "speak",
                    "target_id": "p2",
                    "content": "I accuse an unknown player.",
                    "metadata": {"accuse_target": "not-a-seat"},
                }
            )
        )
        observation = make_observation(
            Role.VILLAGER,
            ActionType.SPEAK,
            phase=Phase.DAY_DISCUSSION,
            targets=("p2", "p3"),
            content_required=True,
        )
        agent = VillagerAgent(provider=provider)
        action = agent.decide(observation, random.Random(82))
        assert_legal(self, action, observation)
        self.assertEqual(agent.memory.decisions[-1]["source"], "planner_fallback")

    def test_provider_receives_role_specific_system_prompt(self) -> None:
        system_prompts: set[str] = set()
        for role, agent_type in ROLE_AGENT_TYPES.items():
            provider = RecordingProvider(
                json.dumps({"action_type": "vote", "target_id": "p2"})
            )
            agent = agent_type(provider=provider)
            observation = make_observation(role, ActionType.VOTE)

            action = agent.decide(observation, random.Random(7))
            assert_legal(self, action, observation)
            self.assertEqual(len(provider.calls), 1)
            messages, _temperature = provider.calls[0]
            self.assertEqual(messages[0]["role"], "system")
            system = messages[0]["content"]
            self.assertIn(agent.prompt, system)
            self.assertIn(role.value, system.lower())
            self.assertIn("简体中文", system)
            self.assertIn("JSON字段名和枚举值保留原样", system)
            system_prompts.add(system)

        self.assertEqual(len(system_prompts), len(Role))

    def test_all_roles_use_chinese_for_speech_and_sheriff_summary(self) -> None:
        for role, agent_type in ROLE_AGENT_TYPES.items():
            for phase, action_type in (
                (Phase.DAY_DISCUSSION, ActionType.SPEAK),
                (Phase.SHERIFF_SPEECH, ActionType.SPEAK),
                (Phase.SHERIFF_PK_SPEECH, ActionType.SPEAK),
                (Phase.DAY_PK_SPEECH, ActionType.SPEAK),
                (Phase.DAY_DISCUSSION, ActionType.SHERIFF_SUMMARIZE),
            ):
                with self.subTest(role=role.value, phase=phase.value, action=action_type.value):
                    private = {"teammates": ["p2"]} if role is Role.WEREWOLF else {}
                    if role is Role.SEER:
                        private["seer_results"] = {"p2": Role.WEREWOLF.value}
                    observation = make_observation(
                        role, action_type, phase=phase, content_required=True,
                        players=self.five_players(), private_facts=private,
                    )
                    action = agent_type().decide(observation, random.Random(8))
                    assert_legal(self, action, observation)
                    self.assertRegex(action.content, r"[\u3400-\u9fff]")
                    self.assertNotIn("I ", action.content)

    def test_english_model_speech_falls_back_to_chinese_for_each_role(self) -> None:
        for role, agent_type in ROLE_AGENT_TYPES.items():
            with self.subTest(role=role.value):
                # 汉字姓名不能使一段英文正文通过语言检查。
                players = self.five_players()
                players[1]["name"] = "张三"
                provider = RecordingProvider(json.dumps({
                    "action_type": "speak", "target_id": "p2",
                    "content": "I recommend voting against 张三.",
                }, ensure_ascii=False))
                agent = agent_type(provider=provider)
                observation = make_observation(
                    role, ActionType.SPEAK, phase=Phase.DAY_DISCUSSION,
                    content_required=True, players=players,
                )
                action = agent.decide(observation, random.Random(4))
                assert_legal(self, action, observation)
                self.assertRegex(action.content, r"[\u3400-\u9fff]")
                self.assertNotIn("I recommend", action.content)
                self.assertEqual(agent.memory.decisions[-1]["source"], "planner_fallback")

    def test_chinese_model_speech_is_used_and_saved_without_translation(self) -> None:
        content = "我希望先听取这位玩家对票型的解释，再决定是否放逐。"
        provider = RecordingProvider(json.dumps({
            "action_type": "speak", "target_id": "p2", "content": content,
        }, ensure_ascii=False))
        agent = VillagerAgent(provider=provider)
        observation = make_observation(
            Role.VILLAGER, ActionType.SPEAK,
            phase=Phase.DAY_DISCUSSION, content_required=True,
        )
        action = agent.decide(observation, random.Random(4))
        self.assertEqual(action.content, content)
        self.assertEqual(agent.memory.decisions[-1]["source"], "provider")
        self.assertEqual(agent.memory.decisions[-1]["action"]["content"], content)

    def test_bad_provider_output_is_repaired_inside_agent(self) -> None:
        invalid_responses = (
            "not json at all",
            json.dumps({"actor_id": "p999", "action_type": "vote", "target_id": "p2"}),
            json.dumps({"action_type": "seer_check", "target_id": "p2"}),
            json.dumps({"action_type": "vote", "target_id": "not-a-player"}),
        )
        for response in invalid_responses:
            provider = RecordingProvider(response)
            agent = VillagerAgent(provider=provider)
            observation = make_observation(Role.VILLAGER, ActionType.VOTE)
            with self.subTest(response=response):
                action = agent.decide(observation, random.Random(13))
                assert_legal(self, action, observation)

        # Content-required actions need the same validation; an empty LLM
        # speech must be replaced before the engine sees it.
        provider = RecordingProvider(json.dumps({"action_type": "speak", "content": ""}))
        speaker = VillagerAgent(provider=provider)
        speech_observation = make_observation(
            Role.VILLAGER,
            ActionType.SPEAK,
            phase=Phase.DAY_DISCUSSION,
            content_required=True,
        )
        assert_legal(
            self,
            speaker.decide(speech_observation, random.Random(17)),
            speech_observation,
        )

    def test_complete_default_game_needs_no_engine_action_repair(self) -> None:
        roles = [
            Role.WEREWOLF,
            Role.WEREWOLF,
            Role.SEER,
            Role.WITCH,
            Role.HUNTER,
            Role.VILLAGER,
            Role.VILLAGER,
            Role.VILLAGER,
        ]
        game = WerewolfGame(
            GameConfig(roles=roles, shuffle_roles=False, seed=20261003, max_rounds=20)
        )

        expected = {player_id: ROLE_AGENT_TYPES[player.role] for player_id, player in game.players.items()}
        for player_id, expected_type in expected.items():
            self.assertIs(type(game.agents[player_id]), expected_type)

        state = game.run()
        self.assertEqual(state["status"], "finished")
        repaired = [
            event
            for event in game.event_log(omniscient=True)
            if event["event_type"] == "agent_action_repaired"
        ]
        self.assertEqual(repaired, [])

    def test_same_seed_reproduces_events_and_long_term_memories(self) -> None:
        config = GameConfig(
            roles=[
                Role.WEREWOLF,
                Role.WEREWOLF,
                Role.SEER,
                Role.WITCH,
                Role.HUNTER,
                Role.VILLAGER,
                Role.VILLAGER,
                Role.VILLAGER,
            ],
            shuffle_roles=False,
            seed=73021,
            max_rounds=20,
        )
        first = WerewolfGame(config)
        second = WerewolfGame(GameConfig.from_dict(config.to_dict()))

        first.run()
        second.run()

        self.assertEqual(
            first.event_log(omniscient=True),
            second.event_log(omniscient=True),
        )
        first_memories = {
            player_id: agent.memory.export()
            for player_id, agent in first.agents.items()
        }
        second_memories = {
            player_id: agent.memory.export()
            for player_id, agent in second.agents.items()
        }
        self.assertEqual(first_memories, second_memories)

    def test_villager_and_seer_memories_do_not_acquire_other_roles_secrets(self) -> None:
        game = WerewolfGame(
            GameConfig(
                roles=[
                    Role.WEREWOLF,
                    Role.WEREWOLF,
                    Role.SEER,
                    Role.WITCH,
                    Role.HUNTER,
                    Role.VILLAGER,
                    Role.VILLAGER,
                    Role.VILLAGER,
                ],
                shuffle_roles=False,
                seed=8803,
                max_rounds=20,
            )
        )
        game.run()

        villager_memory = json.dumps(
            game.agents["p6"].memory.export(), ensure_ascii=False, sort_keys=True
        )
        for forbidden in (
            '"teammates"',
            '"seer_results"',
            '"night_victim"',
            '"antidote"',
            '"poison"',
        ):
            self.assertNotIn(forbidden, villager_memory)

        seer_memory = json.dumps(
            game.agents["p3"].memory.export(), ensure_ascii=False, sort_keys=True
        )
        # A seer may retain their own seer_results, but never the wolves'
        # teammate map nor the witch's victim/potion state.
        for forbidden in ('"teammates"', '"night_victim"', '"antidote"', '"poison"'):
            self.assertNotIn(forbidden, seer_memory)

    def test_agent_internal_state_never_enters_public_state_or_event_log(self) -> None:
        game = WerewolfGame(GameConfig(seed=1919, max_rounds=20))
        game.run()

        serialized = json.dumps(
            {
                "state": game.public_state(),
                "events": game.event_log(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for forbidden in (
            '"memory"',
            '"beliefs"',
            '"private_facts"',
            '"internal_reasoning"',
            '"planner_state"',
            '"role_prompt"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_factory_bound_agent_rejects_owner_and_role_mismatch(self) -> None:
        bound = create_agent("p1", Role.VILLAGER)

        wrong_owner = make_observation(Role.VILLAGER, ActionType.VOTE)
        wrong_owner.viewer_id = "p2"
        wrong_owner.self_state["id"] = "p2"
        with self.assertRaises(ValueError):
            bound.decide(wrong_owner, random.Random(21))

        wrong_role = make_observation(Role.SEER, ActionType.VOTE)
        with self.assertRaises(ValueError):
            bound.decide(wrong_role, random.Random(22))


if __name__ == "__main__":
    unittest.main()
