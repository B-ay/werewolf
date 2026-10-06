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

from backend import GameConfig, Role, WerewolfGame
from backend.agents import (
    HunterAgent,
    RoleMemory,
    SeerAgent,
    VillagerAgent,
    WerewolfAgent,
    WitchAgent,
    default_agent_factory,
)
from backend.models import ActionType, Observation, Phase


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
    round_number: int = 1,
    phase: Phase = Phase.DAY_VOTE,
    targets: Sequence[str] = ("p2", "p3"),
    can_pass: bool = False,
    content_required: bool = False,
    events: Sequence[Mapping[str, Any]] = (),
    private_facts: Mapping[str, Any] | None = None,
) -> Observation:
    """Build the same sanitized value object that the engine gives a policy."""

    return Observation(
        viewer_id="p1",
        round=round_number,
        phase=phase,
        self_state={
            "id": "p1",
            "name": "Agent One",
            "role": role.value,
            "team": role.team.value,
            "alive": True,
            "controller": "ai",
            "death_reason": None,
            "death_round": None,
        },
        players=[
            {"id": "p1", "name": "Agent One", "alive": True, "role": role.value},
            {"id": "p2", "name": "Agent Two", "alive": True, "role": None},
            {"id": "p3", "name": "Agent Three", "alive": True, "role": None},
        ],
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

    rule = observation.legal_actions[0]
    expected = ActionType.coerce(rule["action_type"])
    allowed = {expected}
    if rule.get("can_pass"):
        allowed.add(ActionType.PASS)
    test.assertEqual(action.actor_id, observation.viewer_id)
    test.assertIn(action.action_type, allowed)
    if action.action_type is ActionType.PASS:
        test.assertTrue(rule.get("can_pass"))
        return
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
    def test_default_factory_returns_five_distinct_role_classes(self) -> None:
        for role, expected_type in ROLE_AGENT_TYPES.items():
            with self.subTest(role=role.value):
                agent = default_agent_factory("p1", role)
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
                self.assertEqual(agent.prompt, agent.role_prompt)
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
            system_prompts.add(system)

        self.assertEqual(len(system_prompts), len(Role))

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
        bound = default_agent_factory("p1", Role.VILLAGER)

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
