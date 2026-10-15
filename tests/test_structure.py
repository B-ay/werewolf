"""精简架构、独立记忆、固定种子行为和日志隔离回归。"""
import hashlib
import importlib
import inspect
import json
import random
import tempfile
import unittest
from functools import partial
from pathlib import Path

from backend.agents.base import Agent, BaseAgent
from backend.agents.factory import create_agent
from backend.agents.memory import RoleMemory
from backend.agents.planner import RolePlanner
from backend.game.models import ActionType, GameConfig, Phase, Role
from backend.game.engine import WerewolfGame
from backend.logger import EventLogger, write_json_atomic
from tests.test_role_agents import make_observation


class BackendStructureTests(unittest.TestCase):
    def test_role_classes_are_separate_base_agent_subclasses(self):
        self.assertTrue(inspect.isabstract(BaseAgent))
        for role in Role:
            with self.subTest(role=role.value):
                agent = create_agent("p1", role)
                self.assertIsInstance(agent, BaseAgent)
                self.assertEqual(type(agent).__module__, f"backend.agents.{role.value}")
                # 每个角色都通过同一个基类执行记忆、规划和模型降级。
                self.assertIs(type(agent).decide, BaseAgent.decide)
                self.assertIsInstance(agent, Agent)
                self.assertIsInstance(agent.planner, RolePlanner)
                for removed in ("speak", "vote", "decide_night_action", "role_prompt"):
                    self.assertFalse(hasattr(agent, removed))

    def test_removed_compatibility_modules_are_absent(self):
        for module in (
            "backend.agents.compat", "backend.agents.interfaces", "backend.agents.helpers",
            "backend.engine", "backend.models", "backend.providers", "backend.tournament",
        ):
            with self.subTest(module=module):
                self.assertIsNone(importlib.util.find_spec(module))
        for package in ("backend", "backend.game", "backend.agents"):
            self.assertFalse(hasattr(importlib.import_module(package), "__getattr__"))

    def test_single_factory_supports_bound_provider_and_isolated_memory(self):
        class Provider:
            def complete(self, messages, *, temperature=0.2):
                return '{"action_type":"vote","target_id":"p2"}'

        provider = Provider()
        factory = partial(create_agent, provider=provider, temperature=0.4)
        first, second = factory("p1", Role.VILLAGER), factory("p2", "villager")
        self.assertIs(first.provider, second.provider)
        self.assertIsNot(first.memory, second.memory)
        self.assertIsNot(first.goal, second.goal)
        self.assertIsNot(first.planner, second.planner)
        restored = RoleMemory.from_snapshot(first.memory.export())
        self.assertIs(create_agent("p1", Role.VILLAGER, memory=restored).memory, restored)
        with self.assertRaises(ValueError):
            create_agent("p2", Role.VILLAGER, memory=restored)
        with self.assertRaises(ValueError):
            create_agent("p1", Role.SEER, memory=restored)

    def test_fixed_seed_refactor_matches_original_state_events_and_memory(self):
        # 重构前记录的六局基线；故意保持策略、上下文、事件和 RNG 消耗完全不变。
        baselines = [
            (
                5, False,
                "86f4bb6ff1f0fc89f73bf1f6df304c76563b32609212a9f8da5054673f2c25d0",
                "540d906ae5f55c5b22cb6adc5e405f637bdc7c0fe38e080b8f5c42b20d3d9e45",
                "a46b2ff149beab10101326a3fe3d4f194c4fe1aa8ebc208af8d890c56df22895",
            ),
            (
                5, True,
                "1c926e1ce11576b0db0b15169bef66612d8b9de121f00b31921a44b194589c2d",
                "eb75e259a5cd44baa2a8e45a6430aac55713b0f526fb2c1222b29fdbc754ba90",
                "0117e3136ea854ac6f292d2e45d28bfb16c6aebcbe4a5b23c6a59d8b59d4aa3a",
            ),
            (
                42, False,
                "72c997f3a833b301bd01c317b845b278399b99ab84cab476325fb51c6e775465",
                "6d2d445af09b4df4a5a587b249115da1ea74c794801d8c89b959fc1c6f2dd3e3",
                "80f5a620ad893c8f7dc21f700c6c5463256e55ea24eb9348e43d9c5918acaa3a",
            ),
            (
                42, True,
                "9dea245d19c6213794351f95bd0ad5ec95507e8c67e31ef17a20b267911d5b76",
                "767af1d88aa66b3d63804e542228cedb15fb932ba064b34d8c10528444199cdf",
                "bcf3f9caa2ae0bbadccaacfcea4e44cbb1161a7afde6a1725a3b9e3b6fd20f1d",
            ),
            (
                31415, False,
                "cc857993fd333ab495b0df52d955782a18433d9c34d903086ce989c9a5cff356",
                "c17ad7fc0632f8847de29922869a93190a22436ba46216a910ca588b8d05b0f2",
                "799a48fa98486d65db3cd129ccc94634e4cc7593dfb7bcae9d5625f89644b9fc",
            ),
            (
                31415, True,
                "f1fd691afd600d2e3f96a868969d1a052bfccf87261915a692673b2e5570bea7",
                "4f938003984a28541fa6a0d30345a5d846e332c6e7335147c60d4211a5b5b97c",
                "4bd18c0c4b1051ae96a7e28701baafc350d22331d38f86502074739e4ce1d6fc",
            ),
        ]
        for seed, sheriff, *expected_hashes in baselines:
            with self.subTest(seed=seed, sheriff=sheriff):
                game = WerewolfGame(GameConfig(seed=seed, enable_sheriff=sheriff))
                payloads = {
                    "state": game.run(),
                    "events": game.event_log(omniscient=True),
                    "memories": {seat: agent.memory.export() for seat, agent in game.agents.items()},
                }
                hashes = {
                    name: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    for name, value in payloads.items()
                }
                self.assertEqual(hashes, dict(zip(payloads, expected_hashes)))

    def test_common_view_and_context_do_not_expose_mutable_references(self):
        agent = create_agent("p1", Role.SEER)
        self.assertIsNone(agent.state)
        self.assertEqual(agent.build_context_prompt(), "")
        observation = make_observation(
            Role.SEER, ActionType.SEER_CHECK, phase=Phase.NIGHT_SEER,
            private_facts={"seer_results": {"p2": "not_werewolf"}},
        )
        agent.update_state(observation)
        observation.private_facts["seer_results"]["p2"] = "werewolf"
        view = agent.state
        self.assertEqual(view["private_facts"]["seer_results"]["p2"], "not_werewolf")
        view["private_facts"]["seer_results"]["p2"] = "werewolf"
        context = json.loads(agent.build_context_prompt())
        self.assertEqual(context["observation"]["private_facts"]["seer_results"]["p2"], "not_werewolf")
        self.assertTrue(agent.is_alive)
        self.assertNotIn("teammates", context["observation"]["private_facts"])

    def test_memory_windows_compress_without_changing_private_facts(self):
        memory = RoleMemory(role=Role.SEER, owner_id="p1", max_events=2, max_decisions=2, max_episodes=1)
        agent = create_agent("p1", Role.SEER, memory=memory)
        for turn in range(1, 5):
            observation = make_observation(
                Role.SEER, ActionType.SEER_CHECK, phase=Phase.NIGHT_SEER,
                round_number=turn,
                events=[{"seq": seq, "event_type": "speech", "data": {}} for seq in range(1, turn + 1)],
                private_facts={"seer_results": {"p2": "not_werewolf"}},
            )
            agent.decide(observation, random.Random(turn))
        self.assertEqual([event["seq"] for event in memory.events], [3, 4])
        self.assertEqual([decision["round"] for decision in memory.decisions], [3, 4])
        self.assertEqual(memory.private_facts, {"seer_results": {"p2": "not_werewolf"}})
        memory.add_reflection("复盘" * 800)
        self.assertEqual(len(memory.reflections[0]), 1000)
        memory.begin_episode()
        compressed = memory.prompt_view()
        self.assertEqual([event["seq"] for event in compressed["prior_episodes"][0]["recent_visible_events"]], [3, 4])
        compressed["prior_episodes"][0]["recent_visible_events"][0]["seq"] = 999
        self.assertEqual(memory.episodes[0]["events"][0]["seq"], 3)
        agent.decide(make_observation(Role.SEER, ActionType.SEER_CHECK, phase=Phase.NIGHT_SEER), random.Random(5))
        memory.begin_episode()
        self.assertEqual(len(memory.episodes), 1)
        self.assertEqual(RoleMemory.from_snapshot(memory.export()).export(), memory.export())

    def test_decide_records_exactly_one_decision_in_each_phase(self):
        agent = create_agent("p1", Role.VILLAGER)
        speech = make_observation(
            Role.VILLAGER, ActionType.SPEAK,
            phase=Phase.DAY_DISCUSSION, content_required=True,
        )
        self.assertEqual(agent.decide(speech, random.Random(3)).action_type, ActionType.SPEAK)
        vote = make_observation(Role.VILLAGER, ActionType.VOTE, phase=Phase.DAY_VOTE)
        self.assertEqual(agent.decide(vote, random.Random(3)).action_type, ActionType.VOTE)
        self.assertEqual(agent.memory.turns, 2)
        self.assertEqual(len(agent.memory.decisions), 2)

        seer = create_agent("p1", Role.SEER)
        night = make_observation(Role.SEER, ActionType.SEER_CHECK, phase=Phase.NIGHT_SEER)
        self.assertEqual(seer.decide(night, random.Random(3)).action_type, ActionType.SEER_CHECK)
        self.assertEqual(seer.memory.turns, 1)

    def test_logger_projects_sequences_and_detaches_payloads(self):
        logger = EventLogger()
        logger.append(round=1, phase="night_seer", event_type="seer_check", message="私有查验", data={"nested": {"target": "p2"}}, visibility=("p1",))
        logger.append(round=1, phase="day_discussion", event_type="speech", message="公开发言")
        public = logger.project()
        self.assertEqual([event["seq"] for event in public], [1])
        self.assertEqual(public[0]["event_type"], "speech")
        own = logger.project("p1")
        own[0]["data"]["nested"]["target"] = "p999"
        self.assertEqual(logger.project("p1")[0]["data"]["nested"]["target"], "p2")
        self.assertEqual([event["seq"] for event in logger.project(omniscient=True)], [1, 2])

    def test_atomic_storage_preserves_chinese_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "replay.json"
            write_json_atomic(destination, {"message": "中文对局日志"})
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8"))["message"], "中文对局日志")
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["replay.json"])


if __name__ == "__main__":
    unittest.main()
