"""自进化闭环契约：离线完成，不消耗真实 API 额度。"""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen

from backend.agents.factory import create_agent
from backend.agents.memory import RoleMemory
from backend.agents.strategy import RoleStrategy, propose_strategy
from backend.evaluation.evolution import (
    EvolutionCoordinator, EvolutionRunner, ExperienceStore, game_feedback,
    main, profile_factory, scope_key, validation_gate,
)
from backend.game.models import ActionType, GameConfig, Role
from backend.server import ApiError, ArenaManager, create_server
from tests.test_role_agents import make_observation
from tests.test_server import FakeDeepSeekProvider


def winning_pairs(count=24):
    return [{"baseline": 0.0, "candidate": 1.0, "quality_delta": 0.1, "unsafe": False}
            for _ in range(count)]


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "strategies.json"
        self.store = ExperienceStore(self.path)
        self.runner = EvolutionRunner(self.store, GameConfig(seed=42))
        self.scope = self.runner.scope

    def promotion(self, strategy):
        result = {**validation_gate(winning_pairs()), "baseline_version": "v0",
                  "outcomes": winning_pairs()}
        self.store.commit(self.scope, strategy, result)

    def test_strategy_rejects_private_facts_and_unbounded_parameters(self):
        invalid = [
            {"role": "seer", "facts": {"p2": "werewolf"}},
            {"role": "seer", "parameters": {"poison_threshold": 4}},
            {"role": "witch", "parameters": {"poison_threshold": 100}},
            {"role": "witch", "parameters": {"poison_threshold": float("nan")}},
            {"role": "seer", "parameters": {"suspect_checks": 1}},
            {"role": "seer", "lesson_codes": ["p2 是狼人"]},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RoleStrategy.from_dict(value)

    def test_factory_inherits_generic_experience_but_not_old_game_memory(self):
        profiles = self.store.profiles(self.scope)
        profiles[Role.WITCH] = RoleStrategy(Role.WITCH, "v2", {"poison_threshold": 4}, ("witch_poison",))
        factory = profile_factory(profiles)
        first, second = factory("p1", Role.WITCH), factory("p2", Role.WITCH)
        first.memory.private_facts["night_victim"] = "p8"
        self.assertNotIn("night_victim", second.memory.private_facts)
        self.assertIsNot(first.memory, second.memory)
        self.assertIsNot(first.planner.parameters, second.planner.parameters)
        self.assertEqual(second.memory.experience["version"], "v2")
        self.assertTrue(second.memory.reflections)
        self.assertFalse(second.memory.episodes)
        second.memory.experience["parameters"]["poison_threshold"] = 7
        self.assertEqual(profiles[Role.WITCH].parameters["poison_threshold"], 4)

    def test_memory_snapshot_backwards_compatible_and_restores_strategy(self):
        agent = create_agent("p1", Role.HUNTER, strategy=RoleStrategy(Role.HUNTER, "v3", {"shoot_threshold": 3}, ("hunter_shot",)))
        restored = RoleMemory.from_snapshot(agent.memory.export())
        clone = create_agent("p1", Role.HUNTER, memory=restored)
        self.assertEqual(clone.planner.parameters["shoot_threshold"], 3)
        self.assertEqual(clone.memory.experience, agent.memory.experience)
        old = RoleMemory(role=Role.HUNTER).export()
        self.assertNotIn("experience", old)
        self.assertFalse(RoleMemory.from_snapshot(old).experience)
        with self.assertRaises(ValueError):
            create_agent("p1", Role.WITCH, strategy=RoleStrategy(Role.SEER))

    def test_parameters_change_real_hunter_action(self):
        observation = make_observation(Role.HUNTER, ActionType.HUNTER_SHOOT, can_pass=True)
        baseline = create_agent("p1", Role.HUNTER)
        candidate = create_agent("p1", Role.HUNTER, strategy=RoleStrategy(Role.HUNTER, "v1", {"shoot_threshold": 2}))
        self.assertIs(baseline.decide(observation, random.Random(1)).action_type, ActionType.HUNTER_SHOOT)
        self.assertIs(candidate.decide(observation, random.Random(1)).action_type, ActionType.PASS)

    def test_role_parameters_change_poison_check_and_night_kill(self):
        events = [{"seq": 1, "event_type": "speech", "actor_id": "p3",
                   "round": 1, "data": {"claim_role": "seer", "accuse_target": "p2"}}]
        witch = make_observation(Role.WITCH, ActionType.WITCH_POISON, can_pass=True, events=events)
        self.assertIs(create_agent("p1", Role.WITCH).decide(witch, random.Random(1)).action_type, ActionType.WITCH_POISON)
        cautious = create_agent("p1", Role.WITCH, strategy=RoleStrategy(Role.WITCH, "v1", {"poison_threshold": 4}))
        self.assertIs(cautious.decide(witch, random.Random(1)).action_type, ActionType.PASS)
        seer = make_observation(Role.SEER, ActionType.SEER_CHECK, events=events)
        inquisitive = create_agent("p1", Role.SEER, strategy=RoleStrategy(Role.SEER, "v1", {"suspect_checks": True}))
        self.assertEqual(inquisitive.decide(seer, random.Random(1)).target_id, "p2")
        campaign = [{"seq": 1, "event_type": "sheriff_speech", "actor_id": "p3",
                     "round": 1, "data": {"claim_role": "seer"}}]
        wolf = make_observation(Role.WEREWOLF, ActionType.WOLF_KILL, events=campaign)
        focused = create_agent("p1", Role.WEREWOLF, strategy=RoleStrategy(Role.WEREWOLF, "v1", {"include_campaign_claims": True}))
        self.assertEqual(focused.decide(wolf, random.Random(1)).target_id, "p3")

    def test_public_weight_update_changes_vote(self):
        events = [{"seq": 1, "event_type": "speech", "actor_id": "p2", "round": 1,
                   "data": {"claim_role": "seer", "accuse_target": "p3"}},
                  {"seq": 2, "event_type": "speech", "actor_id": "p3", "round": 1,
                   "data": {"accuse_target": "p2"}}]
        observation = make_observation(Role.VILLAGER, ActionType.VOTE, events=events)
        baseline = create_agent("p1", Role.VILLAGER)
        skeptical = create_agent("p1", Role.VILLAGER, strategy=RoleStrategy(Role.VILLAGER, "v1", {"seer_accuse_weight": 0.5}))
        self.assertEqual(baseline.decide(observation, random.Random(1)).target_id, "p3")
        self.assertEqual(skeptical.decide(observation, random.Random(1)).target_id, "p2")

    def test_model_context_contains_inherited_lesson_and_parameters(self):
        provider = FakeDeepSeekProvider()
        agent = create_agent("p1", Role.HUNTER, provider=provider,
                             strategy=RoleStrategy(Role.HUNTER, "v1", {"shoot_threshold": 2}, ("hunter_shot",)))
        observation = make_observation(Role.HUNTER, ActionType.HUNTER_SHOOT, can_pass=True)
        agent.decide(observation, random.Random(1))
        payload = json.loads(provider.calls[0][0][1]["content"])
        self.assertEqual(payload["memory"]["experience"]["version"], "v1")
        self.assertIn("开枪", str(payload["memory"]))
        self.assertNotIn(provider.secret, json.dumps(payload))

    def test_feedback_is_terminal_aggregate_and_idempotent(self):
        game, report = self.runner._play(42, self.store.profiles(self.scope))
        feedback = game_feedback(game, report)
        self.assertEqual(set(feedback), {role.value for role in Role})
        for counters in feedback.values():
            self.assertEqual(counters["games"], 1)
            self.assertLessEqual(counters["wrong_votes"], counters["votes"])
            self.assertLessEqual(counters["wrong_abilities"], counters["abilities"])
        self.assertTrue(self.store.record(self.scope, "g1", 42, feedback))
        self.assertFalse(self.store.record(self.scope, "g1", 42, feedback))
        self.assertEqual(self.store.summary(self.scope)["training_game_count"], 1)
        self.assertNotIn('"p1"', self.path.read_text(encoding="utf-8"))
        from backend.game.engine import WerewolfGame
        with self.assertRaises(ValueError):
            game_feedback(WerewolfGame(GameConfig(seed=42)), report)

    def test_invalid_feedback_is_transactional(self):
        self.store.record(self.scope, "good", 1, {"witch": {"games": 1}})
        before = self.store.summary(self.scope)
        with self.assertRaises(ValueError):
            self.store.record(self.scope, "bad", 2, {"witch": {"games": 1, "secret_role": 1}})
        self.assertEqual(self.store.summary(self.scope), before)

    def test_scopes_separate_rules_and_models(self):
        config = GameConfig(seed=1)
        self.assertEqual(scope_key(config), scope_key(GameConfig(seed=2)))
        self.assertNotEqual(scope_key(config), scope_key(GameConfig(enable_sheriff=False)))
        self.assertNotEqual(scope_key(config), scope_key(config, FakeDeepSeekProvider()))

    def test_late_feedback_from_old_version_does_not_train_new_version(self):
        self.promotion(RoleStrategy(Role.HUNTER, "v1", {"shoot_threshold": 1}))
        self.store.record(self.scope, "late-old-game", 88, {"hunter": {"games": 1, "losses": 1}},
                          versions={"hunter": "v0"})
        self.assertNotIn("hunter", self.store.summary(self.scope)["feedback"])

    def test_human_actions_are_not_attributed_to_ai_role(self):
        from backend.game.engine import WerewolfGame
        board = {**GameConfig(seed=42).to_dict(), "human_players": [1]}
        game = WerewolfGame(board)
        human_role = game.players["p1"].role
        # 在完整 AI 运行之后投影成人类控制，单独测试反馈边界。
        game.players["p1"].controller = "ai"
        game.run()
        from backend.evaluation import evaluate_game
        report = evaluate_game(game.event_log(omniscient=True), game.result(), game.winner)
        full = game_feedback(game, report)
        game.players["p1"].controller = "human"
        limited = game_feedback(game, report)
        other_ai = [player for player in game.players.values() if player.id != "p1" and player.role == human_role]
        if not other_ai:
            self.assertNotIn(human_role.value, limited)
        else:
            self.assertLessEqual(limited[human_role.value]["votes"], full[human_role.value]["votes"])

    def test_restart_and_two_store_instances_do_not_overwrite_updates(self):
        second = ExperienceStore(self.path)
        self.store.record(self.scope, "g1", 1, {"witch": {"games": 1}})
        second.record(self.scope, "g2", 2, {"witch": {"games": 1}})
        self.assertEqual(self.store.summary(self.scope)["feedback"]["witch"]["games"], 2)
        strategy = RoleStrategy(Role.HUNTER, "v1", {"shoot_threshold": 1}, ("hunter_shot",))
        self.promotion(strategy)
        restarted = ExperienceStore(self.path)
        self.assertEqual(restarted.profiles(self.scope)[Role.HUNTER].version, "v1")

    def test_gate_accepts_only_sufficient_safe_evidence(self):
        self.assertEqual(validation_gate(winning_pairs())["status"], "accepted")
        self.assertEqual(validation_gate(winning_pairs(23))["status"], "inconclusive")
        tied = [{**pair, "baseline": 1} for pair in winning_pairs()]
        self.assertEqual(validation_gate(tied)["status"], "rejected")
        unsafe = winning_pairs()
        unsafe[0]["unsafe"] = True
        self.assertEqual(validation_gate(unsafe)["reason"], "safety_regression")
        worse = [{**pair, "quality_delta": -0.03} for pair in winning_pairs()]
        self.assertEqual(validation_gate(worse)["reason"], "role_contribution_regression")
        with self.assertRaises(ValueError):
            validation_gate([{**winning_pairs()[0], "quality_delta": float("nan")}])

    def test_rejection_insufficient_and_forged_evidence_never_promote(self):
        strategy = RoleStrategy(Role.HUNTER, "v1", {"shoot_threshold": 1})
        for result in [
            {"status": "rejected", "reason": "no_gain", "baseline_version": "v0"},
            {"status": "inconclusive", "reason": "too_few", "baseline_version": "v0"},
            {"status": "accepted", "reason": "forged", "baseline_version": "v0"},
        ]:
            self.store.commit(self.scope, strategy, result)
            self.assertEqual(self.store.profiles(self.scope)[Role.HUNTER].version, "v0")

    def test_stale_version_cannot_replace_newer_version(self):
        self.promotion(RoleStrategy(Role.HUNTER, "v1", {"shoot_threshold": 1}))
        self.promotion(RoleStrategy(Role.HUNTER, "v2", {"shoot_threshold": 2}))
        self.assertEqual(self.store.profiles(self.scope)[Role.HUNTER].version, "v1")
        self.assertEqual(self.store.summary(self.scope)["history"][-1]["reason"], "baseline_version_changed")

    def test_actual_paired_validation_and_seed_holdout(self):
        self.runner.train([42])
        proposals = self.store.candidates(self.scope)
        candidate, _ = proposals[0]
        with self.assertRaises(ValueError):
            self.runner.validate(candidate, [42])
        with self.assertRaises(ValueError):
            self.runner.validate(candidate, [801, 801])
        result = self.runner.validate(candidate, [801, 802])
        self.assertEqual(result["pairs"], 2)
        self.assertIn(result["status"], {"inconclusive", "rejected"})
        self.assertFalse(any(pair["unsafe"] for pair in result["outcomes"]))
        self.assertEqual(self.store.summary(self.scope)["training_game_count"], 1)
        with self.assertRaises(ValueError):
            self.runner.train([801])

    def test_complete_two_round_loop_records_each_candidate_without_auto_promotion(self):
        report = self.runner.run([42], rounds=2, validation_games=2)
        self.assertEqual(len(report["iterations"]), 2)
        self.assertEqual(report["final"]["training_game_count"], 2)
        updates = [update for iteration in report["iterations"] for update in iteration["updates"]]
        self.assertTrue(updates)
        self.assertFalse(any(update["reason"] == "validation_failed" for update in updates))
        self.assertTrue(all(update["status"] != "accepted" for update in updates))
        self.assertTrue(all(value["version"] == "v0" for value in report["final"]["active"].values()))

    def test_accepted_strategy_used_in_next_training_game(self):
        self.promotion(RoleStrategy(Role.HUNTER, "v1", {"shoot_threshold": 1}, ("hunter_shot",)))
        game, _ = self.runner._play(42, self.store.profiles(self.scope))
        hunters = [agent for agent in game.agents.values() if agent.role is Role.HUNTER]
        self.assertEqual(hunters[0].memory.experience["version"], "v1")
        self.assertEqual(hunters[0].planner.parameters["shoot_threshold"], 1)

    def test_paid_runs_require_explicit_authorization(self):
        provider = FakeDeepSeekProvider()
        runner = EvolutionRunner(self.store, provider=provider)
        with self.assertRaises(ValueError):
            runner.train([42])
        with self.assertRaises(ValueError):
            runner.validate(RoleStrategy(Role.HUNTER), [700])
        self.assertEqual(provider.calls, [])
        coordinator = EvolutionCoordinator(self.store)
        self.assertEqual(coordinator.schedule(GameConfig(), provider), "requires_paid_validation")
        self.assertEqual(provider.calls, [])

    def test_authorized_fake_model_validation_is_separate_from_local(self):
        provider = FakeDeepSeekProvider()
        runner = EvolutionRunner(self.store, provider=provider, allow_paid_validation=True)
        runner.train([42])
        candidate = self.store.candidates(runner.scope)[0][0]
        result = runner.validate(candidate, [8501, 8502])
        self.assertEqual(result["pairs"], 2)
        self.assertEqual(result["mode"], provider.model)
        self.assertTrue(provider.calls)
        self.assertEqual(self.store.summary(self.scope)["training_game_count"], 0)

    def test_proposals_are_bounded_and_require_feedback(self):
        for role in Role:
            self.assertIsNone(propose_strategy(RoleStrategy(role), {}, 1))
            candidate = propose_strategy(RoleStrategy(role), {"games": 1, "losses": 1}, 1)
            self.assertEqual(candidate.role, role)
            self.assertTrue(candidate.lessons)
            RoleStrategy.from_dict(candidate.to_dict())

    def test_cli_writes_offline_report(self):
        output = Path(self.temp.name) / "report.json"
        code = main(["--seeds", "42", "--validation-games", "1", "--store", str(self.path), "--output", str(output)])
        self.assertEqual(code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], "1.0")
        self.assertTrue(report["iterations"][0]["updates"])


class EvolutionServerTests(unittest.TestCase):
    def test_feedback_arriving_during_validation_is_not_lost(self):
        with TemporaryDirectory() as directory:
            coordinator = EvolutionCoordinator(ExperienceStore(Path(directory) / "strategies.json"))
            started, finish = threading.Event(), threading.Event()
            calls = []
            def optimize(runner):
                calls.append(runner.scope)
                if len(calls) == 1:
                    started.set()
                    if not finish.wait(5):
                        raise RuntimeError("test synchronization timeout")
                return []
            with patch.object(EvolutionRunner, "optimize", optimize):
                coordinator.schedule(GameConfig())
                self.assertTrue(started.wait(5))
                coordinator.schedule(GameConfig())
                finish.set()
                coordinator.wait(10)
            self.assertEqual(len(calls), 2)
            self.assertEqual(coordinator.status()["jobs"][calls[0]]["status"], "finished")

    def test_opt_in_feedback_status_and_no_automatic_paid_calls(self):
        with TemporaryDirectory() as directory, patch("backend.server.DATA_ROOT", Path(directory)):
            store = ExperienceStore(Path(directory) / "evolution" / "strategies.json")
            provider = FakeDeepSeekProvider()
            manager = ArenaManager(provider, evolution_store=store)
            normal = manager.create({"seed": 42})
            self.assertNotIn("evolution", normal)
            with self.assertRaises(ApiError):
                manager.create({"enable_evolution": "false"})
            enabled = manager.create({"seed": 42, "enable_evolution": True})
            record = manager.require(enabled["game_id"])
            self.assertEqual(enabled["evolution"]["versions"]["seer"], "v0")
            self.assertFalse(manager.evolution_status()["scopes"][enabled["evolution"]["scope"]]["feedback"])
            record.game.run()
            # 实际调度器另行测试；此处隔离长批次，只检查完局接线和幂等性。
            with patch.object(manager.learning(), "schedule", return_value="running") as schedule:
                manager._finalize_if_needed(record)
                manager._finalize_if_needed(record)
                schedule.assert_called_once()
            self.assertTrue(record.report["evolution"]["recorded"])
            self.assertTrue(manager.state(record.game_id)["evaluation"]["evolution"])
            saved = json.loads((Path(directory) / f"{record.game_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["evaluation"]["evolution"]["versions_used"]["seer"], "v0")
            paid = manager.create({"strategy": "deepseek", "seed": 43, "enable_evolution": True})
            paid_record = manager.require(paid["game_id"])
            paid_record.game.run()
            call_count = len(provider.calls)
            manager._finalize_if_needed(paid_record)
            self.assertEqual(len(provider.calls), call_count)
            self.assertEqual(paid_record.report["evolution"]["validation_status"], "requires_paid_validation")
            self.assertNotIn(provider.secret, json.dumps(manager.evolution_status()))
            server = create_server("127.0.0.1", 0, manager=manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/evolution") as response:
                    self.assertIn("scopes", json.load(response))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_local_background_validation_can_complete(self):
        with TemporaryDirectory() as directory:
            store = ExperienceStore(Path(directory) / "strategies.json")
            runner = EvolutionRunner(store)
            runner.train([42])
            coordinator = EvolutionCoordinator(store)
            original = EvolutionRunner.optimize
            # 调度/线程/落盘使用真实流程，缩小验证样本使单测迅速结束。
            with patch.object(EvolutionRunner, "optimize", lambda self: original(self, 1)):
                self.assertEqual(coordinator.schedule(GameConfig()), "running")
                coordinator.wait(30)
            result = coordinator.status()["jobs"][runner.scope]
            self.assertEqual(result["status"], "finished")
            self.assertTrue(result["updates"])
            self.assertFalse(any(update["reason"] == "validation_failed" for update in result["updates"]))


if __name__ == "__main__":
    unittest.main()
