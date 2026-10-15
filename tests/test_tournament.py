import json
import tempfile
import unittest
from pathlib import Path

from backend.agents.villager import VillagerAgent
from backend.game.models import Role
from backend.evaluation.tournament import (
    Competitor,
    TournamentError,
    TournamentRunner,
    _instantiate_agent,
    _parse_seeds,
    main,
    run_tournament,
)


class FakeGame:
    def __init__(self, config, agent_factory=None):
        self.config = dict(config)
        self.agent_factory = agent_factory
        self.status = "running"
        self.winner = None
        self.events = []

    def run(self, max_steps=100):
        self.status = "finished"
        self.winner = "village" if self.config["seed"] % 2 else "werewolves"
        self.events = [
            {"seq": 1, "round": 1, "event_type": "speech", "actor_id": "p1", "data": {"vote_intent": "p2"}},
            {"seq": 2, "round": 1, "event_type": "vote_cast", "actor_id": "p1", "target_id": "p2"},
            {"seq": 3, "round": 1, "event_type": "death", "target_id": "p2"},
            {"seq": 4, "round": 1, "event_type": "game_finished", "data": {"winner": self.winner}},
        ]
        return self.public_state(omniscient=True)

    def public_state(self, viewer_id=None, omniscient=False):
        return {
            "status": self.status,
            "winner": self.winner,
            "players": [
                {"id": "p1", "role": "seer", "team": "village", "alive": True},
                {"id": "p2", "role": "werewolf", "team": "werewolves", "alive": False},
            ],
        }

    def event_log(self, viewer_id=None, omniscient=False):
        return self.events

    def result(self):
        return {"status": self.status, "winner": self.winner, "players": self.public_state()["players"]}


class TournamentTests(unittest.TestCase):
    def test_agent_factories_receive_seat_context_and_create_independent_instances(self):
        class ContextAgent:
            def __init__(self, player_id, role):
                self.player_id = player_id
                self.role = role

            def decide(self, observation, rng):
                raise NotImplementedError("construction-only test double")

        contextual = _instantiate_agent(
            Competitor("context", factory=ContextAgent), "p7", Role.SEER
        )
        self.assertEqual(contextual.player_id, "p7")
        self.assertIs(contextual.role, Role.SEER)

        # 统一工厂显式创建新实例，不再反射签名或猜测如何克隆原型。
        competitor = Competitor("fresh", factory=lambda player_id, role: VillagerAgent(owner_id=player_id))
        first = _instantiate_agent(competitor, "p1", Role.VILLAGER)
        second = _instantiate_agent(competitor, "p2", Role.VILLAGER)
        self.assertIsNot(first, second)
        self.assertIsNot(first.memory, second.memory)
        self.assertEqual((first.owner_id, second.owner_id), ("p1", "p2"))

    def test_unsupported_agent_factories_fail_explicitly(self):
        with self.assertRaisesRegex(TypeError, "prototypes are unsupported"):
            Competitor("prototype", factory=VillagerAgent())
        for factory in (lambda: VillagerAgent(), lambda player_id: VillagerAgent()):
            with self.subTest(factory=factory), self.assertRaisesRegex(TypeError, "accept"):
                _instantiate_agent(Competitor("wrong-signature", factory=factory), "p1", Role.VILLAGER)
        with self.assertRaisesRegex(TypeError, "return an Agent"):
            _instantiate_agent(Competitor("wrong-result", factory=lambda player_id, role: {}), "p1", Role.VILLAGER)

    def test_multi_seed_multi_game_collection_and_rotation(self):
        created = []

        def factory(config, agent_factory=None):
            created.append(dict(config))
            return FakeGame(config, agent_factory)

        report = run_tournament(
            [1, 2],
            games_per_seed=2,
            game_config={"player_count": 2},
            competitors=[
                {"name": "alpha", "model": "model-a", "version": "v2"},
                {"name": "beta", "model": "model-b", "version": "v1"},
            ],
            game_factory=factory,
            tournament_id="cup",
        )

        self.assertEqual(len(created), 4)
        self.assertEqual([item["seed"] for item in created], [1, 1, 2, 2])
        self.assertEqual(report["summaries"]["scheduled_games"], 4)
        self.assertEqual(report["summaries"]["completed_games"], 4)
        self.assertEqual(report["summaries"]["winner_counts"], {"village": 2, "werewolves": 2})
        self.assertEqual(len(report["evaluations"]), 4)
        self.assertEqual(len(report["leaderboard"]["rankings"]), 2)
        first_metadata = report["evaluations"][0]["participant_metadata"]
        second_metadata = report["evaluations"][1]["participant_metadata"]
        self.assertEqual(first_metadata["p1"]["agent"], "alpha")
        self.assertEqual(second_metadata["p1"]["agent"], "beta")
        json.dumps(report, ensure_ascii=False)

    def test_failure_isolation_and_fail_fast(self):
        def unreliable(config, *, agent_factory):
            if config["seed"] == 2:
                raise RuntimeError("broken seed")
            return FakeGame(config, agent_factory)

        report = TournamentRunner(unreliable).run(
            [1, 2, 3], game_config={"player_count": 2}
        )
        self.assertEqual(report["summaries"]["completed_games"], 2)
        self.assertEqual(report["summaries"]["failed_games"], 1)
        self.assertEqual(report["failures"][0]["seed"], 2)
        self.assertEqual(report["failures"][0]["error_type"], "RuntimeError")

        with self.assertRaises(RuntimeError):
            TournamentRunner(unreliable).run(
                [2], game_config={"player_count": 2}, fail_fast=True
            )

    def test_non_terminal_games_are_reported(self):
        class StuckGame(FakeGame):
            def run(self, max_steps=100):
                return self.public_state()

        report = run_tournament(
            [7], game_config={"player_count": 2}, game_factory=StuckGame
        )
        self.assertEqual(report["summaries"]["completed_games"], 0)
        self.assertEqual(report["failures"][0]["error_type"], "TournamentError")
        self.assertIn("did not finish", report["failures"][0]["error"])

    def test_seed_parser_and_cli_json_output(self):
        self.assertEqual(_parse_seeds("1,3-5,2"), (1, 3, 4, 5, 2))
        self.assertEqual(_parse_seeds("3-1"), (3, 2, 1))

        # The real engine is used here as a minimal API/CLI integration smoke test.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            exit_code = main(
                [
                    "--seeds",
                    "11",
                    "--games-per-seed",
                    "1",
                    "--player-count",
                    "5",
                    "--output",
                    str(output),
                    "--compact",
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["summaries"]["completed_games"], 1)
            self.assertEqual(payload["config"]["seeds"], [11])


if __name__ == "__main__":
    unittest.main()
