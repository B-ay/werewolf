import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from backend.game.models import GameConfig
from backend.game.engine import WerewolfGame
from backend.server import ApiError, ArenaManager, GameRecord, create_server


class FakeDeepSeekProvider:
    model = "deepseek-test-model"

    def __init__(self):
        self.secret = "unit-test-secret-never-serialize"
        self.calls = []

    def complete(self, messages, *, temperature=0.2):
        self.calls.append((messages, temperature))
        # Provider output is deliberately generic. BaseAgent's existing legal
        # action validation will fall back to its planner whenever pass is not
        # legal, so this fake never needs network access or hidden game state.
        return json.dumps({"action_type": "pass"})


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deepseek_provider = FakeDeepSeekProvider()
        cls.manager = ArenaManager(deepseek_provider=cls.deepseek_provider)
        cls.server = create_server("127.0.0.1", 0, manager=cls.manager)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, method="GET", payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as response:
            return response.code, json.loads(response.read().decode("utf-8"))

    def test_health_static_and_complete_game_api(self):
        status, health = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])
        self.assertEqual(
            health["transport"],
            {"type": "http_polling", "poll_interval_ms": 2200},
        )
        self.assertEqual(
            health["providers"]["deepseek"],
            {"configured": True, "model": "deepseek-test-model"},
        )
        self.assertNotIn(
            self.deepseek_provider.secret,
            json.dumps(health, ensure_ascii=False),
        )

        with urlopen(f"http://127.0.0.1:{self.port}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        self.assertIn("NOCTURNE", html)

        status, created = self.request(
            "/api/games", "POST", {"player_count": 8, "seed": 808}
        )
        self.assertEqual(status, 201)
        game_id = created["game_id"]
        _, live_report = self.request(f"/api/games/{game_id}/evaluation")
        self.assertIsNone(live_report["result_metrics"]["winner"])
        self.request(
            f"/api/games/{game_id}/run",
            "POST",
            {"delay": 0, "max_steps": 1000},
        )

        state = None
        for _ in range(100):
            _, state = self.request(f"/api/games/{game_id}?omniscient=true")
            if state["status"] == "finished":
                break
            time.sleep(0.01)
        self.assertIsNotNone(state)
        self.assertEqual(state["status"], "finished")
        self.assertIn(state["winner"], {"village", "werewolves", "draw"})

        _, event_page = self.request(f"/api/games/{game_id}/events?after=2")
        self.assertTrue(event_page["events"])
        self.assertTrue(all(event["seq"] > 2 for event in event_page["events"]))

        _, report = self.request(f"/api/games/{game_id}/evaluation")
        self.assertEqual(report["result_metrics"]["winner"], state["winner"])
        self.assertIn("process_metrics", report)
        self.assertIn("replay", report)

    def test_deepseek_strategy_injects_provider_into_five_role_agents(self):
        status, created = self.request(
            "/api/games",
            "POST",
            {"player_count": 5, "seed": 911, "strategy": "deepseek"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["strategy"], "deepseek")
        self.assertEqual(created["model"], "deepseek-test-model")
        self.assertNotIn(
            self.deepseek_provider.secret,
            json.dumps(created, ensure_ascii=False),
        )

        record = self.manager.require(created["game_id"])
        self.assertEqual(
            {type(agent).__name__ for agent in record.game.agents.values()},
            {
                "WerewolfAgent",
                "SeerAgent",
                "WitchAgent",
                "HunterAgent",
                "VillagerAgent",
            },
        )
        self.assertTrue(
            all(
                getattr(agent, "provider", None) is self.deepseek_provider
                for agent in record.game.agents.values()
            )
        )

    def test_sheriff_config_defaults_on_and_can_be_disabled(self):
        status, default_game = self.request(
            "/api/games", "POST", {"player_count": 5, "seed": 905}
        )
        self.assertEqual(status, 201)
        self.assertTrue(default_game["config"]["enable_sheriff"])
        self.assertTrue(default_game["sheriff"]["enabled"])

        status, disabled = self.request(
            "/api/games",
            "POST",
            {"player_count": 5, "seed": 906, "enable_sheriff": False},
        )
        self.assertEqual(status, 201)
        self.assertFalse(disabled["config"]["enable_sheriff"])
        self.assertFalse(disabled["sheriff"]["enabled"])
        self.assertEqual(disabled["sheriff"]["badge_status"], "void")

        record = self.manager.require(disabled["game_id"])
        record.game.run()
        self.assertFalse(
            any(
                event.event_type.startswith("sheriff_election")
                or event.event_type.startswith("sheriff_candidacy")
                for event in record.game.events
            )
        )

    def test_llm_alias_and_missing_provider_validation_are_safe(self):
        alias_manager = ArenaManager(deepseek_provider=self.deepseek_provider)
        alias_state = alias_manager.create(
            {"player_count": 5, "seed": 912, "strategy": "llm"}
        )
        self.assertEqual(alias_state["strategy"], "deepseek")
        self.assertEqual(alias_state["model"], "deepseek-test-model")

        unavailable = ArenaManager(deepseek_provider=None)
        with self.assertRaises(ApiError) as context:
            unavailable.create(
                {"player_count": 5, "seed": 913, "strategy": "deepseek"}
            )
        self.assertEqual(context.exception.status, 400)
        self.assertIn("尚未配置", context.exception.message)

        with self.assertRaises(ApiError) as key_context:
            unavailable.create(
                {
                    "player_count": 5,
                    "seed": 914,
                    "strategy": "heuristic",
                    "api_key": "request-body-secret",
                }
            )
        self.assertEqual(key_context.exception.status, 400)
        self.assertIn("环境变量", key_context.exception.message)

    def test_player_view_route_does_not_return_referee_state(self):
        roles = [
            "werewolf",
            "werewolf",
            "seer",
            "witch",
            "hunter",
            "villager",
            "villager",
            "villager",
        ]
        _, created = self.request(
            "/api/games",
            "POST",
            {"roles": roles, "shuffle_roles": False, "seed": 3},
        )
        game_id = created["game_id"]

        _, wolf_view = self.request(
            f"/api/state?game_id={game_id}&viewer=p1"
        )
        visible_roles = {player["id"]: player["role"] for player in wolf_view["players"]}
        self.assertEqual(visible_roles["p1"], "werewolf")
        self.assertEqual(visible_roles["p2"], "werewolf")
        self.assertIsNone(visible_roles["p3"])
        self.assertNotIn("roles", wolf_view["config"])
        self.assertNotIn("seed", wolf_view["config"])
        self.assertFalse(
            any(event["event_type"] == "role_assigned" and event["actor_id"] == "p3" for event in wolf_view["events"])
        )

        _, redacted_evaluation = self.request(
            f"/api/games/{game_id}/evaluation?viewer=p1&omniscient=true"
        )
        self.assertFalse(redacted_evaluation["available"])
        self.assertEqual(redacted_evaluation["player_metrics"], [])
        self.assertEqual(redacted_evaluation["process_metrics"], {})

        _, forced_wolf_view = self.request(
            f"/api/games/{game_id}?viewer=p1&omniscient=true"
        )
        forced_roles = {
            player["id"]: player["role"] for player in forced_wolf_view["players"]
        }
        self.assertIsNone(forced_roles["p3"])
        self.assertNotIn("roles", forced_wolf_view["config"])
        self.assertNotIn("seed", forced_wolf_view["config"])

        _, public_view = self.request(
            f"/api/state?game_id={game_id}&viewer=public&omniscient=true"
        )
        self.assertTrue(all(player["role"] is None for player in public_view["players"]))
        self.assertNotIn("seed", public_view["config"])
        self.assertNotIn("updated_at", public_view)
        self.assertFalse(
            any(event["event_type"] == "role_assigned" for event in public_view["events"])
        )

        # Mutating endpoints must preserve the same projection. A malicious or
        # buggy client cannot combine a player id with omniscient=true.
        _, step_view = self.request(
            f"/api/games/{game_id}/step",
            "POST",
            {"viewer": "p1", "omniscient": True},
        )
        step_roles = {player["id"]: player["role"] for player in step_view["players"]}
        self.assertEqual(step_roles["p1"], "werewolf")
        self.assertEqual(step_roles["p2"], "werewolf")
        self.assertIsNone(step_roles["p3"])
        self.assertFalse(
            any(event["event_type"] == "role_assigned" and event["actor_id"] == "p3" for event in step_view["events"])
        )

        # Public stepping advances across private referee decisions to the
        # next public change, and a contradictory omniscient flag is ignored.
        _, public_step = self.request(
            f"/api/games/{game_id}/step",
            "POST",
            {"viewer": "public", "omniscient": True},
        )
        self.assertGreater(public_step["event_count"], public_view["event_count"])
        self.assertNotIn("updated_at", public_step)
        self.assertNotIn("seed", public_step["config"])
        self.assertTrue(
            all(
                event["event_type"] not in {"role_assigned", "seer_check", "wolf_vote"}
                for event in public_step["events"]
            )
        )

        _, run_view = self.request(
            f"/api/games/{game_id}/run",
            "POST",
            {"viewer": "p1", "omniscient": True, "delay": 0.1, "max_steps": 2},
        )
        self.assertIsNone(
            next(player for player in run_view["players"] if player["id"] == "p3")["role"]
        )

    def test_mixed_game_creation_never_returns_referee_snapshot(self):
        roles = [
            "hunter",
            "werewolf",
            "werewolf",
            "seer",
            "witch",
            "villager",
            "villager",
            "villager",
        ]
        _, created = self.request(
            "/api/games",
            "POST",
            {
                "roles": roles,
                "shuffle_roles": False,
                "human_player": "p1",
                "seed": 11,
            },
        )
        visible = {player["id"]: player["role"] for player in created["players"]}
        self.assertEqual(created["mode"], "mixed")
        self.assertEqual(visible["p1"], "hunter")
        self.assertTrue(all(visible[player_id] is None for player_id in visible if player_id != "p1"))
        self.assertNotIn("roles", created["config"])
        self.assertNotIn("seed", created["config"])
        private_assignments = [
            event
            for event in created["events"]
            if event["event_type"] == "role_assigned"
        ]
        self.assertEqual([event["actor_id"] for event in private_assignments], ["p1"])

    def test_stale_background_runner_cannot_advance_or_own_new_generation(self):
        manager = ArenaManager()
        config = GameConfig(player_count=5, seed=29)
        game = WerewolfGame(config)
        record = GameRecord("generation-test", game, config)
        record.running = True
        record.run_generation = 2
        before = len(game.events)

        # Simulate the worker from generation 1 reaching its loop after a
        # rapid stop/start has already installed generation 2.
        manager._run_loop(record, delay=0, max_steps=10, generation=1)

        self.assertEqual(len(game.events), before)
        self.assertTrue(record.running)
        self.assertEqual(record.run_generation, 2)


if __name__ == "__main__":
    unittest.main()
