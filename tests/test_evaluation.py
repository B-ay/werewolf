import unittest

from backend.evaluation import EloLeaderboard, build_leaderboard, evaluate_game


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.players = [
            {"id": "seer", "name": "先知", "role": "seer", "alive": True},
            {"id": "v1", "role": "villager", "alive": True},
            {"id": "v2", "role": "villager", "alive": False},
            {"id": "wolf1", "role": "werewolf", "alive": False},
            {"id": "wolf2", "role": "werewolf", "alive": False},
        ]
        self.events = [
            {
                "seq": 1,
                "round": 1,
                "phase": "night",
                "event_type": "seer_check",
                "actor_id": "seer",
                "target_id": "wolf1",
                "data": {"result": "werewolves"},
                "visibility": ["seer"],
            },
            {
                "seq": 2,
                "round": 1,
                "phase": "day_discussion",
                "event_type": "speech",
                "actor_id": "seer",
                "message": "我查验 wolf1 是狼人，今天投票给 wolf1",
                "data": {"vote_intent": "wolf1", "claims": [{"target": "wolf1", "alignment": "werewolves"}]},
                "visibility": ["public"],
            },
            {
                "seq": 3,
                "round": 1,
                "phase": "day_discussion",
                "event_type": "speech",
                "actor_id": "v1",
                "data": {"vote_intent": "wolf1"},
                "visibility": ["public"],
            },
            {
                "seq": 4,
                "round": 1,
                "phase": "day_discussion",
                "event_type": "speech",
                "actor_id": "v2",
                "data": {"vote_intent": "wolf2"},
                "visibility": ["public"],
            },
            {"seq": 5, "round": 1, "phase": "day_vote", "event_type": "vote_cast", "actor_id": "v1", "target_id": "wolf1"},
            {"seq": 6, "round": 1, "phase": "day_vote", "event_type": "vote_cast", "actor_id": "v2", "target_id": "wolf1"},
            {"seq": 7, "round": 1, "phase": "night_vote", "event_type": "wolf_vote", "actor_id": "wolf1", "target_id": "v2", "visibility": ["wolf1", "wolf2"]},
            {"seq": 8, "round": 1, "phase": "night_vote", "event_type": "wolf_vote", "actor_id": "wolf2", "target_id": "v2", "visibility": ["wolf1", "wolf2"]},
            {"seq": 9, "round": 1, "phase": "day_result", "event_type": "execution", "target_id": "wolf1"},
            {"seq": 10, "round": 2, "phase": "end", "event_type": "game_over", "data": {"winner": "village"}},
        ]

    def test_core_result_and_process_metrics(self):
        report = evaluate_game(self.events, self.players)

        self.assertEqual(report["result_metrics"]["winner"], "village")
        self.assertEqual(report["result_metrics"]["duration_rounds"], 2)
        self.assertEqual(report["process_metrics"]["information_leakage"]["leak_count"], 0)
        consistency = report["process_metrics"]["speech_vote_consistency"]
        self.assertEqual(consistency["sample_size"], 2)
        self.assertEqual(consistency["rate"], 0.5)
        self.assertEqual(report["process_metrics"]["seer_check_utilization"]["rate"], 1.0)
        self.assertEqual(report["process_metrics"]["wolf_coordination"]["mean_agreement"], 1.0)
        self.assertTrue(any(point["target_id"] == "wolf1" for point in report["turning_points"]))
        self.assertIn("村庄阵营获胜", report["replay"]["summary"])
        self.assertIn("seer", report["player_metrics"])

    def test_information_leakage_and_authorized_reveal(self):
        events = [
            {
                "type": "speech",
                "actor": "v1",
                "data": {"private_info": {"wolf": "w1"}},
                "visibility": "public",
            },
            {
                "type": "role_reveal",
                "target": "w1",
                "data": {"secret_role": "werewolf", "authorized_reveal": True},
                "visibility": "public",
            },
        ]
        leakage = evaluate_game(events, self.players)["process_metrics"]["information_leakage"]
        self.assertEqual(leakage["leak_count"], 1)
        self.assertEqual(leakage["public_event_count"], 2)
        self.assertEqual(leakage["isolation_score"], 0.5)

    def test_whole_game_record_and_missing_fields_are_safe(self):
        report = evaluate_game({"game_id": "g-1", "events": [{}], "players": [{"id": "p"}]})
        self.assertEqual(report["game_id"], "g-1")
        self.assertIsNone(report["result_metrics"]["winner"])
        self.assertIsNone(report["process_metrics"]["speech_vote_consistency"]["rate"])
        self.assertIsNone(report["process_metrics"]["seer_check_utilization"]["rate"])
        self.assertEqual(report["player_metrics"]["p"]["action_count"], 0)

        empty = evaluate_game(None)
        self.assertEqual(empty["result_metrics"]["event_count"], 0)
        self.assertEqual(empty["player_metrics"], {})
        self.assertIsNone(empty["process_metrics"]["information_leakage"]["isolation_score"])

    def test_elo_tracks_model_versions_and_is_zero_sum(self):
        board = EloLeaderboard(k_factor=32)
        delta = board.record_game(
            [
                {"agent": "alpha", "model": "m", "version": "v2", "role": "seer", "team": "village"},
                {"agent": "beta", "model": "m", "version": "v1", "role": "werewolf", "team": "werewolves"},
            ],
            winner="village",
            game_id="g1",
        )
        self.assertAlmostEqual(sum(delta.values()), 0.0)
        ranking = board.rankings()
        self.assertEqual(ranking[0]["key"], "alpha@m:v2")
        self.assertGreater(ranking[0]["rating"], 1500)
        self.assertEqual(ranking[0]["roles"]["seer"]["win_rate"], 1.0)
        with self.assertRaises(ValueError):
            board.record_game([], game_id="g1")

    def test_leaderboard_can_consume_evaluations(self):
        evaluation = evaluate_game(self.events, self.players, game_id="eval-1")
        result = build_leaderboard([evaluation])
        self.assertEqual(result["games_recorded"], 1)
        self.assertEqual(len(result["rankings"]), len(self.players))
        winner_ratings = [item["rating"] for item in result["rankings"] if item["factions"].get("village")]
        loser_ratings = [item["rating"] for item in result["rankings"] if item["factions"].get("werewolves")]
        self.assertGreater(min(winner_ratings), max(loser_ratings))


if __name__ == "__main__":
    unittest.main()
