"""评测精简的日志契约、事件索引和重构前结果基线。"""
import copy
import hashlib
import unittest

from backend.evaluation import evaluate_game
from backend.evaluation.common import _canonical_events
from backend.evaluation.tournament import run_tournament
from backend.game.engine import WerewolfGame
from backend.game.models import GameConfig, Role


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class EvaluationContractTests(unittest.TestCase):
    def test_fixed_seed_evaluations_match_original_full_report(self):
        # 摘要包含报告的全部字段、列表顺序和指标，不只检查胜负或综合分。
        cases = [
            (5, False, "8b9ab8cef058c8f65b16944381ca4fa386f6f2277b491e62151f480de0e93944"),
            (5, True, "9cf185eee2646958cb98098dd67f15e2533ad3d395c8cc85de16e61d43dfb5c4"),
            (42, False, "921a84bde3199a4edf61efe61e73c0c8692eb723db037cb03c31c48e023cb485"),
            (42, True, "e2ed3b6024e0c6d02238a112ba58266c2ddff411008d15fc7728584df4f00a82"),
            (31415, False, "5e098724bd081aeeda8abecd6757f123fb374e9290e4fd65f619803d0b6360d0"),
            (31415, True, "91e3bc4abe45fe7c2ef7e0e621b42e5ec0b28ce58c874ff55ac751bbd6844565"),
        ]
        for seed, sheriff, expected in cases:
            with self.subTest(seed=seed, sheriff=sheriff):
                game = WerewolfGame(GameConfig(seed=seed, enable_sheriff=sheriff))
                state = game.run()
                report = evaluate_game(game.event_log(omniscient=True), state, game_id="baseline")
                self.assertEqual(digest(report), expected)

    def test_tournament_summary_and_elo_match_original_full_report(self):
        cases = [
            (False, "4d6d243535774f8a0678611df10b7314c4045433f057ef21b607b0f4fd78a21b"),
            (True, "cd19cf23dd1188863eda15b75b0a8411d1e6bffc90cf161ac60018e878da56b1"),
        ]
        for sheriff, expected in cases:
            with self.subTest(sheriff=sheriff):
                report = run_tournament(
                    (5, 42), games_per_seed=2,
                    game_config={"player_count": 8, "enable_sheriff": sheriff},
                    competitors=["alpha:model-a:v1", "beta:model-b:v2"],
                    tournament_id="baseline",
                )
                self.assertEqual(digest(report), expected)

    def test_event_index_separates_channels_without_mutating_input(self):
        events = [
            {"seq": 1, "round": 1, "event_type": "sheriff_vote_cast", "actor_id": "p1", "data": {"abstained": True}},
            {"seq": 2, "round": 1, "event_type": "wolf_vote", "actor_id": "p1", "target_id": "p2"},
            {"seq": 3, "round": 1, "phase": "day_vote", "event_type": "vote_cast", "actor_id": "p1", "target_id": "p2"},
            {"seq": 4, "round": 1, "phase": "day_pk_vote", "event_type": "vote_cast", "actor_id": "p1", "target_id": "p2"},
        ]
        before = copy.deepcopy(events)
        index = _canonical_events(events)
        self.assertEqual([event.seq for event in index.groups["wolf_vote"]], [2])
        self.assertEqual([event.seq for event in index.groups["exile_vote"]], [3, 4])
        self.assertEqual([event.seq for event in index.groups["primary_exile_vote"]], [3])
        self.assertEqual(index.groups["sheriff_ballot"], [])
        self.assertEqual(len(index.by_actor["p1"]), 4)
        evaluate_game(events, [{"id": "p1", "role": "werewolf"}, {"id": "p2", "role": "villager"}])
        self.assertEqual(events, before)

    def test_external_formats_and_roles_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "event_type"):
            evaluate_game([{"type": "speech"}])
        with self.assertRaisesRegex(ValueError, "requires events"):
            evaluate_game({"event_log": []})
        with self.assertRaisesRegex(TypeError, "iterable"):
            evaluate_game("external-log")
        with self.assertRaisesRegex(ValueError, "player id"):
            evaluate_game([], [{"name": "missing-id"}])
        with self.assertRaisesRegex(ValueError, "unsupported role"):
            evaluate_game([], [{"id": "p1", "role": "guard"}])
        with self.assertRaisesRegex(ValueError, "integers"):
            evaluate_game([{"event_type": "speech", "round": "Day 1"}])

    def test_unknown_event_names_are_not_classified_by_substring(self):
        report = evaluate_game([
            {"event_type": "external_vote", "actor_id": "p1", "target_id": "p2"},
            {"event_type": "external_death", "target_id": "p2"},
        ], [{"id": "p1", "role": "villager"}, {"id": "p2", "role": "werewolf"}])
        self.assertEqual(report["process_metrics"]["exile_vote_accuracy"]["ballot_count"], 0)
        self.assertEqual(report["turning_points"], [])

    def test_missing_samples_and_old_replay_fields_stay_compatible(self):
        self.assertEqual(evaluate_game({})["result_metrics"]["event_count"], 0)
        report = evaluate_game({"events": [], "state": {"players": [{"id": "p1", "role": Role.VILLAGER, "alive": None}]}})
        self.assertEqual(report["schema_version"], "1.1")
        self.assertIsNone(report["process_metrics"]["sheriff"]["badge_flow"]["execution_rate"])
        self.assertIsNone(report["process_metrics"]["speech_vote_consistency"]["rate"])
        self.assertIsNone(report["player_metrics"]["p1"]["vote_accuracy"])
        self.assertFalse(report["process_metrics"]["sheriff"]["election"]["held"])

    def test_domain_event_and_player_objects_use_explicit_contract(self):
        game = WerewolfGame(GameConfig(seed=11))
        game.run()
        # Domain objects and serialized snapshots must produce exactly the same result.
        self.assertEqual(
            evaluate_game(game.events, game.players, game.winner),
            evaluate_game(game.event_log(omniscient=True), game.result()["players"], game.winner),
        )

    def test_compatibility_aliases_are_removed(self):
        from backend.evaluation import evaluator, leaderboard
        self.assertFalse(hasattr(evaluator, "GameEvaluator"))
        self.assertFalse(hasattr(evaluator, "analyze_game"))
        self.assertFalse(hasattr(leaderboard, "Leaderboard"))
        self.assertFalse(hasattr(leaderboard, "aggregate_leaderboard"))
        self.assertFalse(hasattr(leaderboard.EloLeaderboard, "add_game"))


if __name__ == "__main__":
    unittest.main()
