import unittest

from backend.game.models import ActionType, GameConfig, Role
from backend.game.engine import WerewolfGame
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
            {"seq": 9, "round": 1, "phase": "day_result", "event_type": "death", "target_id": "wolf1"},
            {"seq": 10, "round": 2, "phase": "finished", "event_type": "game_finished", "data": {"winner": "village"}},
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
                "event_type": "speech",
                "actor": "v1",
                "data": {"private_info": {"wolf": "w1"}},
                "visibility": "public",
            },
            {
                "event_type": "death",
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
        report = evaluate_game({"game_id": "g-1", "events": [{"event_type": "phase_started"}], "players": [{"id": "p"}]})
        self.assertEqual(report["schema_version"], "1.1")
        self.assertEqual(report["game_id"], "g-1")
        self.assertIsNone(report["result_metrics"]["winner"])
        self.assertIsNone(report["process_metrics"]["speech_vote_consistency"]["rate"])
        self.assertIsNone(report["process_metrics"]["seer_check_utilization"]["rate"])
        self.assertEqual(report["player_metrics"]["p"]["action_count"], 0)

        empty = evaluate_game(None)
        self.assertEqual(empty["result_metrics"]["event_count"], 0)
        self.assertEqual(empty["player_metrics"], {})
        self.assertIsNone(empty["process_metrics"]["information_leakage"]["isolation_score"])

    def test_sheriff_metrics_and_vote_channels_are_isolated(self):
        events = [
            {
                "seq": 1,
                "round": 1,
                "phase": "night_seer",
                "event_type": "seer_check",
                "actor_id": "seer",
                "target_id": "v1",
                "data": {"result": "village"},
                "visibility": ["seer"],
            },
            {"seq": 2, "round": 1, "phase": "sheriff_run", "event_type": "sheriff_election_started"},
            {
                "seq": 3,
                "round": 1,
                "phase": "sheriff_run",
                "event_type": "sheriff_candidacy_result",
                "data": {"candidates": ["seer", "wolf1"], "speech_order": ["wolf1", "seer"]},
            },
            {
                "seq": 4,
                "round": 1,
                "phase": "sheriff_speech",
                "event_type": "sheriff_speech",
                "actor_id": "seer",
                "data": {
                    "vote_intent": "wolf1",
                    "badge_flow": {
                        "mode": "double",
                        "targets": ["v1", "wolf1"],
                        "fallback_target": "v2",
                    },
                },
            },
            {
                "seq": 5,
                "round": 1,
                "phase": "sheriff_vote",
                "event_type": "sheriff_vote_cast",
                "actor_id": "v1",
                "target_id": "seer",
            },
            {
                "seq": 6,
                "round": 1,
                "phase": "sheriff_vote",
                "event_type": "sheriff_vote_result",
                "data": {"round": 1, "tally": {"seer": 2}, "votes": {"v1": "seer", "v2": "seer"}},
            },
            {
                "seq": 7,
                "round": 1,
                "phase": "dawn_reveal",
                "event_type": "sheriff_elected",
                "actor_id": "seer",
                "data": {"holder_id": "seer"},
            },
            {
                "seq": 8,
                "round": 1,
                "phase": "sheriff_order",
                "event_type": "sheriff_order",
                "actor_id": "seer",
                "data": {"direction": "clockwise", "order": ["v1", "wolf1", "v2", "seer"]},
            },
            {
                "seq": 9,
                "round": 1,
                "phase": "day_discussion",
                "event_type": "sheriff_summary",
                "actor_id": "seer",
                "target_id": "wolf1",
                "data": {"sheriff_nomination": "wolf1"},
            },
            {
                "seq": 10,
                "round": 1,
                "phase": "day_vote",
                "event_type": "vote_cast",
                "actor_id": "seer",
                "target_id": "wolf1",
            },
            {
                "seq": 11,
                "round": 1,
                "phase": "day_vote",
                "event_type": "vote_cast",
                "actor_id": "v1",
                "target_id": "wolf1",
            },
            {
                "seq": 12,
                "round": 1,
                "phase": "day_vote",
                "event_type": "vote_cast",
                "actor_id": "v2",
                "target_id": "wolf2",
            },
            {
                "seq": 13,
                "round": 1,
                "phase": "day_vote",
                "event_type": "day_vote_result",
                "target_id": "wolf1",
                "data": {
                    "sheriff_id": "seer",
                    "tally_units": {"wolf1": 5, "wolf2": 4},
                    "unweighted_tally_units": {"wolf1": 4, "wolf2": 4},
                    "sheriff_vote_decisive": True,
                    "eliminated": "wolf1",
                },
            },
            {
                "seq": 14,
                "round": 2,
                "phase": "night_vote",
                "event_type": "wolf_vote",
                "actor_id": "wolf1",
                "target_id": "seer",
                "visibility": ["wolf1", "wolf2"],
            },
            {
                "seq": 15,
                "round": 2,
                "phase": "night_vote",
                "event_type": "wolf_vote",
                "actor_id": "wolf2",
                "target_id": "seer",
                "visibility": ["wolf1", "wolf2"],
            },
            {
                "seq": 16,
                "round": 2,
                "phase": "night_seer",
                "event_type": "seer_check",
                "actor_id": "seer",
                "target_id": "wolf1",
                "data": {"result": "werewolves"},
                "visibility": ["seer"],
            },
            {"seq": 17, "round": 2, "phase": "dawn_reveal", "event_type": "death", "target_id": "seer"},
            {
                "seq": 18,
                "round": 2,
                "phase": "sheriff_badge",
                "event_type": "badge_transferred",
                "actor_id": "seer",
                "target_id": "v1",
            },
        ]
        report = evaluate_game(events, self.players)

        self.assertEqual(report["schema_version"], "1.1")
        self.assertEqual(report["process_metrics"]["exile_vote_accuracy"]["sample_size"], 3)
        self.assertEqual(report["process_metrics"]["wolf_coordination"]["night_attack"]["mean_agreement"], 1.0)
        self.assertEqual(report["process_metrics"]["wolf_coordination"]["sheriff_election"]["rounds_evaluated"], 0)
        election = report["process_metrics"]["sheriff_election"]
        self.assertEqual(election["winner_id"], "seer")
        self.assertTrue(election["seer_elected"])
        leadership = report["process_metrics"]["sheriff_leadership"]
        self.assertEqual(leadership["decisive_weighted_vote_count"], 1)
        self.assertEqual(leadership["badge_transfer_history"][-1]["to_id"], "v1")
        self.assertAlmostEqual(leadership["nomination_response_rate"], 2 / 3, places=3)
        badge_flow = report["process_metrics"]["badge_flow_execution"]
        self.assertEqual(badge_flow["execution_rate"], 1.0)
        self.assertEqual(report["player_metrics"]["seer"]["sheriff_decisive_vote_count"], 1)

    def test_old_replay_without_sheriff_fields_remains_compatible(self):
        report = evaluate_game(self.events, self.players)
        election = report["process_metrics"]["sheriff_election"]
        self.assertFalse(election["held"])
        self.assertIsNone(election["winner_id"])
        self.assertEqual(report["process_metrics"]["badge_flow_execution"]["declaration_count"], 0)

    def test_hidden_sheriff_ballots_and_exile_pk_are_separate_channels(self):
        events = [
            {"seq": 1, "round": 1, "phase": "sheriff_run", "event_type": "sheriff_election_started"},
            {
                "seq": 2,
                "round": 1,
                "phase": "sheriff_run",
                "event_type": "sheriff_candidacy_result",
                "data": {"candidates": ["seer", "v1"], "speech_order": ["v1", "seer"]},
            },
            {
                "seq": 3,
                "round": 1,
                "phase": "sheriff_vote",
                "event_type": "sheriff_vote_cast",
                "actor_id": "wolf1",
                "target_id": "seer",
                "data": {"round": 1, "abstained": False},
                "visibility": ["observer"],
            },
            {
                "seq": 4,
                "round": 1,
                "phase": "sheriff_vote",
                "event_type": "sheriff_vote_cast",
                "actor_id": "wolf2",
                "target_id": "seer",
                "data": {"round": 1, "abstained": False},
                "visibility": ["observer"],
            },
            {
                "seq": 5,
                "round": 1,
                "phase": "sheriff_vote",
                "event_type": "sheriff_vote_cast",
                "actor_id": "v2",
                "data": {"round": 1, "abstained": True},
                "visibility": ["observer"],
            },
            {
                "seq": 6,
                "round": 1,
                "phase": "sheriff_vote",
                "event_type": "sheriff_vote_result",
                # reveal_vote_choices=false: only the public aggregate exists.
                "data": {"round": 1, "tally": {"seer": 2}, "tied": ["seer"]},
            },
            {
                "seq": 7,
                "round": 1,
                "phase": "sheriff_vote",
                "event_type": "sheriff_elected",
                "actor_id": "seer",
                "data": {"holder_id": "seer", "automatic": False},
            },
            {
                "seq": 8,
                "round": 1,
                "phase": "day_discussion",
                "event_type": "sheriff_summary",
                "actor_id": "seer",
                "target_id": "wolf1",
                "data": {"sheriff_nomination": "wolf1"},
            },
            {"seq": 9, "round": 1, "phase": "day_vote", "event_type": "vote_cast", "actor_id": "seer", "target_id": "wolf1", "visibility": ["observer"]},
            {"seq": 10, "round": 1, "phase": "day_vote", "event_type": "vote_cast", "actor_id": "v1", "target_id": "wolf1", "visibility": ["observer"]},
            {"seq": 11, "round": 1, "phase": "day_vote", "event_type": "vote_cast", "actor_id": "v2", "target_id": "wolf2", "visibility": ["observer"]},
            {
                "seq": 12,
                "round": 1,
                "phase": "day_vote",
                "event_type": "day_vote_result",
                "data": {"round": 1, "tally_units": {"wolf1": 4, "wolf2": 4}, "tied": ["wolf1", "wolf2"]},
            },
            {"seq": 13, "round": 1, "phase": "day_vote", "event_type": "day_pk_started", "data": {"candidates": ["wolf2", "wolf1"]}},
            {"seq": 14, "round": 1, "phase": "day_pk_speech", "event_type": "day_pk_speech", "actor_id": "wolf2", "data": {"vote_intent": "wolf1"}},
            {"seq": 15, "round": 1, "phase": "day_pk_speech", "event_type": "day_pk_speech", "actor_id": "wolf1", "data": {"vote_intent": "wolf2"}},
            {"seq": 16, "round": 1, "phase": "day_pk_vote", "event_type": "vote_cast", "actor_id": "seer", "target_id": "wolf2", "visibility": ["observer"]},
            {"seq": 17, "round": 1, "phase": "day_pk_vote", "event_type": "vote_cast", "actor_id": "v1", "target_id": "wolf2", "visibility": ["observer"]},
            {
                "seq": 18,
                "round": 1,
                "phase": "day_pk_vote",
                "event_type": "day_vote_result",
                "target_id": "wolf2",
                "data": {"round": 2, "tally_units": {"wolf2": 5}, "eliminated": "wolf2"},
            },
        ]
        report = evaluate_game(events, self.players)
        election = report["process_metrics"]["sheriff_election"]
        self.assertEqual(election["speech_order"], ["v1", "seer"])
        self.assertEqual(election["valid_ballot_count"], 2)
        self.assertEqual(election["cast_ballot_count"], 3)
        self.assertEqual(election["abstention_count"], 1)
        self.assertEqual(
            report["process_metrics"]["wolf_coordination"]["sheriff_election"]["mean_agreement"],
            1.0,
        )
        # Both normal and PK exile ballots are evaluated as exile choices.
        self.assertEqual(report["process_metrics"]["exile_vote_accuracy"]["ballot_count"], 5)
        # The PK is constrained and must not be compared with the sheriff's
        # original nomination or counted as a second nomination response.
        consistency = report["process_metrics"]["speech_vote_consistency"]
        self.assertEqual(consistency["sample_size"], 1)
        leadership = report["process_metrics"]["sheriff_leadership"]
        self.assertEqual(leadership["nomination_response_sample_size"], 3)
        self.assertEqual(leadership["nomination_followed_count"], 2)

    def test_actual_engine_hidden_sheriff_log_is_evaluable(self):
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
                human_players=list(range(1, 9)),
                seed=71,
                enable_sheriff=True,
                reveal_vote_choices=False,
            )
        )

        def act(action_type, target_id=None, *, content=None):
            game.run()
            pending = game.pending_action
            self.assertIsNotNone(pending)
            payload = {
                "actor_id": pending.actor_id,
                "action_type": action_type.value,
            }
            if target_id is not None:
                payload["target_id"] = target_id
            if content is not None:
                payload["content"] = content
            game.submit_action(payload)

        # Complete night one without a casualty, then run p3/p4 so the two
        # wolves remain in the original police-off electorate.
        act(ActionType.WOLF_KILL, "p8")
        act(ActionType.WOLF_KILL, "p8")
        act(ActionType.SEER_CHECK, "p1")
        act(ActionType.WITCH_SAVE, "p8")
        for player_id in game.players:
            act(ActionType.SHERIFF_RUN if player_id in {"p3", "p4"} else ActionType.PASS)
        act(ActionType.SPEAK, content="campaign")
        act(ActionType.SPEAK, content="campaign")
        act(ActionType.PASS)
        act(ActionType.PASS)
        for voter in ("p1", "p2", "p5", "p6", "p7", "p8"):
            game.run()
            self.assertEqual(game.pending_action.actor_id, voter)
            act(ActionType.SHERIFF_VOTE, "p3")

        public_result = next(
            event
            for event in reversed(game.public_state()["events"])
            if event["event_type"] == "sheriff_vote_result"
        )
        self.assertNotIn("votes", public_result["data"])
        state = game.public_state(omniscient=True)
        report = evaluate_game(
            game.event_log(omniscient=True),
            state["players"],
            state.get("winner"),
        )
        election = report["process_metrics"]["sheriff_election"]
        self.assertEqual(election["winner_id"], "p3")
        self.assertEqual(election["valid_ballot_count"], 6)
        self.assertEqual(election["cast_ballot_count"], 6)
        coordination = report["process_metrics"]["wolf_coordination"]
        self.assertEqual(coordination["night_attack"]["mean_agreement"], 1.0)
        self.assertEqual(coordination["sheriff_election"]["mean_agreement"], 1.0)
        self.assertEqual(report["process_metrics"]["exile_vote_accuracy"]["ballot_count"], 0)

    def test_badge_flow_single_double_partial_and_dead_receiver(self):
        def flow_report(events):
            return evaluate_game(events, self.players)["process_metrics"]["badge_flow_execution"]

        scenarios = {
            "single_dead_good_receiver": (
                [
                    {"seq": 1, "round": 1, "phase": "night_seer", "event_type": "seer_check", "actor_id": "seer", "target_id": "v1", "data": {"result": "not_werewolf"}, "visibility": ["seer"]},
                    {"seq": 2, "round": 1, "phase": "sheriff_speech", "event_type": "sheriff_speech", "actor_id": "seer", "data": {"badge_flow": {"mode": "single", "targets": ["v1"]}}},
                    {"seq": 3, "round": 1, "phase": "sheriff_vote", "event_type": "sheriff_elected", "actor_id": "seer", "data": {"holder_id": "seer"}},
                    {"seq": 4, "round": 2, "phase": "dawn_reveal", "event_type": "death", "target_id": "v1"},
                    {"seq": 5, "round": 2, "phase": "dawn_reveal", "event_type": "death", "target_id": "seer"},
                    {"seq": 6, "round": 2, "phase": "sheriff_badge", "event_type": "badge_destroyed", "actor_id": "seer"},
                ],
                "destroy",
                None,
                "intended_receiver_dead",
            ),
            "double_only_first_check": (
                [
                    {"seq": 1, "round": 1, "phase": "night_seer", "event_type": "seer_check", "actor_id": "seer", "target_id": "v1", "data": {"result": "not_werewolf"}, "visibility": ["seer"]},
                    {"seq": 2, "round": 1, "phase": "sheriff_speech", "event_type": "sheriff_speech", "actor_id": "seer", "data": {"badge_flow": {"mode": "double", "targets": ["v1", "wolf1"], "fallback_target": "v2"}}},
                    {"seq": 3, "round": 1, "phase": "sheriff_vote", "event_type": "sheriff_elected", "actor_id": "seer", "data": {"holder_id": "seer"}},
                    {"seq": 4, "round": 2, "phase": "dawn_reveal", "event_type": "death", "target_id": "seer"},
                    {"seq": 5, "round": 2, "phase": "sheriff_badge", "event_type": "badge_transferred", "actor_id": "seer", "target_id": "v1"},
                ],
                "transfer",
                "v1",
                "single_good",
            ),
            "double_wolves_dead_fallback": (
                [
                    {"seq": 1, "round": 1, "phase": "sheriff_speech", "event_type": "sheriff_speech", "actor_id": "seer", "data": {"badge_flow": {"mode": "double", "targets": ["wolf1", "wolf2"], "fallback_target": "v1"}}},
                    {"seq": 2, "round": 1, "phase": "night_seer", "event_type": "seer_check", "actor_id": "seer", "target_id": "wolf1", "data": {"result": "werewolf"}, "visibility": ["seer"]},
                    {"seq": 3, "round": 2, "phase": "night_seer", "event_type": "seer_check", "actor_id": "seer", "target_id": "wolf2", "data": {"result": "werewolf"}, "visibility": ["seer"]},
                    {"seq": 4, "round": 2, "phase": "dawn_reveal", "event_type": "death", "target_id": "v1"},
                    {"seq": 5, "round": 2, "phase": "dawn_reveal", "event_type": "death", "target_id": "seer"},
                    {"seq": 6, "round": 2, "phase": "sheriff_badge", "event_type": "badge_destroyed", "actor_id": "seer"},
                ],
                "destroy",
                None,
                "intended_receiver_dead",
            ),
            "double_wolves_without_fallback": (
                [
                    {"seq": 1, "round": 1, "phase": "sheriff_speech", "event_type": "sheriff_speech", "actor_id": "seer", "data": {"badge_flow": {"mode": "double", "targets": ["wolf1", "wolf2"]}}},
                    {"seq": 2, "round": 1, "phase": "night_seer", "event_type": "seer_check", "actor_id": "seer", "target_id": "wolf1", "data": {"result": "werewolf"}, "visibility": ["seer"]},
                    {"seq": 3, "round": 2, "phase": "night_seer", "event_type": "seer_check", "actor_id": "seer", "target_id": "wolf2", "data": {"result": "werewolf"}, "visibility": ["seer"]},
                    {"seq": 4, "round": 2, "phase": "dawn_reveal", "event_type": "death", "target_id": "seer"},
                    {"seq": 5, "round": 2, "phase": "sheriff_badge", "event_type": "badge_destroyed", "actor_id": "seer"},
                ],
                "destroy",
                None,
                "double_both_wolves_without_fallback",
            ),
        }
        for name, (events, action, target, reason) in scenarios.items():
            with self.subTest(name=name):
                metric = flow_report(events)
                self.assertEqual(metric["evaluable_count"], 1)
                self.assertEqual(metric["execution_rate"], 1.0)
                detail = metric["declarations"][0]
                self.assertEqual(detail["expected_action"], action)
                self.assertEqual(detail["expected_target"], target)
                self.assertEqual(detail["expected_reason"], reason)

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
