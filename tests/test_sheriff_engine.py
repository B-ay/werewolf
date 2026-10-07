import unittest

from backend import ActionType, GameConfig, Phase, Role, WerewolfGame
from backend.models import LegalAction


ROLES = [
    Role.WEREWOLF,
    Role.WEREWOLF,
    Role.SEER,
    Role.WITCH,
    Role.HUNTER,
    Role.VILLAGER,
    Role.VILLAGER,
    Role.VILLAGER,
]


class SheriffEngineTests(unittest.TestCase):
    def make_game(self, *, seed: int = 11, **overrides) -> WerewolfGame:
        options = {
            "roles": ROLES,
            "shuffle_roles": False,
            "seed": seed,
            "human_players": list(range(1, 9)),
            "enable_sheriff": True,
            "max_rounds": 5,
        }
        options.update(overrides)
        return WerewolfGame(
            GameConfig(**options)
        )

    def pending(self, game: WerewolfGame):
        game.run()
        self.assertIsNotNone(game.pending_action)
        return game.pending_action

    def act(
        self,
        game: WerewolfGame,
        action_type: ActionType | str,
        target_id: str | None = None,
        *,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        pending = self.pending(game)
        payload = {
            "actor_id": pending.actor_id,
            "action_type": (
                action_type.value if isinstance(action_type, ActionType) else action_type
            ),
        }
        if target_id is not None:
            payload["target_id"] = target_id
        if content is not None:
            payload["content"] = content
        if metadata is not None:
            payload["metadata"] = metadata
        game.submit_action(payload)

    def finish_night(
        self,
        game: WerewolfGame,
        *,
        wolf_target: str = "p8",
        save: bool = False,
        poison_target: str | None = None,
    ) -> None:
        self.act(game, ActionType.WOLF_KILL, wolf_target)
        self.act(game, ActionType.WOLF_KILL, wolf_target)
        self.act(game, ActionType.SEER_CHECK, "p1")
        self.act(
            game,
            ActionType.WITCH_SAVE if save else ActionType.PASS,
            wolf_target if save else None,
        )
        if not save:
            self.act(
                game,
                ActionType.WITCH_POISON if poison_target else ActionType.PASS,
                poison_target,
            )

    def run_for_sheriff(self, game: WerewolfGame, candidates: set[str]) -> None:
        for player_id in game.players:
            pending = self.pending(game)
            self.assertEqual(pending.actor_id, player_id)
            self.act(
                game,
                ActionType.SHERIFF_RUN if player_id in candidates else ActionType.PASS,
            )

    def finish_campaign_speeches(self, game: WerewolfGame) -> list[str]:
        order: list[str] = []
        while game.phase in (Phase.SHERIFF_SPEECH, Phase.SHERIFF_PK_SPEECH):
            pending = self.pending(game)
            order.append(pending.actor_id)
            self.act(game, ActionType.SPEAK, content=f"campaign {pending.actor_id}")
        return order

    def keep_all_candidates(self, game: WerewolfGame) -> None:
        while game.phase is Phase.SHERIFF_WITHDRAW:
            self.act(game, ActionType.PASS)

    def elect_single(self, game: WerewolfGame, player_id: str) -> None:
        self.run_for_sheriff(game, {player_id})
        self.finish_campaign_speeches(game)
        self.keep_all_candidates(game)
        self.assertEqual(game.phase, Phase.DAWN_REVEAL)
        self.assertEqual(game.public_state()["sheriff"]["holder_id"], player_id)

    def elect_single_seer(self, game: WerewolfGame) -> None:
        self.elect_single(game, "p3")

    def test_void_election_paths_and_complete_abstention_reveal(self):
        no_runner = self.make_game(seed=12)
        self.finish_night(no_runner, save=True)
        self.run_for_sheriff(no_runner, set())
        self.assertEqual(no_runner.phase, Phase.DAWN_REVEAL)
        self.assertEqual(no_runner.public_state()["sheriff"]["badge_status"], "void")
        no_runner_void = [
            event for event in no_runner.events if event.event_type == "sheriff_election_void"
        ][-1]
        self.assertEqual(no_runner_void.data["reason_code"], "no_candidates")

        all_withdraw = self.make_game(seed=13)
        self.finish_night(all_withdraw, save=True)
        self.run_for_sheriff(all_withdraw, {"p1", "p2"})
        self.finish_campaign_speeches(all_withdraw)
        self.act(all_withdraw, ActionType.SHERIFF_WITHDRAW)
        # Withdrawal choices stay sealed until the complete batch closes.
        self.assertEqual(all_withdraw.public_state()["sheriff"]["withdrawn"], [])
        self.act(all_withdraw, ActionType.SHERIFF_WITHDRAW)
        self.assertEqual(all_withdraw.phase, Phase.DAWN_REVEAL)
        self.assertEqual(
            all_withdraw.public_state()["sheriff"]["withdrawn"], ["p1", "p2"]
        )
        all_withdraw_void = [
            event
            for event in all_withdraw.events
            if event.event_type == "sheriff_election_void"
        ][-1]
        self.assertEqual(all_withdraw_void.data["reason_code"], "all_withdrew")

        abstained = self.make_game(seed=14)
        self.finish_night(abstained, save=True)
        self.run_for_sheriff(abstained, {"p1", "p2"})
        self.finish_campaign_speeches(abstained)
        self.keep_all_candidates(abstained)
        voters = ["p3", "p4", "p5", "p6", "p7", "p8"]
        for voter in voters:
            self.assertEqual(self.pending(abstained).actor_id, voter)
            self.act(abstained, ActionType.PASS)
        result = [
            event for event in abstained.events if event.event_type == "sheriff_vote_result"
        ][-1]
        self.assertEqual(result.data["votes"], {voter: None for voter in voters})
        self.assertEqual(result.data["tally"], {})
        void = [
            event for event in abstained.events if event.event_type == "sheriff_election_void"
        ][-1]
        self.assertEqual(void.data["reason_code"], "no_valid_ballots")

    def test_sealed_choices_are_referee_only_and_hidden_batch_stays_auditable(self):
        game = self.make_game(seed=17, reveal_vote_choices=False)
        self.finish_night(game, save=True)

        self.act(game, ActionType.SHERIFF_RUN)
        public_types = {event["event_type"] for event in game.public_state()["events"]}
        player_types = {event["event_type"] for event in game.observe("p2").events}
        referee_types = {
            event["event_type"] for event in game.event_log(omniscient=True)
        }
        self.assertNotIn("sheriff_run_choice", public_types)
        self.assertNotIn("sheriff_run_choice", player_types)
        self.assertIn("sheriff_run_choice", referee_types)
        for player_id in list(game.players)[1:]:
            self.act(
                game,
                ActionType.SHERIFF_RUN if player_id == "p2" else ActionType.PASS,
            )
        self.finish_campaign_speeches(game)
        self.keep_all_candidates(game)

        # Only original police-off players may vote, and an individual sealed
        # ballot cannot be observed by the next voter.
        self.assertEqual(self.pending(game).actor_id, "p3")
        self.act(game, ActionType.SHERIFF_VOTE, "p1")
        self.assertFalse(
            any(event["event_type"] == "sheriff_vote_cast" for event in game.observe("p4").events)
        )
        self.assertTrue(
            any(
                event["event_type"] == "sheriff_vote_cast"
                and event["actor_id"] == "p3"
                for event in game.event_log(omniscient=True)
            )
        )
        for voter in ("p4", "p5", "p6", "p7", "p8"):
            self.assertEqual(self.pending(game).actor_id, voter)
            self.act(game, ActionType.SHERIFF_VOTE, "p1")
        public_result = next(
            event
            for event in reversed(game.public_state()["events"])
            if event["event_type"] == "sheriff_vote_result"
        )
        self.assertNotIn("votes", public_result["data"])
        public_events = game.public_state()["events"]
        self.assertEqual(
            [event["seq"] for event in public_events],
            list(range(1, len(public_events) + 1)),
        )

    def test_pending_first_night_sheriff_dies_then_cannot_badge_same_batch_victim(self):
        game = self.make_game(seed=21)
        self.finish_night(game, wolf_target="p3", poison_target="p5")
        self.elect_single_seer(game)

        before_reveal = game.public_state(omniscient=True)
        self.assertTrue(next(p for p in before_reveal["players"] if p["id"] == "p3")["alive"])
        self.assertTrue(next(p for p in before_reveal["players"] if p["id"] == "p5")["alive"])
        self.assertNotIn("pending_dawn_deaths", before_reveal)
        game.step()

        pending = self.pending(game)
        self.assertEqual(game.phase, Phase.SHERIFF_BADGE)
        self.assertEqual(pending.actor_id, "p3")
        transfer = next(
            legal
            for legal in pending.legal_actions
            if legal.action_type is ActionType.SHERIFF_TRANSFER
        )
        self.assertNotIn("p3", transfer.target_ids)
        self.assertNotIn("p5", transfer.target_ids)
        self.assertFalse(game.players["p3"].alive)
        self.assertFalse(game.players["p5"].alive)

    def test_counterclockwise_order_repeats_and_daily_state_is_reset_before_choice(self):
        game = self.make_game(seed=27, discussion_rounds=2)
        self.finish_night(game, wolf_target="p8", save=True)
        self.elect_single_seer(game)
        game.step()
        self.act(game, ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE)
        expected = ["p2", "p1", "p8", "p7", "p6", "p5", "p4", "p3"]
        self.assertEqual(game.public_state()["sheriff"]["day_speech_order"], expected)

        observed: list[str] = []
        for _ in range(2):
            for speaker in expected:
                self.assertEqual(self.pending(game).actor_id, speaker)
                observed.append(speaker)
                if speaker == "p3":
                    self.act(
                        game,
                        ActionType.SHERIFF_SUMMARIZE,
                        "p1",
                        content="daily summary",
                    )
                else:
                    self.act(game, ActionType.SPEAK, content="discussion")
        self.assertEqual(observed, expected + expected)
        self.assertEqual(
            len([event for event in game.events if event.event_type == "sheriff_summary"]),
            2,
        )

        # Simulate arrival at the following dawn: yesterday's public order is
        # cleared before the living sheriff is asked to choose today's order.
        game.round = 2
        game._start_day()
        state = game.public_state()["sheriff"]
        self.assertEqual(game.phase, Phase.SHERIFF_ORDER)
        self.assertIsNone(state["day_direction"])
        self.assertEqual(state["day_speech_order"], [])

    def test_sheriff_hunter_handles_badge_before_final_shot_when_exiled(self):
        game = self.make_game(seed=28)
        self.finish_night(game, wolf_target="p8", save=True)
        self.elect_single(game, "p5")
        game.step()
        self.act(game, ActionType.SHERIFF_ORDER_CLOCKWISE)
        order = ["p6", "p7", "p8", "p1", "p2", "p3", "p4", "p5"]
        for speaker in order:
            if speaker == "p5":
                self.act(
                    game,
                    ActionType.SHERIFF_SUMMARIZE,
                    "p1",
                    content="summary",
                )
            else:
                self.act(game, ActionType.SPEAK, content="discussion")
        for voter in game.players:
            self.assertEqual(self.pending(game).actor_id, voter)
            self.act(game, ActionType.VOTE, "p1" if voter == "p5" else "p5")

        self.assertEqual(game.phase, Phase.SHERIFF_BADGE)
        self.assertEqual(self.pending(game).actor_id, "p5")
        self.act(game, ActionType.SHERIFF_TRANSFER, "p6")
        self.assertEqual(game.phase, Phase.HUNTER_SHOT)
        self.assertEqual(self.pending(game).actor_id, "p5")
        event_types = [event.event_type for event in game.events]
        self.assertLess(
            event_types.index("badge_transfer_required"),
            event_types.index("hunter_triggered"),
        )

    def test_metadata_player_references_and_badge_flow_are_sanitized(self):
        game = self.make_game(seed=30)
        self.finish_night(game, wolf_target="p8", save=True)
        self.run_for_sheriff(game, {"p3"})
        self.act(
            game,
            ActionType.SPEAK,
            content="campaign",
            metadata={
                "claim_role": "seer",
                "accuse_target": "p999",
                "defend_target": "p2",
                "vote_intent": "ghost",
                "badge_flow": {
                    "mode": "double",
                    "targets": ["p6", "p7"],
                    "fallback_target": "p8",
                },
            },
        )
        speech = [event for event in game.events if event.event_type == "sheriff_speech"][-1]
        self.assertEqual(speech.data["claim_role"], "seer")
        self.assertEqual(speech.data["defend_target"], "p2")
        self.assertNotIn("accuse_target", speech.data)
        self.assertNotIn("vote_intent", speech.data)
        self.assertEqual(
            speech.data["badge_flow"],
            {
                "mode": "double",
                "targets": ["p6", "p7"],
                "fallback_target": "p8",
            },
        )
        invalid = game._sanitize_speech_metadata(
            {"badge_flow": {"mode": "double", "targets": ["p1", "p2", "p3"]}}
        )
        self.assertNotIn("badge_flow", invalid)

    def test_safe_fallback_is_valid_for_every_new_sheriff_action(self):
        game = self.make_game(seed=32)
        rules = (
            LegalAction(ActionType.SHERIFF_RUN, can_pass=True),
            LegalAction(ActionType.SHERIFF_WITHDRAW, can_pass=True),
            LegalAction(ActionType.SHERIFF_VOTE, ("p2",), can_pass=True),
            LegalAction(ActionType.SHERIFF_ORDER_CLOCKWISE),
            LegalAction(ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE),
            LegalAction(
                ActionType.SHERIFF_SUMMARIZE,
                ("p2",),
                content_required=True,
            ),
            LegalAction(ActionType.SHERIFF_TRANSFER, ("p2",)),
            LegalAction(ActionType.SHERIFF_DESTROY),
        )
        for rule in rules:
            with self.subTest(action_type=rule.action_type.value):
                fallback = game._safe_fallback("p1", (rule,))
                game._validate_action(fallback, (rule,), "p1")
                if rule.content_required:
                    self.assertTrue(fallback.content and fallback.content.strip())

    def test_election_choices_are_sealed_and_two_ties_void_badge(self):
        game = self.make_game(seed=19)
        self.finish_night(game, wolf_target="p8")

        # A pending first-night casualty remains fully alive and no dawn result
        # exists during the complete election.
        self.assertEqual(self.pending(game).actor_id, "p1")
        self.act(game, ActionType.SHERIFF_RUN)
        state = game.public_state()
        self.assertEqual(state["sheriff"]["candidates"], [])
        self.assertTrue(next(p for p in state["players"] if p["id"] == "p8")["alive"])
        self.assertFalse(any(e["event_type"] == "night_result" for e in state["events"]))

        for player_id in list(game.players)[1:]:
            self.assertEqual(self.pending(game).actor_id, player_id)
            self.act(
                game,
                ActionType.SHERIFF_RUN if player_id == "p2" else ActionType.PASS,
            )
        first_order = self.finish_campaign_speeches(game)
        self.assertCountEqual(first_order, ["p1", "p2"])
        self.keep_all_candidates(game)

        voters = ["p3", "p4", "p5", "p6", "p7", "p8"]
        for index, voter in enumerate(voters):
            self.assertEqual(self.pending(game).actor_id, voter)
            self.act(game, ActionType.SHERIFF_VOTE, "p1" if index < 3 else "p2")
        expected_pk = [pid for pid in reversed(first_order) if pid in {"p1", "p2"}]
        self.assertEqual(game.public_state()["sheriff"]["pk_candidates"], expected_pk)
        self.assertEqual(self.finish_campaign_speeches(game), expected_pk)
        for index, voter in enumerate(voters):
            self.assertEqual(self.pending(game).actor_id, voter)
            self.act(game, ActionType.SHERIFF_VOTE, "p1" if index < 3 else "p2")

        self.assertEqual(game.phase, Phase.DAWN_REVEAL)
        self.assertEqual(game.public_state()["sheriff"]["badge_status"], "void")
        self.assertTrue(game.players["p8"].alive)
        game.step()
        self.assertFalse(game.players["p8"].alive)

    def test_single_candidate_order_summary_and_decisive_one_point_five_vote(self):
        game = self.make_game(seed=23)
        self.finish_night(game, wolf_target="p8")
        self.elect_single_seer(game)
        game.step()  # reveal the delayed first-night death

        self.assertEqual(self.pending(game).actor_id, "p3")
        self.act(game, ActionType.SHERIFF_ORDER_CLOCKWISE)
        expected_order = ["p4", "p5", "p6", "p7", "p1", "p2", "p3"]
        self.assertEqual(game.public_state()["sheriff"]["day_speech_order"], expected_order)
        for speaker in expected_order:
            self.assertEqual(self.pending(game).actor_id, speaker)
            if speaker == "p3":
                self.act(
                    game,
                    ActionType.SHERIFF_SUMMARIZE,
                    "p1",
                    content="I nominate p1.",
                    metadata={"sheriff_nomination": "p1"},
                )
            else:
                self.act(game, ActionType.SPEAK, content=f"speech {speaker}")

        votes = {
            "p1": "p2",
            "p2": "p1",
            "p3": "p1",
            "p4": "p1",
            "p5": "p2",
            "p6": "p2",
            "p7": "p4",
        }
        for voter, target in votes.items():
            self.assertEqual(self.pending(game).actor_id, voter)
            self.act(game, ActionType.VOTE, target)

        result = [e for e in game.events if e.event_type == "day_vote_result"][-1]
        self.assertEqual(result.target_id, "p1")
        self.assertEqual(result.data["tally_units"]["p1"], 7)
        self.assertEqual(result.data["unweighted_tally_units"]["p1"], 6)
        self.assertEqual(result.data["unweighted_tally_units"]["p2"], 6)
        self.assertTrue(result.data["sheriff_vote_decisive"])

    def test_day_runoff_uses_reverse_speech_order_and_second_tie_executes_nobody(self):
        game = self.make_game(seed=29)
        self.finish_night(game, wolf_target="p8", save=True)
        self.elect_single_seer(game)
        game.step()
        self.act(game, ActionType.SHERIFF_ORDER_CLOCKWISE)
        day_order = ["p4", "p5", "p6", "p7", "p8", "p1", "p2", "p3"]
        for speaker in day_order:
            if speaker == "p3":
                self.act(
                    game,
                    ActionType.SHERIFF_SUMMARIZE,
                    "p1",
                    content="runoff nomination",
                )
            else:
                self.act(game, ActionType.SPEAK, content="discussion")

        first_votes = {
            "p1": "p3",
            "p2": "p1",
            "p3": "p2",
            "p4": "p3",
            "p5": "p3",
            "p6": "p1",
            "p7": "p1",
            "p8": "p2",
        }
        for voter, target in first_votes.items():
            self.assertEqual(self.pending(game).actor_id, voter)
            self.act(game, ActionType.VOTE, target)

        self.assertEqual(game.phase, Phase.DAY_PK_SPEECH)
        self.assertEqual(game.public_state()["sheriff"]["pk_candidates"], [])
        pk_order: list[str] = []
        while game.phase is Phase.DAY_PK_SPEECH:
            pending = self.pending(game)
            pk_order.append(pending.actor_id)
            self.act(game, ActionType.SPEAK, content="pk")
        self.assertEqual(pk_order, ["p3", "p1"])

        expected_voters = ["p2", "p4", "p5", "p6", "p7", "p8"]
        for index, voter in enumerate(expected_voters):
            self.assertEqual(self.pending(game).actor_id, voter)
            self.act(game, ActionType.VOTE, "p1" if index < 3 else "p3")
        self.assertEqual(game.round, 2)
        self.assertTrue(all(player.alive for player in game.players.values()))
        second = [e for e in game.events if e.event_type == "day_vote_result"][-1]
        self.assertEqual(second.data["round"], 2)
        self.assertIsNone(second.data["eliminated"])

    def test_badge_transfer_precedes_hunter_and_chains_when_new_sheriff_is_shot(self):
        game = self.make_game(seed=31)
        # The sheriff is poisoned while the hunter is wolf-killed; badge
        # handling must still run before the hunter's eligible final shot.
        self.finish_night(game, wolf_target="p5", poison_target="p3")
        self.elect_single_seer(game)
        game.step()

        self.assertEqual(self.pending(game).actor_id, "p3")
        self.assertEqual(game.phase, Phase.SHERIFF_BADGE)
        self.act(game, ActionType.SHERIFF_TRANSFER, "p6")
        self.assertEqual(self.pending(game).actor_id, "p5")
        self.assertEqual(game.phase, Phase.HUNTER_SHOT)
        self.act(game, ActionType.HUNTER_SHOOT, "p6")
        self.assertEqual(self.pending(game).actor_id, "p6")
        self.assertEqual(game.phase, Phase.SHERIFF_BADGE)
        self.act(game, ActionType.SHERIFF_DESTROY)

        event_types = [event.event_type for event in game.events]
        first_badge = event_types.index("badge_transfer_required")
        hunter = event_types.index("hunter_triggered")
        shot = event_types.index("hunter_shot")
        second_badge = event_types.index("badge_transfer_required", first_badge + 1)
        self.assertLess(first_badge, hunter)
        self.assertLess(hunter, shot)
        self.assertLess(shot, second_badge)
        self.assertEqual(game.public_state()["sheriff"]["badge_status"], "destroyed")


if __name__ == "__main__":
    unittest.main()
