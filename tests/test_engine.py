import unittest

from backend.game.models import GameConfig, GameStatus, Role
from backend.game.engine import InvalidAction, WerewolfGame


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


class EngineTests(unittest.TestCase):
    def test_full_ai_game_is_deterministic_and_structured(self):
        first = WerewolfGame(GameConfig(seed=31415))
        second = WerewolfGame(GameConfig(seed=31415))

        first_state = first.run()
        second_state = second.run()

        self.assertEqual(first_state["status"], "finished")
        self.assertIn(first_state["winner"], {"village", "werewolves", "draw"})
        self.assertEqual(first.result(), second.result())
        self.assertEqual(first.event_log(omniscient=True), second.event_log(omniscient=True))
        events = first.event_log(omniscient=True)
        self.assertEqual([event["seq"] for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[-1]["event_type"], "game_finished")
        self.assertFalse(any(event["event_type"] == "agent_action_repaired" for event in events))

    def test_full_game_uses_chinese_agent_speech_and_private_feedback(self):
        game = WerewolfGame(GameConfig(roles=ROLES, shuffle_roles=False, seed=31415))
        game.run()
        communication_types = {
            "speech", "sheriff_speech", "sheriff_pk_speech", "sheriff_summary",
            "day_pk_speech", "wolf_vote", "wolf_target", "seer_check",
            "witch_save", "witch_save_passed", "witch_poison", "witch_poison_passed",
        }
        communications = [
            event for event in game.event_log(omniscient=True)
            if event["event_type"] in communication_types
        ]
        self.assertTrue(communications)
        self.assertTrue(any(event["event_type"] == "wolf_vote" for event in communications))
        self.assertTrue(any(event["event_type"] == "sheriff_speech" for event in communications))
        for event in communications:
            with self.subTest(event=event["event_type"], seq=event["seq"]):
                self.assertRegex(event["message"], r"[\u3400-\u9fff]")
        # 交流文案改变不改变私有夜票的可见范围。
        public = game.event_log()
        self.assertFalse(any(event["event_type"] == "wolf_vote" for event in public))

    def test_roles_and_private_events_are_isolated(self):
        game = WerewolfGame(
            GameConfig(roles=ROLES, shuffle_roles=False, seed=1)
        )

        public = game.public_state()
        self.assertTrue(all(player["role"] is None for player in public["players"]))
        self.assertFalse(any(event["event_type"] == "role_assigned" for event in public["events"]))

        wolf = game.public_state("p1")
        roles = {player["id"]: player["role"] for player in wolf["players"]}
        self.assertEqual(roles["p1"], "werewolf")
        self.assertEqual(roles["p2"], "werewolf")
        self.assertIsNone(roles["p3"])
        wolf_private_types = {event["event_type"] for event in wolf["events"]}
        self.assertIn("wolf_team", wolf_private_types)

        seer_observation = game.observe("p3")
        self.assertEqual(seer_observation.self_state["role"], "seer")
        self.assertNotIn("teammates", seer_observation.private_facts)
        self.assertFalse(
            any(event["event_type"] == "wolf_team" for event in seer_observation.events)
        )

    def test_observation_hides_shuffle_seed_and_is_deeply_detached(self):
        game = WerewolfGame(GameConfig(seed=17))
        observation = game.observe("p1")

        started = next(
            event for event in observation.events if event["event_type"] == "game_started"
        )
        self.assertNotIn("seed", started["data"])

        # A policy is untrusted code at the information boundary.  Mutating a
        # nested observation payload must never rewrite the authority log.
        started["data"]["role_counts"]["werewolf"] = 999
        fresh = game.observe("p1")
        fresh_started = next(
            event for event in fresh.events if event["event_type"] == "game_started"
        )
        self.assertNotEqual(fresh_started["data"]["role_counts"]["werewolf"], 999)
        self.assertNotEqual(game.events[0].data["role_counts"]["werewolf"], 999)

        serialized = fresh.to_dict()
        serialized["events"][0]["data"]["role_counts"]["werewolf"] = 777
        self.assertNotEqual(fresh.events[0]["data"]["role_counts"]["werewolf"], 777)

    def test_human_player_can_complete_all_pending_actions(self):
        game = WerewolfGame(
            GameConfig(
                roles=ROLES,
                shuffle_roles=False,
                human_player="p3",
                seed=9,
                max_rounds=8,
            )
        )
        guard = 0
        while game.status is not GameStatus.FINISHED and guard < 2000:
            game.run()
            if game.pending_action:
                pending = game.pending_action
                legal = pending.legal_actions[0]
                payload = {
                    "actor_id": pending.actor_id,
                    "action_type": legal.action_type.value,
                }
                if legal.target_ids:
                    payload["target_id"] = legal.target_ids[0]
                if legal.content_required:
                    payload["content"] = "基于公开信息，我会继续观察票型。"
                game.submit_action(payload)
            guard += 1

        self.assertEqual(game.status, GameStatus.FINISHED)
        self.assertLess(guard, 2000)
        self.assertTrue(any(event.event_type == "action_required" for event in game.events))

    def test_human_illegal_action_is_rejected_without_losing_prompt(self):
        game = WerewolfGame(
            GameConfig(roles=ROLES, shuffle_roles=False, human_player="p1", seed=2)
        )
        game.run()
        self.assertIsNotNone(game.pending_action)
        actor = game.pending_action.actor_id

        with self.assertRaises(InvalidAction):
            game.submit_action(
                {"actor_id": actor, "action_type": "seer_check", "target_id": actor}
            )

        self.assertIsNotNone(game.pending_action)
        self.assertEqual(game.status, GameStatus.WAITING_FOR_HUMAN)

    def test_private_night_human_prompt_is_hidden_from_other_views(self):
        game = WerewolfGame(
            GameConfig(
                roles=ROLES,
                shuffle_roles=False,
                human_player="p3",
                seed=5,
            )
        )
        game.run()
        self.assertEqual(game.phase.value, "night_seer")

        public = game.public_state()
        other_player = game.public_state("p1")
        actor = game.public_state("p3")
        referee = game.public_state(omniscient=True)

        for hidden in (public, other_player):
            self.assertEqual(hidden["status"], "running")
            self.assertIsNone(hidden["pending_action"])
            self.assertEqual(hidden["stage"], "night")
        self.assertEqual(actor["status"], "waiting_for_human")
        self.assertEqual(actor["pending_action"]["actor_id"], "p3")
        self.assertEqual(actor["stage"], "night_seer")
        self.assertEqual(referee["status"], "waiting_for_human")


if __name__ == "__main__":
    unittest.main()
