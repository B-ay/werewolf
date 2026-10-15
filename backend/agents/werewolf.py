"""Werewolf 角色专属目标、中文提示词及本地规划策略。"""

from __future__ import annotations
import random
from copy import deepcopy
from typing import Any, Mapping, Sequence
from .memory import RoleGoal, RoleMemory
from .planner import RolePlanner
from ..game.models import Action, ActionType, Observation, Phase, Role
from .base import BaseAgent


class WerewolfPlanner(RolePlanner):
    """狼人专属规划器：队友协同、夜刀聚合、悍跳与身份威胁清除。

    存活狼人中座位号最小者被稳定选为默认悍跳者。该狼人把伪造查验和警徽流
    存在 ``memory.strategy_state['werewolf_sheriff']``，后续发言复用同一账本，
    避免每轮随机改口；其他狼人不上警，并在警下票阶段支持该悍跳队友。
    """

    role = Role.WEREWOLF

    @staticmethod
    def _wolf_ids(observation: Observation) -> list[str]:
        """从狼人私有队友表中得到按座位排序的存活狼队成员。"""

        actor = observation.viewer_id
        teammates = {
            str(item) for item in observation.private_facts.get("teammates", [])
        }
        teammates.add(actor)
        alive = set(RolePlanner._visible_player_ids(observation, alive_only=True))
        return sorted(teammates & alive, key=RolePlanner._seat_key)

    def _fake_seer_speech(
        self,
        observation: Observation,
        memory: RoleMemory,
    ) -> Action:
        """生成可跨轮复用的假预言家竞选发言。

        首次调用才建立假查验账本和警徽流；后续调用读取长期策略状态，不重新
        抽签，从而让悍跳口径保持一致。假金水只从非狼人玩家中选择。
        """

        actor = observation.viewer_id
        names = {
            str(player["id"]): str(player.get("name", player["id"]))
            for player in observation.players
        }
        wolf_ids = set(self._wolf_ids(observation))
        public_pool = [
            player_id
            for player_id in sorted(
                RolePlanner._visible_player_ids(observation, alive_only=True), key=RolePlanner._seat_key
            )
            if player_id not in wolf_ids
        ]
        state = memory.strategy_state.setdefault("werewolf_sheriff", {})
        ledger = state.setdefault("fake_seer_results", {})
        if not ledger and public_pool:
            # 固定一份“假金水”比每轮重新编造更一致，也不会误把狼队友打成查杀；
            # 后续竞选或 PK 发言都会复用这一账本。
            ledger[public_pool[0]] = "not_werewolf"
        checked_target = next(iter(ledger), None)

        flow = state.get("badge_flow")
        if not isinstance(flow, Mapping):
            future = [target for target in public_pool if target not in ledger][:2]
            if future:
                fallback = next(
                    (
                        target
                        for target in public_pool
                        if target not in future and target not in ledger
                    ),
                    None,
                )
                flow = {
                    "mode": "double" if len(future) == 2 else "single",
                    "targets": future,
                    "fallback_target": fallback,
                }
                state["badge_flow"] = deepcopy(flow)

        metadata: dict[str, Any] = {"claim_role": Role.SEER.value}
        parts = ["我是预言家，参与本轮警长竞选。"]
        if checked_target is not None:
            metadata["defend_target"] = checked_target
            parts.append(
                f"我首夜查验了{names.get(checked_target, checked_target)}，查验结果是好人。"
            )
        if isinstance(flow, Mapping):
            metadata["badge_flow"] = deepcopy(dict(flow))
            labels = [names.get(str(item), str(item)) for item in flow["targets"]]
            parts.append("我的警徽流查验顺序是：" + "，然后查验".join(labels) + "。")
        return Action(
            actor,
            ActionType.SPEAK,
            target_id=checked_target,
            content="".join(parts),
            metadata=metadata,
        )

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        """按“警长事务 → 夜刀 → 发言/放逐票 → 弃权”的优先级规划。"""

        del goal
        legal = RolePlanner._legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        teammates = [str(item) for item in observation.private_facts.get("teammates", [])]
        # 最低存活座次固定担任悍跳者，防止多个狼人同时跳预言家争警徽。
        wolf_ids = self._wolf_ids(observation)
        bluffer = wolf_ids[0] if wolf_ids else actor
        is_bluffer = actor == bluffer
        memory.strategy_state.setdefault("werewolf_sheriff", {})[
            "bluffer_id"
        ] = bluffer

        rules = RolePlanner._legal_rules(observation)
        # 普通警长选择保护队友；到警长投票阶段则给悍跳者一个压倒性支持分。
        sheriff_adjustments: dict[str, float] = {
            teammate: -30.0 for teammate in teammates
        }
        sheriff_excluded: Sequence[str] = teammates
        if ActionType.SHERIFF_VOTE in rules:
            sheriff_adjustments[bluffer] = 100.0
            sheriff_excluded = ()
        sheriff_action = self._plan_sheriff_action(
            observation,
            memory,
            rng,
            run_for_sheriff=is_bluffer,
            stay_in_race=is_bluffer,
            score_adjustments=sheriff_adjustments,
            excluded=sheriff_excluded,
            preferred_badge_targets=sorted(teammates, key=RolePlanner._seat_key),
        )
        if sheriff_action is not None:
            return sheriff_action

        if kind is ActionType.WOLF_KILL:
            # 狼人依次提交封存夜票；观察中若已有本轮可见的 wolf_vote，则对已有
            # 目标加权以促进收敛。公开跳预言家的玩家也会被视为高价值威胁。
            threat = {target: 0.0 for target in targets}
            for event in observation.events:
                if event.get("event_type") == "wolf_vote" and event.get("round") == observation.round:
                    target = (event.get("data") or {}).get("target_id")
                    if target in threat:
                        threat[str(target)] += float(self.parameters.get("coordination_weight", 12.0))
                if event.get("event_type") == "speech" or (
                    self.parameters.get("include_campaign_claims") and event.get("event_type") in {"sheriff_speech", "sheriff_pk_speech"}
                ):
                    data = event.get("data") or {}
                    speaker = event.get("actor_id")
                    if speaker in threat and data.get("claim_role") == Role.SEER.value:
                        threat[str(speaker)] += float(self.parameters.get("seer_threat", 8.0))
            return Action(actor, kind, target_id=RolePlanner._best_target(targets, threat, rng))
        if kind is ActionType.SPEAK:
            if observation.phase in (
                Phase.SHERIFF_SPEECH,
                Phase.SHERIFF_PK_SPEECH,
            ) and is_bluffer:
                return self._fake_seer_speech(observation, memory)
            return self._common_speech(observation, rng, excluded=teammates)
        if kind is ActionType.VOTE:
            adjustments: dict[str, float] = {item: -30.0 for item in teammates}
            for event in observation.events:
                data = event.get("data") or {}
                speaker = event.get("actor_id")
                if speaker in targets and data.get("claim_role") == Role.SEER.value:
                    adjustments[str(speaker)] = adjustments.get(str(speaker), 0.0) + 6.0
            return self._common_vote(observation, rng, score_adjustments=adjustments, excluded=teammates)
        return self._pass(actor)


class WerewolfAgent(BaseAgent):
    """狼人 Agent：独立目标、欺骗提示词、长期记忆和 ``WerewolfPlanner``。"""

    ROLE = Role.WEREWOLF

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "保护狼队存活，并使狼人数量达到与好人阵营的人数平衡。",
            ("与已知狼队友协调夜间击杀目标。", "优先处理威胁狼队的信息角色。", "保持可信的公开立场，避免暴露队友。", "由最低座次存活狼人持续悍跳竞选警长。"),
            ("仅使用可见的公开证据与授权的队友列表。", "夜间不得选择已知狼队友为击杀目标。", "保持假查验和假警徽流声明前后一致。"),
        )

    def _make_prompt(self) -> str:
        return (
            "你扮演狼人。依据private_facts中的队友身份与已提交的狼队夜票协调击杀目标，"
            "公开发言时谨慎伪装。最低座次存活狼人应竞选警长，可以悍跳预言家，"
            "但必须维持一致的假查验账本和假警徽流；其他狼人支持该候选人。"
            "成为警长后合理安排发言顺序、归票目标与传徽策略。不得泄露狼队名单，"
            "不得声称获得观察范围之外的隐藏事实。" + self.CHINESE_COMMUNICATION_RULE
        )

    def _make_planner(self) -> RolePlanner:
        return WerewolfPlanner()
