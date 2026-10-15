"""Seer 角色专属目标、中文提示词及本地规划策略。"""

from __future__ import annotations
import random
from copy import deepcopy
from typing import Any, Mapping, Optional
from .memory import RoleGoal, RoleMemory
from .planner import RolePlanner
from ..game.models import Action, ActionType, Observation, Phase, Role
from .base import BaseAgent


class SeerPlanner(RolePlanner):
    """预言家专属规划器：查验、真结果披露、警徽流和证据驱动投票。

    引擎提供的 ``private_facts['seer_results']`` 是唯一可信查验源。预言家默认
    上警且不退水；首次竞选时建立一/双夜警徽流，并优先按该计划查验。死亡
    处理警徽时，再根据已完成结果编码“传给谁或撕毁”。
    """

    role = Role.SEER

    @staticmethod
    def _badge_flow(
        observation: Observation,
        memory: RoleMemory,
    ) -> Optional[dict[str, Any]]:
        """读取或首次建立警徽流计划，并把它持久化到角色策略记忆。"""

        state = memory.strategy_state.setdefault("seer_sheriff", {})
        # 已声明的警徽流必须保持稳定，不能因下一轮嫌疑变化而悄悄改写。
        existing = state.get("badge_flow")
        if isinstance(existing, Mapping):
            return deepcopy(dict(existing))

        results = {
            str(target): str(result)
            for target, result in (
                observation.private_facts.get("seer_results") or {}
            ).items()
        }
        candidates = [
            player_id
            for player_id in sorted(
                RolePlanner._visible_player_ids(observation, alive_only=True), key=RolePlanner._seat_key
            )
            if player_id != observation.viewer_id and player_id not in results
        ]
        if memory.experience.get("parameters", {}).get("suspect_checks"):
            scores = RolePlanner._event_suspicion(observation, memory.experience["parameters"])
            candidates.sort(key=lambda player_id: -scores.get(player_id, 0.0))
        targets = candidates[:2]
        if not targets:
            return None
        fallback = next(
            (
                player_id
                for player_id in sorted(
                    RolePlanner._visible_player_ids(observation, alive_only=True), key=RolePlanner._seat_key
                )
                if player_id != observation.viewer_id
                and player_id not in targets
                and player_id not in results
            ),
            None,
        )
        flow = {
            "mode": "double" if len(targets) == 2 else "single",
            "targets": targets,
            "fallback_target": fallback,
        }
        state["badge_flow"] = deepcopy(flow)
        return flow

    @staticmethod
    def _badge_recipient(
        observation: Observation,
        memory: RoleMemory,
    ) -> Optional[str]:
        """按已完成的真实查验解码警徽接收者；``None`` 表示应撕徽。

        单夜或只完成第一验：第一目标为好人则传给他，否则撕徽。双验完成后：
        一好一狼传给好人；两好撕徽；两狼传给预先声明的外置位。
        """

        flow = memory.strategy_state.get("seer_sheriff", {}).get("badge_flow")
        if not isinstance(flow, Mapping):
            return None
        targets = [str(item) for item in flow.get("targets") or []]
        if not targets:
            return None
        results = {
            str(target): str(result)
            for target, result in (
                observation.private_facts.get("seer_results") or {}
            ).items()
        }

        first = targets[0]
        if first not in results:
            return None
        first_good = results[first] != Role.WEREWOLF.value
        if len(targets) == 1 or targets[1] not in results:
            return first if first_good else None

        second = targets[1]
        second_good = results[second] != Role.WEREWOLF.value
        if first_good != second_good:
            return first if first_good else second
        if first_good and second_good:
            return None
        fallback = flow.get("fallback_target")
        return str(fallback) if fallback is not None else None

    def _campaign_speech(
        self,
        observation: Observation,
        memory: RoleMemory,
    ) -> Action:
        """公开真实查验和既定警徽流，生成预言家竞选/PK 发言。"""

        actor = observation.viewer_id
        names = {
            str(player["id"]): str(player.get("name", player["id"]))
            for player in observation.players
        }
        results = {
            str(target): str(result)
            for target, result in (
                observation.private_facts.get("seer_results") or {}
            ).items()
        }
        metadata: dict[str, Any] = {"claim_role": Role.SEER.value}
        parts = ["我是预言家，参与本轮警长竞选。"]
        target: Optional[str] = None
        if results:
            # 保留引擎插入顺序，让首夜查验先于后续查验出现在竞选陈述中。
            target, result = next(iter(results.items()))
            if result == Role.WEREWOLF.value:
                metadata.update(
                    {"accuse_target": target, "vote_intent": target}
                )
                parts.append(
                    f"我查验了{names.get(target, target)}，查验结果是狼人。"
                )
            else:
                metadata["defend_target"] = target
                parts.append(
                    f"我查验了{names.get(target, target)}，查验结果是好人。"
                )
        flow = self._badge_flow(observation, memory)
        if flow is not None:
            metadata["badge_flow"] = deepcopy(flow)
            labels = [names.get(str(item), str(item)) for item in flow["targets"]]
            parts.append("我的警徽流查验顺序是：" + "，然后查验".join(labels) + "。")
        return Action(
            actor,
            ActionType.SPEAK,
            target_id=target,
            content="".join(parts),
            metadata=metadata,
        )

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        """按“警徽处理 → 竞选事务 → 查验 → 发言/投票 → 弃权”规划。"""

        del goal
        legal = RolePlanner._legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        results = observation.private_facts.get("seer_results") or {}
        # 真查杀在嫌疑模型中获得强正权重，真金水获得负权重，覆盖纯社交判断。
        adjustments = {
            str(target): (20.0 if result == Role.WEREWOLF.value else -8.0)
            for target, result in results.items()
        }

        rules = RolePlanner._legal_rules(observation)
        # 死亡后的警徽处理需要先计算警徽流编码；其他警长阶段使用同一公共规划。
        if (
            ActionType.SHERIFF_TRANSFER in rules
            or ActionType.SHERIFF_DESTROY in rules
        ):
            recipient = self._badge_recipient(observation, memory)
            sheriff_action = self._plan_sheriff_action(
                observation,
                memory,
                rng,
                run_for_sheriff=True,
                stay_in_race=True,
                score_adjustments=adjustments,
                preferred_badge_targets=([recipient] if recipient else ()),
                destroy_badge=True,
            )
        else:
            sheriff_action = self._plan_sheriff_action(
                observation,
                memory,
                rng,
                run_for_sheriff=True,
                stay_in_race=True,
                score_adjustments=adjustments,
            )
        if sheriff_action is not None:
            return sheriff_action

        if kind is ActionType.SEER_CHECK:
            # 优先兑现已公开的警徽流；目标已查验或不可选时才退回未查验座位。
            unchecked = [target for target in targets if target not in results]
            flow = memory.strategy_state.get("seer_sheriff", {}).get("badge_flow")
            if isinstance(flow, Mapping):
                for planned_target in flow.get("targets") or ():
                    planned_target = str(planned_target)
                    if planned_target in unchecked:
                        return Action(actor, kind, target_id=planned_target)
            pool = sorted(unchecked or targets)
            if self.parameters.get("suspect_checks"):
                target = RolePlanner._best_target(pool, self._event_suspicion(observation, self.parameters), rng)
                return Action(actor, kind, target_id=target)
            return Action(actor, kind, target_id=rng.choice(pool) if pool else None)
        if kind is ActionType.SPEAK:
            if observation.phase in (
                Phase.SHERIFF_SPEECH,
                Phase.SHERIFF_PK_SPEECH,
            ):
                return self._campaign_speech(observation, memory)
            names = {str(player["id"]): str(player["name"]) for player in observation.players}
            wolves = sorted(str(pid) for pid, result in results.items() if result == Role.WEREWOLF.value)
            if wolves:
                target = wolves[0]
                return Action(
                    actor,
                    kind,
                    target_id=target,
                    content=f"我是预言家，查验了{names.get(target, target)}，结果是狼人。我建议本轮投票放逐对方。",
                    metadata={"claim_role": Role.SEER.value, "accuse_target": target, "vote_intent": target},
                )
            if results:
                target = sorted(str(pid) for pid in results)[-1]
                return Action(
                    actor,
                    kind,
                    target_id=target,
                    content=f"我是预言家，对{names.get(target, target)}的查验结果是好人。",
                    metadata={"claim_role": Role.SEER.value, "defend_target": target},
                )
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            return self._common_vote(observation, rng, score_adjustments=adjustments)
        return self._pass(actor)


class SeerAgent(BaseAgent):
    """预言家 Agent：独立目标、真实查验提示词、长期记忆和 ``SeerPlanner``。"""

    ROLE = Role.SEER

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "利用真实查验帮助好人阵营找出并放逐所有狼人。",
            ("每夜查验尚未查过且有信息价值的玩家。", "保存查验结果，并选择合适时机公开。", "积极竞选警长，声明单双夜警徽流，死亡传徽时编码已完成的查验结果。", "依据已证实的狼人和好人结果投票。"),
            ("仅将seer_results作为真实的私有查验结果。", "不得编造查验结果。", "警徽流只声明未来查验目标；私有查验结果须通过合法的公开发言披露。"),
        )

    def _make_prompt(self) -> str:
        return (
            "你扮演预言家。依据seer_results维护真实查验账本，优先查验尚未查过的玩家，"
            "区分已验证的事实与仅依据发言产生的怀疑。积极上警并坚持竞选，公开真实"
            "且已完成的查验，声明稳定的单夜或双夜警徽流。死亡时根据已完成的查验"
            "选择传徽或撕徽。不得把尚未进行的查验当作结果。" + self.CHINESE_COMMUNICATION_RULE
        )

    def _make_planner(self) -> RolePlanner:
        return SeerPlanner()
