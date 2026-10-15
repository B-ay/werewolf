"""Witch 角色专属目标、中文提示词及本地规划策略。"""

from __future__ import annotations
import random
from .memory import RoleGoal, RoleMemory
from .planner import RolePlanner
from ..game.models import Action, ActionType, Observation, Role
from .base import BaseAgent


class WitchPlanner(RolePlanner):
    """女巫专属规划器：先处理警长事务，再以保守策略管理两瓶一次性药剂。

    解药只根据引擎提供的 ``night_victim`` 和合法目标使用；毒药要求公开嫌疑
    达到阈值，第三轮后才以有限概率放宽。规划器不自行猜测未公开的夜间事实。
    """

    role = Role.WITCH

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        """根据当前合法阶段选择不上警、救人、毒人、发言或投票。"""

        del goal
        sheriff_action = self._plan_sheriff_action(
            observation,
            memory,
            rng,
            run_for_sheriff=False,
            stay_in_race=False,
        )
        if sheriff_action is not None:
            return sheriff_action
        legal = RolePlanner._legal(observation)
        kind = ActionType.coerce(legal["action_type"])
        actor = observation.viewer_id
        targets = [str(item) for item in legal.get("target_ids", [])]
        if kind is ActionType.WITCH_SAVE:
            # 只有引擎明确告知且目标仍在合法列表中时才使用解药。
            victim = observation.private_facts.get("night_victim")
            if victim and str(victim) in targets:
                return Action(actor, kind, target_id=str(victim))
            return self._pass(actor)
        if kind is ActionType.WITCH_POISON:
            # 前两轮坚持证据阈值；后期为避免永久囤药，允许可复现 RNG 小概率出毒。
            suspicion = self._event_suspicion(observation, self.parameters)
            target = RolePlanner._best_target(targets, suspicion, rng)
            score = suspicion.get(target or "", 0.0)
            if target and (
                score >= float(self.parameters.get("poison_threshold", 2.0))
                or (observation.round >= 3 and rng.random() < float(self.parameters.get("late_poison_probability", 0.45)))
            ):
                return Action(actor, kind, target_id=target)
            return self._pass(actor)
        if kind is ActionType.SPEAK:
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            return self._common_vote(observation, rng)
        return self._pass(actor)


class WitchAgent(BaseAgent):
    """女巫 Agent：独立目标、药剂管理提示词、长期记忆和 ``WitchPlanner``。"""

    ROLE = Role.WITCH

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "合理使用解药与毒药，提高好人阵营的存活和获胜机会。",
            ("持续记录两瓶一次性药剂的使用状态。", "解药可用时优先救助有价值的生命。", "仅在可见证据足够充分时使用毒药。"),
            ("仅使用private_facts中授权的night_victim与药剂状态。", "不得把私有夜间信息直接当作已公开的确定事实。"),
        )

    def _make_prompt(self) -> str:
        return (
            "你扮演女巫。将药剂视为稀缺资源，结合后续回合安排用药。仅使用提供的"
            "night_victim和potions状态，依据累积的可见证据决定毒人。通常不上警；"
            "若当选警长，依据公开证据安排发言顺序、归票和传徽或撕徽。"
            + self.CHINESE_COMMUNICATION_RULE
        )

    def _make_planner(self) -> RolePlanner:
        return WitchPlanner()
