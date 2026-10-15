"""Hunter 角色专属目标、中文提示词及本地规划策略。"""

from __future__ import annotations
import random
from .memory import RoleGoal, RoleMemory
from .planner import RolePlanner
from ..game.models import Action, ActionType, Observation, Role
from .base import BaseAgent


class HunterPlanner(RolePlanner):
    """猎人专属规划器：公开信息推理，以及死亡时的证据化开枪。"""

    role = Role.HUNTER

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        """优先处理警徽，再从合法射击目标中选择公开嫌疑最高者。"""

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
        if kind is ActionType.HUNTER_SHOOT:
            scores = self._event_suspicion(observation, self.parameters)
            target = RolePlanner._best_target(targets, scores, rng)
            threshold = self.parameters.get("shoot_threshold")
            if threshold is not None and scores.get(target or "", 0.0) < threshold:
                return self._pass(actor)
            return Action(actor, kind, target_id=target) if target else self._pass(actor)
        if kind is ActionType.SPEAK:
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            return self._common_vote(observation, rng)
        return self._pass(actor)


class HunterAgent(BaseAgent):
    """猎人 Agent：独立目标、开枪约束提示词、长期记忆和 ``HunterPlanner``。"""

    ROLE = Role.HUNTER

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "帮助好人阵营推理，并在死亡开枪时选择证据充分的目标。",
            ("依据公开发言和票型形成一致的判断。", "开枪时优先选择证据最强的怀疑对象。", "缺乏证据且不开枪更稳妥时保留开枪权。"),
            ("不得暗示掌握公开证据之外的隐藏身份信息。", "只从当前合法目标中选择开枪对象。"),
        )

    def _make_prompt(self) -> str:
        return (
            "你扮演猎人。存活时依据公开证据推理，维护嫌疑排序，为可能触发的死亡"
            "开枪做准备。通常不上警；若当选警长，依据公开信息行使警长能力，"
            "死亡时先处理警徽，再决定开枪。不得声称拥有查验能力或隐藏身份信息。"
            + self.CHINESE_COMMUNICATION_RULE
        )

    def _make_planner(self) -> RolePlanner:
        return HunterPlanner()
