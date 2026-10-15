"""Villager 角色专属目标、中文提示词及本地规划策略。"""

from __future__ import annotations
import random
from .memory import RoleGoal, RoleMemory
from .planner import RolePlanner
from ..game.models import Action, ActionType, Observation, Role
from .base import BaseAgent


class VillagerPlanner(RolePlanner):
    """村民专属规划器：只用公开发言和票型保持言行一致。"""

    role = Role.VILLAGER

    def plan(self, observation: Observation, memory: RoleMemory, goal: RoleGoal, rng: random.Random) -> Action:
        """规划不上警、普通发言和放逐投票，不引入任何神职私有能力。"""

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
        kind = ActionType.coerce(RolePlanner._legal(observation)["action_type"])
        actor = observation.viewer_id
        if kind is ActionType.SPEAK:
            return self._common_speech(observation, rng)
        if kind is ActionType.VOTE:
            return self._common_vote(observation, rng)
        return self._pass(actor)


class VillagerAgent(BaseAgent):
    """村民 Agent：独立目标、公开推理提示词、长期记忆和 ``VillagerPlanner``。"""

    ROLE = Role.VILLAGER

    def _make_goal(self) -> RoleGoal:
        return RoleGoal(
            self.role,
            "帮助好人阵营识别并放逐所有狼人。",
            ("比较公开身份主张、指控和言票一致性。", "清晰解释怀疑依据，并随新证据调整判断。", "投票给公开证据最充分的怀疑对象。"),
            ("仅依据公开信息和自己的身份推理。", "不得编造任何神职查验或技能结果。"),
        )

    def _make_prompt(self) -> str:
        return (
            "你扮演村民，没有私有查验能力。依据公开发言、身份主张和投票行为推理，"
            "发言须与普通村民的可知信息一致。通常不上警；若当选警长，仅依据公开"
            "证据安排发言顺序、归票，并决定传徽或撕徽。" + self.CHINESE_COMMUNICATION_RULE
        )

    def _make_planner(self) -> RolePlanner:
        return VillagerPlanner()
