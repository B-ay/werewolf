"""共享本地规划操作：发言、投票、警长定序、归票和警徽处理。"""

from __future__ import annotations
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence
from .memory import RoleGoal, RoleMemory
from ..game.models import Action, ActionType, Observation, Role


class RolePlanner(ABC):
    """五类角色本地规划器的公共基类。

    规划器不访问网络，也不依赖大模型。它读取当前观察、长期记忆、角色目标和
    由引擎注入的随机数生成器，按当前合法动作阶段生成基线 ``Action``。公共的
    发言、投票、警长竞选/归票/传徽逻辑在这里复用，角色差异由子类叠加。
    """

    role: Role

    def __init__(self) -> None:
        # 参数是每个规划器独占的版本快照；对局进行中不会被后台晋级修改。
        self.parameters: dict[str, float | bool] = {}

    @staticmethod
    def _legal(observation: Observation) -> dict[str, Any]:
        """取得当前动作规则；多数阶段只会提供一类合法动作。"""

        if not observation.legal_actions:
            raise ValueError("agent was asked to act without a legal action")
        return observation.legal_actions[0]

    @staticmethod
    def _legal_rules(observation: Observation) -> dict[ActionType, dict[str, Any]]:
        """按动作类型索引全部合法规则，兼容警长方向等多选阶段。"""

        rules: dict[ActionType, dict[str, Any]] = {}
        for rule in observation.legal_actions:
            try:
                kind = ActionType.coerce(rule.get("action_type"))
            except (TypeError, ValueError):
                continue
            rules[kind] = rule
        return rules

    @staticmethod
    def _seat_key(player_id: str) -> tuple[int, int | str]:
        """按座位数字排序 p12 等常规 ID；其他 ID 使用稳定的字符串顺序。"""

        match = re.fullmatch(r"p(\d+)", str(player_id), flags=re.IGNORECASE)
        if match:
            return (0, int(match.group(1)))
        return (1, str(player_id))

    @staticmethod
    def _visible_player_ids(
        observation: Observation, *, alive_only: bool = False
    ) -> list[str]:
        return [
            str(player["id"])
            for player in observation.players
            if player.get("id") is not None
            and (not alive_only or bool(player.get("alive")))
        ]

    @staticmethod
    def _suspicion_from_events(
        events: Sequence[Mapping[str, Any]],
        parameters: Optional[Mapping[str, float | bool]] = None,
    ) -> dict[str, float]:
        """从可见公开发言构建可解释的嫌疑分数。

        这不是机器学习模型，而是一组确定性的启发式权重：预言家声明的指控权重
        更高，保护会降低嫌疑，警长归票会显著提高目标嫌疑。角色规划器可再叠加
        私有且合法的角色证据，例如预言家的查验结果。
        """

        params = parameters or {}
        scores: dict[str, float] = {}
        public_speech_types = {
            "speech",
            "sheriff_speech",
            "sheriff_pk_speech",
            "sheriff_summary",
            "day_pk_speech",
        }
        for event in events:
            if event.get("event_type") not in public_speech_types:
                continue
            data = event.get("data") or {}
            if not isinstance(data, Mapping):
                continue
            claimed_seer = data.get("claim_role") == Role.SEER.value
            target = data.get("accuse_target")
            if target:
                weight = float(params.get("seer_accuse_weight", 3.0) if claimed_seer else params.get("accuse_weight", 1.0))
                scores[str(target)] = scores.get(str(target), 0.0) + weight

            defended = data.get("defend_target")
            if defended:
                weight = float(params.get("seer_defend_weight", 2.0) if claimed_seer else params.get("defend_weight", 0.75))
                scores[str(defended)] = scores.get(str(defended), 0.0) - weight

            # A Sheriff summary is the formal consolidation recommendation.  It
            # should influence baseline voters, while role-private evidence can
            # still override it through each planner's score adjustments.
            nomination = data.get("sheriff_nomination")
            if nomination:
                scores[str(nomination)] = scores.get(str(nomination), 0.0) + float(params.get("nomination_weight", 4.0))
        return scores

    @staticmethod
    def _event_suspicion(observation: Observation, parameters: Optional[Mapping[str, float | bool]] = None) -> dict[str, float]:
        return RolePlanner._suspicion_from_events(observation.events, parameters)

    @staticmethod
    def _best_target(
        candidates: Sequence[str], scores: Mapping[str, float], rng: random.Random
    ) -> Optional[str]:
        if not candidates:
            return None
        best_score = max((scores.get(item, 0.0) for item in candidates), default=0.0)
        tied = sorted(item for item in candidates if scores.get(item, 0.0) == best_score)
        return rng.choice(tied)

    @abstractmethod
    def plan(
        self,
        observation: Observation,
        memory: RoleMemory,
        goal: RoleGoal,
        rng: random.Random,
    ) -> Action:
        """为对应角色返回一个本地基线动作。"""

    def _common_speech(
        self,
        observation: Observation,
        rng: random.Random,
        *,
        excluded: Sequence[str] = (),
    ) -> Action:
        """依据公开嫌疑分生成普通发言，并同步填写指控和投票意向 metadata。"""

        actor = observation.viewer_id
        legal = RolePlanner._legal(observation)
        targets = [str(item) for item in legal.get("target_ids", [])]
        excluded_set = set(excluded)
        candidates = [
            target for target in targets if target != actor and target not in excluded_set
        ]
        candidates = candidates or [target for target in targets if target != actor]
        suspicion = self._event_suspicion(observation, self.parameters)
        target = RolePlanner._best_target(candidates, suspicion, rng)
        names = {str(player["id"]): str(player["name"]) for player in observation.players}
        if target:
            content = (
                f"我目前比较怀疑{names.get(target, target)}。我希望对方进一步解释自己的立场；"
                "如果没有新的证据改变我的判断，我建议本轮投票放逐对方。"
            )
            metadata = {"accuse_target": target, "vote_intent": target}
        else:
            content = "我目前还没有明确的怀疑对象，会继续比较大家的发言和票型。"
            metadata = {}
        return Action(
            actor,
            ActionType.SPEAK,
            target_id=target,
            content=content,
            metadata=metadata,
        )

    def _common_vote(
        self,
        observation: Observation,
        rng: random.Random,
        *,
        score_adjustments: Optional[Mapping[str, float]] = None,
        excluded: Sequence[str] = (),
    ) -> Action:
        """在公开嫌疑分上叠加角色修正，再从最高分合法目标中投票。"""

        actor = observation.viewer_id
        legal = RolePlanner._legal(observation)
        targets = [str(item) for item in legal.get("target_ids", [])]
        excluded_set = set(excluded)
        preferred = [target for target in targets if target not in excluded_set] or targets
        scores = self._event_suspicion(observation, self.parameters)
        for target, amount in (score_adjustments or {}).items():
            scores[str(target)] = scores.get(str(target), 0.0) + float(amount)
        target = RolePlanner._best_target(preferred, scores, rng)
        return Action(actor, ActionType.VOTE, target_id=target)

    @staticmethod
    def _pass(actor: str) -> Action:
        return Action(actor, ActionType.PASS)

    def _ranked_target(
        self,
        observation: Observation,
        rule: Mapping[str, Any],
        rng: random.Random,
        *,
        score_adjustments: Optional[Mapping[str, float]] = None,
        excluded: Sequence[str] = (),
        prefer_low_score: bool = False,
    ) -> Optional[str]:
        """对指定动作的合法目标排序；同分时用对局 RNG 做可复现破局。"""

        targets = [str(item) for item in rule.get("target_ids", [])]
        excluded_set = {str(item) for item in excluded}
        pool = [target for target in targets if target not in excluded_set] or targets
        if not pool:
            return None
        scores = self._event_suspicion(observation, self.parameters)
        for target, amount in (score_adjustments or {}).items():
            scores[str(target)] = scores.get(str(target), 0.0) + float(amount)
        extreme = (
            min(scores.get(target, 0.0) for target in pool)
            if prefer_low_score
            else max(scores.get(target, 0.0) for target in pool)
        )
        tied = sorted(
            target for target in pool if scores.get(target, 0.0) == extreme
        )
        return rng.choice(tied)

    def _sheriff_order(
        self,
        observation: Observation,
        *,
        score_adjustments: Optional[Mapping[str, float]] = None,
    ) -> Action:
        """选择发言方向，使高嫌疑玩家尽量更早发言、警长最后总结。

        分别模拟顺/逆时针顺序，并对越靠前的位置赋予越高权重；两边得分相同
        时稳定选择顺时针。
        """

        actor = observation.viewer_id
        rules = RolePlanner._legal_rules(observation)
        clockwise = ActionType.SHERIFF_ORDER_CLOCKWISE
        counterclockwise = ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE
        if clockwise not in rules:
            return Action(actor, counterclockwise)
        if counterclockwise not in rules:
            return Action(actor, clockwise)

        alive = sorted(
            RolePlanner._visible_player_ids(observation, alive_only=True), key=RolePlanner._seat_key
        )
        if actor not in alive or len(alive) <= 2:
            return Action(actor, clockwise)
        anchor = alive.index(actor)
        cw_order = alive[anchor + 1 :] + alive[:anchor]
        ccw_order = list(reversed(cw_order))
        suspicion = self._event_suspicion(observation, self.parameters)
        for target, amount in (score_adjustments or {}).items():
            suspicion[str(target)] = suspicion.get(str(target), 0.0) + float(amount)

        def order_score(order: Sequence[str]) -> float:
            size = len(order)
            return sum(
                (size - index) * suspicion.get(player_id, 0.0)
                for index, player_id in enumerate(order)
            )

        selected = counterclockwise if order_score(ccw_order) > order_score(cw_order) else clockwise
        return Action(actor, selected)

    def _plan_sheriff_action(
        self,
        observation: Observation,
        memory: RoleMemory,
        rng: random.Random,
        *,
        run_for_sheriff: bool,
        stay_in_race: bool,
        score_adjustments: Optional[Mapping[str, float]] = None,
        excluded: Sequence[str] = (),
        preferred_badge_targets: Sequence[str] = (),
        destroy_badge: bool = False,
    ) -> Optional[Action]:
        """统一规划所有角色共享的警长竞选、归票、排序和警徽处理。

        返回 ``None`` 表示当前阶段不是警长相关阶段，子类应继续规划角色技能、
        发言或放逐票；返回 ``Action`` 则表示警长动作已经完成，可立即结束规划。
        ``score_adjustments`` 和 ``excluded`` 让不同角色在共享算法上加入可信度、
        队友保护或查验结果，而不复制整套警长状态机。
        """

        del memory
        actor = observation.viewer_id
        rules = RolePlanner._legal_rules(observation)
        if ActionType.SHERIFF_RUN in rules:
            if run_for_sheriff:
                return Action(actor, ActionType.SHERIFF_RUN)
            return self._pass(actor)

        if ActionType.SHERIFF_WITHDRAW in rules:
            if stay_in_race:
                return self._pass(actor)
            return Action(actor, ActionType.SHERIFF_WITHDRAW)

        if ActionType.SHERIFF_VOTE in rules:
            rule = rules[ActionType.SHERIFF_VOTE]
            target = self._ranked_target(
                observation,
                rule,
                rng,
                score_adjustments=score_adjustments,
                excluded=excluded,
            )
            if target is None:
                return self._pass(actor)
            return Action(actor, ActionType.SHERIFF_VOTE, target_id=target)

        if (
            ActionType.SHERIFF_ORDER_CLOCKWISE in rules
            or ActionType.SHERIFF_ORDER_COUNTERCLOCKWISE in rules
        ):
            return self._sheriff_order(
                observation,
                score_adjustments=score_adjustments,
            )

        if ActionType.SHERIFF_SUMMARIZE in rules:
            rule = rules[ActionType.SHERIFF_SUMMARIZE]
            target = self._ranked_target(
                observation,
                rule,
                rng,
                score_adjustments=score_adjustments,
                excluded=excluded,
            )
            if target is None:
                # An engine should always provide a nomination target. Keeping
                # this path legal makes standalone policies robust to an empty
                # target list in an interrupted game.
                return self._pass(actor)
            names = {
                str(player["id"]): str(player.get("name", player["id"]))
                for player in observation.players
            }
            return Action(
                actor,
                ActionType.SHERIFF_SUMMARIZE,
                target_id=target,
                content=(
                    f"听完大家的发言后，我建议今天归票到{names.get(target, target)}。"
                    "请大家结合公开信息核对判断，并集中投票。"
                ),
                metadata={"sheriff_nomination": target, "vote_intent": target},
            )

        badge_rules = {
            kind: rules[kind]
            for kind in (ActionType.SHERIFF_TRANSFER, ActionType.SHERIFF_DESTROY)
            if kind in rules
        }
        if badge_rules:
            transfer = badge_rules.get(ActionType.SHERIFF_TRANSFER)
            if transfer is not None:
                legal_targets = [str(item) for item in transfer.get("target_ids", [])]
                for target in preferred_badge_targets:
                    if str(target) in legal_targets:
                        return Action(
                            actor,
                            ActionType.SHERIFF_TRANSFER,
                            target_id=str(target),
                        )
            if destroy_badge and ActionType.SHERIFF_DESTROY in badge_rules:
                return Action(actor, ActionType.SHERIFF_DESTROY)
            if transfer is not None:
                target = self._ranked_target(
                    observation,
                    transfer,
                    rng,
                    score_adjustments=score_adjustments,
                    excluded=excluded,
                    prefer_low_score=True,
                )
                if target is not None:
                    return Action(
                        actor, ActionType.SHERIFF_TRANSFER, target_id=target
                    )
            if ActionType.SHERIFF_DESTROY in badge_rules:
                return Action(actor, ActionType.SHERIFF_DESTROY)
            return self._pass(actor)
        return None
