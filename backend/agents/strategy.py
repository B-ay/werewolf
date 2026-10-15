"""可继承的角色策略：只允许有界参数和通用经验，不接收旧局身份或任意提示词。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from ..game.models import Role


# 经验来自固定模板，而不是把裁判日志、玩家编号或模型生成的任意文字带入新局。
LESSONS = {
    "vote_evidence": "投票应区分真实技能结果与他人的身份声明；重复指控不等于独立证据。",
    "wolf_claims": "评估夜刀威胁时同时考虑警上和白天的预言家声明，并与狼队可见意图协调。",
    "seer_checks": "先兑现公开警徽流；没有明确计划时，优先查验尚未查过且公开嫌疑较高的玩家。",
    "witch_poison": "毒药不可仅凭弱指控使用；证据不足时允许留药，不必因轮次增加而强行出毒。",
    "hunter_shot": "开枪前比较公开证据强度，合法目标存在并不意味着必须开枪。",
    "villager_claims": "自称预言家不代表身份可信；不要无条件放大未经验证的指控或归票。",
}

COMMON_PARAMETERS = {
    "accuse_weight": (0.0, 4.0),
    "seer_accuse_weight": (0.0, 6.0),
    "defend_weight": (0.0, 4.0),
    "seer_defend_weight": (0.0, 6.0),
    "nomination_weight": (0.0, 6.0),
}
ROLE_PARAMETERS = {
    Role.WEREWOLF: {"seer_threat": (0.0, 24.0), "coordination_weight": (0.0, 24.0), "include_campaign_claims": bool},
    Role.SEER: {"suspect_checks": bool},
    Role.WITCH: {"poison_threshold": (0.0, 8.0), "late_poison_probability": (0.0, 1.0)},
    Role.HUNTER: {"shoot_threshold": (-1000.0, 8.0)},
    Role.VILLAGER: {},
}
ROLE_LESSON = {
    Role.WEREWOLF: "wolf_claims", Role.SEER: "seer_checks", Role.WITCH: "witch_poison",
    Role.HUNTER: "hunter_shot", Role.VILLAGER: "villager_claims",
}


@dataclass(frozen=True)
class RoleStrategy:
    """一个可审计版本；不修改模型权重，也不在运行中改写源代码。"""

    role: Role
    version: str = "v0"
    parameters: Mapping[str, float | bool] = field(default_factory=dict)
    lesson_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role.coerce(self.role))
        if not re.fullmatch(r"v\d+", self.version):
            raise ValueError("strategy version must use v<number>")
        allowed = {**COMMON_PARAMETERS, **ROLE_PARAMETERS[self.role]}
        checked: dict[str, float | bool] = {}
        for name, value in self.parameters.items():
            if name not in allowed:
                raise ValueError(f"unsupported strategy parameter: {name}")
            bounds = allowed[name]
            if bounds is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be boolean")
                checked[name] = value
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be numeric")
                number = float(value)
                if not math.isfinite(number) or not bounds[0] <= number <= bounds[1]:
                    raise ValueError(f"{name} is outside strategy bounds")
                checked[name] = number
        codes = tuple(dict.fromkeys(self.lesson_codes))
        if any(code not in {"vote_evidence", ROLE_LESSON[self.role]} for code in codes):
            raise ValueError("experience contains an unsupported lesson")
        object.__setattr__(self, "parameters", MappingProxyType(checked))
        object.__setattr__(self, "lesson_codes", codes)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "version": self.version,
                "parameters": dict(self.parameters), "lesson_codes": list(self.lesson_codes)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoleStrategy":
        if set(value) - {"role", "version", "parameters", "lesson_codes"}:
            raise ValueError("strategy must not contain game facts or free-form prompts")
        return cls(Role.coerce(value["role"]), str(value.get("version", "v0")),
                   value.get("parameters", {}), tuple(value.get("lesson_codes", ())))

    @property
    def lessons(self) -> list[str]:
        return [LESSONS[code] for code in self.lesson_codes]


def propose_strategy(current: RoleStrategy, feedback: Mapping[str, Any], revision: int) -> RoleStrategy | None:
    """由终局统计提出一个有界候选；只提案，不直接替换正式版本。"""
    if not feedback.get("games") or not (
        feedback.get("losses") or feedback.get("wrong_votes") or feedback.get("wrong_abilities")
        or feedback.get("inconsistent_votes")
    ):
        return None
    params = dict(current.parameters)
    role = current.role
    if role is Role.WITCH:
        params["poison_threshold"] = min(6.0, params.get("poison_threshold", 2.0) + 1.0)
        params["late_poison_probability"] = max(0.0, round(params.get("late_poison_probability", 0.45) - 0.15, 2))
    elif role is Role.HUNTER:
        params["shoot_threshold"] = min(4.0, max(0.0, params.get("shoot_threshold", 0.0)) + 1.0)
    elif role is Role.SEER:
        params["suspect_checks"] = True
        params["seer_accuse_weight"] = max(1.0, params.get("seer_accuse_weight", 3.0) - 0.5)
    elif role is Role.WEREWOLF:
        params["include_campaign_claims"] = True
        params["seer_threat"] = min(20.0, params.get("seer_threat", 8.0) + 2.0)
    else:
        params["seer_accuse_weight"] = max(1.0, params.get("seer_accuse_weight", 3.0) - 0.5)
        params["nomination_weight"] = max(1.0, params.get("nomination_weight", 4.0) - 0.5)
    codes = tuple(dict.fromkeys((*current.lesson_codes, "vote_evidence", ROLE_LESSON[role])))
    if params == dict(current.parameters) and codes == current.lesson_codes:
        return None
    return RoleStrategy(role, f"v{revision}", params, codes)
