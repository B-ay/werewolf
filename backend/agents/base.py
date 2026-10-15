"""BaseAgent：五类角色共享的视角绑定、记忆、规划、模型调用和安全降级。"""

from __future__ import annotations
import json
import random
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Mapping, Optional, Protocol, runtime_checkable
from .memory import RoleGoal, RoleMemory
from .planner import RolePlanner
from .strategy import RoleStrategy
from ..game.models import Action, ActionType, Observation, Role
from ..llm.base import LLMProvider


@runtime_checkable
class Agent(Protocol):
    """引擎只要求同步 decide 接口，测试替身无需继承 BaseAgent。"""

    def decide(self, observation: Observation, rng: random.Random) -> Action:
        """从本座位合法观察中选择一个动作。"""
        ...


class BaseAgent(ABC):
    """角色 Agent 的共享安全外壳；具体策略由五类 Planner 分别实现。

    一次决策的完整流水线如下：

    1. 校验观察中的角色、座位归属和合法动作列表；
    2. 将本座位可见信息写入 ``RoleMemory``；
    3. 调用专属 ``RolePlanner`` 生成不依赖网络的基线动作；
    4. 再校验一次基线，异常时构造最小合法动作兜底；
    5. 如果注入了模型 Provider，把“角色提示词 + 角色目标 + 当前观察 + 紧凑
       记忆”发给模型，并尝试用其 JSON 动作覆盖基线；
    6. 模型调用、解析或合法性校验任一步失败，都回退到已经准备好的基线动作；
    7. 把最终动作及来源写回记忆，供后续轮次保持一致。

    ``BaseAgent`` 不直接依赖 DeepSeek SDK。启动层把
    ``backend.llm.provider.DeepSeekProvider`` 作为 ``LLMProvider`` 注入后，步骤 5
    才会真实发起 DeepSeek 请求；未注入时只执行本地规划器。
    """

    ROLE: Role

    CHINESE_COMMUNICATION_RULE = (
        "局内交流统一使用简体中文：公开发言、警上竞选、PK发言、警长总结及任何行动说明的"
        "content 都必须用简体中文，不要输出英文句子。玩家名称、玩家ID以及JSON字段名和"
        "枚举值保留原样。即使观察或历史记忆中出现英文，新的发言仍须使用简体中文。"
    )

    def __init__(
        self,
        *,
        provider: Optional[LLMProvider] = None,
        memory: Optional[RoleMemory] = None,
        temperature: float = 0.2,
        owner_id: Optional[str] = None,
        strategy: Optional[RoleStrategy] = None,
    ) -> None:
        """组装该角色独立的目标、长期记忆、提示词和规划器。

        Provider 可以在不同座位间共享（它只是无状态传输适配器），但
        ``RoleMemory`` 必须按座位独立创建，并与角色和 ``owner_id`` 一致。
        """

        self.role = self.ROLE
        self.owner_id = str(owner_id) if owner_id is not None else None
        self.provider = provider
        self.temperature = float(temperature)
        self.goal = self._make_goal()
        self.memory = memory if memory is not None else RoleMemory(
            role=self.role,
            owner_id=self.owner_id,
            viewer_id=self.owner_id,
        )
        if self.memory.role is None:
            self.memory.role = self.role
        elif self.memory.role is not self.role:
            raise ValueError(
                f"memory role {self.memory.role.value} cannot be used by {self.role.value}"
            )
        if self.memory.owner_id is not None:
            if self.owner_id is not None and self.owner_id != self.memory.owner_id:
                raise ValueError("agent owner_id conflicts with memory owner_id")
            self.owner_id = self.memory.owner_id
        elif self.owner_id is not None:
            self.memory.owner_id = self.owner_id
            self.memory.viewer_id = self.owner_id
        self.prompt = self._make_prompt()
        self.planner = self._make_planner()
        if strategy is not None:
            self.memory.inherit_strategy(strategy)
        if self.memory.experience:
            inherited = RoleStrategy.from_dict(self.memory.experience)
            if inherited.role is not self.role:
                raise ValueError("strategy does not match agent role")
            self.planner.parameters = dict(inherited.parameters)
        self._state_view: Optional[Observation] = None
        # 仅供完局验证汇总，不进入公开事件或其他玩家的观察；不受记忆窗口裁剪影响。
        self.decision_stats = {"planner_repairs": 0, "model_attempts": 0, "model_fallbacks": 0}

    @abstractmethod
    def _make_goal(self) -> RoleGoal:
        raise NotImplementedError

    @abstractmethod
    def _make_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _make_planner(self) -> RolePlanner:
        raise NotImplementedError

    def update_state(self, observation: Observation) -> None:
        """接收本座位的合法视角，校验归属并更新独立记忆。

        参考模板的 update_state，但只接收现有 Observation 协议。当前视角保留
        深拷贝，state 属性也返回副本，调用者无法借助它修改裁判或其他 Agent。
        """
        # 第一层：身份与座位绑定校验，防止把一个角色 Agent 错配给其他座位。
        if not observation.legal_actions:
            raise ValueError("agent was asked to act without a legal action")
        observed_role = Role.coerce(observation.self_state["role"])
        if observed_role is not self.role:
            raise ValueError(f"{type(self).__name__} cannot control role {observed_role.value}")
        if str(observation.self_state.get("id")) != observation.viewer_id:
            raise ValueError("observation viewer_id does not match self_state.id")
        if self.owner_id is None:
            self.owner_id = observation.viewer_id
            self.memory.owner_id = self.owner_id
            self.memory.viewer_id = self.owner_id
        elif self.owner_id != observation.viewer_id:
            raise ValueError(
                f"{type(self).__name__} belongs to {self.owner_id}, "
                f"not {observation.viewer_id}"
            )
        # 第二层：先记忆当前观察，再让规划器读取最新的可见历史和策略状态。
        self.memory.ingest(observation)
        self._state_view = deepcopy(observation)

    @property
    def state(self) -> Optional[dict[str, Any]]:
        """当前玩家可见状态的独立字典，尚未接收观察时为 None。"""

        return self._state_view.to_dict() if self._state_view is not None else None

    @property
    def is_alive(self) -> bool:
        """从最近一次合法观察中读取本人的存活状态。"""

        return bool(self.memory.self_state.get("alive", True))

    def build_context_prompt(self) -> str:
        """将当前合法视角与裁剪后的记忆组合为所有角色共用的模型上下文。"""

        if self._state_view is None:
            return ""
        payload = {
            "observation": self._state_view.to_dict(),
            "memory": self.memory.prompt_view(),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def plan(self, observation: Observation, rng: random.Random) -> Action:
        """在摄取本轮观察后运行角色专属规划器，并统一处理非法基线兜底。"""

        planned = self.planner.plan(observation, self.memory, self.goal, rng)
        if not BaseAgent._action_is_legal(planned, observation):
            self.decision_stats["planner_repairs"] += 1
            planned = self._minimal_legal_action(observation, rng)
        return planned

    def decide(self, observation: Observation, rng: random.Random) -> Action:
        """统一执行视角更新、记忆摄取、规划、模型覆盖、校验、降级及最终动作记录。

        保留现有同步接口，每次只执行一个合法动作，不额外消耗随机数或重复调用
        模型。角色子类只负责自己的目标、提示词和 Planner，复用本方法的全部边界。
        """

        self.update_state(observation)
        planned = self.plan(observation, rng)
        action = planned
        source = "planner"
        if self.provider is not None:
            self.decision_stats["model_attempts"] += 1
            try:
                # 第四层：模型候选必须解析为 Action，并重新通过完整合法性白名单。
                candidate = self._model_action(observation)
                if not BaseAgent._action_is_legal(candidate, observation):
                    raise ValueError("provider selected an action outside legal_actions")
                if not BaseAgent._uses_chinese_content(candidate, observation):
                    raise ValueError("provider content must be in Chinese")
                action = candidate
                source = "provider"
            except Exception:
                self.decision_stats["model_fallbacks"] += 1
                # 网络错误、DeepSeek 超时、非 JSON、未知动作、越权目标等都不会
                # 中断对局；统一退回已验证的本地规划结果。
                action = planned
                source = "planner_fallback"
        # 只记录最终采用的动作，而不是失败的模型候选，保证历史与真实执行一致。
        self.memory.remember_decision(action, observation, source=source)
        return action

    def _model_action(self, observation: Observation) -> Action:
        """构造受限提示词，调用 Provider，并把返回 JSON 转换为 ``Action``。

        系统消息包含本角色专属提示词、结构化目标和安全/输出协议；用户消息只含
        当前 ``Observation`` 与 ``RoleMemory.prompt_view()``。当 Provider 是
        ``DeepSeekProvider`` 时，``complete`` 会发送这两条消息到配置的
        DeepSeek OpenAI-compatible 接口。本方法只负责生成候选动作，外层
        ``decide`` 仍会执行最终合法性校验和异常降级。
        """

        assert self.provider is not None
        # 明确要求“只返回一个 JSON 对象”，并把警徽流结构限制写入系统提示词，
        # 尽量从生成端减少格式漂移；安全性仍以代码校验为准，而非相信提示词。
        system = (
            f"{self.prompt}\n\n"
            f"角色目标：\n{json.dumps(self.goal.to_dict(), ensure_ascii=False)}\n\n"
            f"交流语言要求：\n{self.CHINESE_COMMUNICATION_RULE}\n\n"
            "安全与输出规则：\n只能使用提供的observation和memory，不得臆造隐藏角色、"
            "私有事件或当前不可用的行动。只返回一个JSON对象，包含action_type，"
            "以及可选的target_id、content、metadata。动作仅可从legal_actions中选择。"
            "公开警徽流声明metadata.badge_flow必须严格使用以下结构："
            '{"mode":"single|double","targets":["player_id"],'
            '"fallback_target":"player_id or null"}；single必须有一个目标，double必须'
            "有两个目标，不得在该结构中包含查验真值。警长总结中的"
            "metadata.sheriff_nomination必须与target_id一致。"
            'JSON示例：{"action_type":"speak","target_id":"p2",'
            '"content":"我希望进一步听取这位玩家的解释。","metadata":{}}。'
        )
        # prompt_view 已按当前局/历史局两层裁剪；不会把 GameEngine 或全知日志传出。
        raw = self.provider.complete(
            (
                {"role": "system", "content": system},
                {"role": "user", "content": self.build_context_prompt()},
            ),
            temperature=self.temperature,
        )
        return Action.from_dict(BaseAgent._parse_json_object(raw), default_actor=observation.viewer_id)

    @staticmethod
    def _public_metadata_is_valid(action: Action, observation: Observation) -> bool:
        """仅依据当前视角校验公开发言中的结构化字段。

        警徽流只声明未来查验的座位，不允许夹带真实或伪造的查验真值。这里也会
        验证指控、保护、投票意向等座位引用，避免模型把不存在或不可见的座位写入
        公共事件流。该校验不读取隐藏状态，因此不会反向造成信息泄漏。
        """

        metadata = action.metadata
        if not isinstance(metadata, Mapping):
            return False
        player_ids = set(RolePlanner._visible_player_ids(observation))

        flow = metadata.get("badge_flow")
        if flow is not None:
            if not isinstance(flow, Mapping):
                return False
            if set(flow) != {"mode", "targets", "fallback_target"}:
                return False
            mode = flow.get("mode")
            if mode not in {"single", "double"}:
                return False
            targets = flow.get("targets")
            if not isinstance(targets, list):
                return False
            expected_count = 1 if mode == "single" else 2
            if len(targets) != expected_count:
                return False
            normalized_targets = [str(item) for item in targets]
            if len(set(normalized_targets)) != len(normalized_targets):
                return False
            if any(target not in player_ids for target in normalized_targets):
                return False
            if observation.viewer_id in normalized_targets:
                return False
            fallback = flow.get("fallback_target")
            if fallback is not None:
                fallback = str(fallback)
                if (
                    fallback not in player_ids
                    or fallback == observation.viewer_id
                    or fallback in normalized_targets
                ):
                    return False

        nomination = metadata.get("sheriff_nomination")
        if nomination is not None:
            nomination = str(nomination)
            if nomination not in player_ids or nomination == observation.viewer_id:
                return False
            if (
                action.action_type is ActionType.SHERIFF_SUMMARIZE
                and action.target_id != nomination
            ):
                return False

        # 公共发言中的座位引用也须验证，防止模型把不存在的玩家写入事件流。
        for key in ("accuse_target", "defend_target", "vote_intent"):
            referenced = metadata.get(key)
            if referenced is not None and str(referenced) not in player_ids:
                return False
        claimed_role = metadata.get("claim_role")
        if claimed_role is not None and str(claimed_role) not in {
            role.value for role in Role
        }:
            return False
        return True

    @staticmethod
    def _parse_json_object(text: str) -> Mapping[str, Any]:
        """把模型文本解析为 JSON 对象，并兼容外层带少量说明文字的响应。"""

        text = text.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("LLM response did not contain a JSON object")
            value = json.loads(match.group(0))
        if not isinstance(value, Mapping):
            raise ValueError("LLM response must be a JSON object")
        return value

    @staticmethod
    def _uses_chinese_content(action: Action, observation: Observation) -> bool:
        """模型自然语言须包含中文正文，玩家名称和ID中的汉字不能充当正文。

        无文本的投票、技能动作照常通过；空发言仍由动作合法性检查拒绝。此检查只作用
        于模型候选，人类和脚本输入不被改写。模型返回纯英文时使用已准备好的中文基线。
        """

        content = (action.content or "").strip()
        if not content:
            return True
        references = {str(player.get("name", "")) for player in observation.players}
        references.update(str(player["id"]) for player in observation.players)
        for reference in sorted(references, key=len, reverse=True):
            if reference:
                content = content.replace(reference, "")
        return re.search(r"[\u3400-\u9fff]", content) is not None

    @staticmethod
    def _action_is_legal(action: Action, observation: Observation) -> bool:
        """在动作离开 Agent 边界前进行最终白名单校验。

        校验涵盖行动者、动作类型、目标、是否允许弃权、必填发言以及公开 metadata。
        DeepSeek 的输出只有通过这里的检查才可能进入引擎。
        """

        if action.actor_id != observation.viewer_id:
            return False
        if not BaseAgent._public_metadata_is_valid(action, observation):
            return False
        if action.action_type is ActionType.PASS:
            return action.target_id is None and any(
                bool(rule.get("can_pass")) for rule in observation.legal_actions
            )

        for rule in observation.legal_actions:
            try:
                expected = ActionType.coerce(rule.get("action_type"))
            except (TypeError, ValueError):
                continue
            if action.action_type is not expected:
                continue
            targets = [str(item) for item in rule.get("target_ids", [])]
            if action.target_id is not None and action.target_id not in targets:
                return False
            if expected is not ActionType.SPEAK and targets and action.target_id is None:
                return False
            if bool(rule.get("content_required")) and not (action.content or "").strip():
                return False
            return True
        return False

    @staticmethod
    def _minimal_legal_action(observation: Observation, rng: random.Random) -> Action:
        """在本地规划器意外返回非法动作时构造最小可执行兜底。

        可弃权阶段优先 ``pass``；否则从排序后的合法目标中用对局 RNG 选择，并在
        必填发言阶段补一条不含隐藏信息的中性文本。
        """

        actor = observation.viewer_id
        rule = RolePlanner._legal(observation)
        if rule.get("can_pass"):
            return Action(actor, ActionType.PASS)
        kind = ActionType.coerce(rule["action_type"])
        targets = [str(item) for item in rule.get("target_ids", [])]
        target = rng.choice(sorted(targets)) if targets else None
        content = "我会仅依据目前可见的信息继续分析。"
        return Action(actor, kind, target_id=target, content=content if rule.get("content_required") else None)
