"""唯一角色创建入口：按座位创建独立 Agent，可共享无状态模型 Provider。"""

from __future__ import annotations

from typing import Callable

from .base import Agent, BaseAgent
from .hunter import HunterAgent
from .memory import RoleMemory
from .strategy import RoleStrategy
from .seer import SeerAgent
from .villager import VillagerAgent
from .werewolf import WerewolfAgent
from .witch import WitchAgent
from ..game.models import Role
from ..llm.base import LLMProvider


AgentFactory = Callable[[str, Role], Agent]

_ROLE_AGENTS: dict[Role, type[BaseAgent]] = {
    Role.WEREWOLF: WerewolfAgent,
    Role.SEER: SeerAgent,
    Role.WITCH: WitchAgent,
    Role.HUNTER: HunterAgent,
    Role.VILLAGER: VillagerAgent,
}


def create_agent(
    player_id: str,
    role: Role | str,
    *,
    provider: LLMProvider | None = None,
    memory: RoleMemory | None = None,
    temperature: float = 0.2,
    strategy: RoleStrategy | None = None,
) -> BaseAgent:
    """创建本座位的目标、记忆、提示词和专属规划器。

    默认运行本地规划；注入 DeepSeekProvider 后启用模型决策与本地降级。
    服务端可用 functools.partial 绑定 Provider，无需另一层工厂包装。
    Provider 可以共享，但记忆必须独立；复用快照时由 BaseAgent 校验角色和归属。
    """

    return _ROLE_AGENTS[Role.coerce(role)](
        provider=provider,
        memory=memory,
        temperature=temperature,
        owner_id=player_id,
        strategy=strategy,
    )
