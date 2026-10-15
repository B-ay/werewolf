"""与模型厂商无关的最小补全协议。"""

from __future__ import annotations
from typing import Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """与厂商无关的大模型接口。

    ``BaseAgent`` 只依赖这个协议，不直接保存 API Key、拼接 HTTP 请求或了解
    DeepSeek 的接口细节。``backend.llm.provider.DeepSeekProvider`` 负责把这里的
    ``messages`` 转换为 DeepSeek Chat Completions 请求。
    """

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        """Return one text completion."""
