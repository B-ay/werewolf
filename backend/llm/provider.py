"""大模型 Provider 适配层。

本模块只实现 :class:`backend.llm.base.LLMProvider` 所要求的最小接口：角色
Agent 把已经组织好的 ``system``/``user`` 消息交给 Provider，Provider 负责
调用 DeepSeek Chat Completions API，并把模型最终生成的文本返回给 Agent。
它不参与角色推理、动作合法性判断或本地规划器兜底，因此模型供应商相关的
网络细节不会渗入 Agent 和游戏引擎。

安全边界尤其重要：API Key 只能来自启动服务器时的环境变量或服务端显式
构造参数，不能由创建对局的请求传入，也不会进入浏览器状态、游戏事件日志、
结构化回放或异常文本。``DeepSeekProvider.api_key`` 还被设置为
``repr=False``，以防调试输出意外打印密钥。本模块本身不会把密钥写入任何
配置文件；是否持久化环境变量由部署环境负责。

为减少外部依赖，HTTP 请求完全使用 Python 标准库实现。调用是同步、非流式、
JSON 模式的，并对配置、输入消息、响应大小和响应结构做严格校验。可重试的
瞬时故障最多重试一次；最终错误会被转换为不含上游响应正文、请求头或原始
异常信息的脱敏异常，随后角色 Agent 可以安全地回退到本地规划器。
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


# DeepSeek 官方的 Chat Completions 入口。endpoint 可在服务端构造 Provider 时
# 覆盖，但会在 __post_init__ 中强制检查为不带账号、查询串和片段的 HTTPS URL。
DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"

# 未设置 DEEPSEEK_MODEL 时使用的模型名。模型名属于普通配置，可以公开；只有
# DEEPSEEK_API_KEY 是敏感信息。
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"

# 只对限流和明确的服务端瞬时故障重试。400/401/402/422 等请求、认证或账户
# 问题重试没有意义，因此会立即返回脱敏错误。
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 503})

# 限制响应体，避免异常上游返回导致进程无界占用内存；多读取 1 字节用于判断
# 是否已经超过上限。
_MAX_RESPONSE_BYTES = 1_000_000

# Retry-After 和指数退避都被限制在较短范围，避免单个 Agent 请求长期阻塞整局。
_MAX_RETRY_DELAY_SECONDS = 2.0


class DeepSeekConfigurationError(ValueError):
    """服务端 DeepSeek 配置缺失、格式错误或不安全时抛出。"""


class DeepSeekProviderError(RuntimeError):
    """经过主动脱敏的 DeepSeek 请求或响应错误。

    异常只保留调用方可安全展示的固定说明和可选 HTTP 状态码，不携带 API Key、
    Authorization 请求头、上游响应正文或底层网络异常文本。这样即使异常随后
    出现在服务端日志或观战 UI 中，也不会泄露凭据和供应商返回的私有信息。
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class DeepSeekProvider:
    """使用标准库同步调用 DeepSeek Chat Completions 的 Provider。

    属性说明：

    ``api_key``
        服务端密钥。字段设置 ``repr=False``，所以 dataclass 的 ``repr`` 不会
        显示它；请求时只放入 ``Authorization: Bearer ...`` 头，不会进入 JSON
        请求体。
    ``model``
        传给 DeepSeek 的模型名，默认值可由 ``DEEPSEEK_MODEL`` 环境变量覆盖。
    ``endpoint``
        Chat Completions HTTPS 地址，只允许安全 URL 形态。
    ``timeout``
        每次 ``urlopen`` 的超时秒数，而非整局游戏的总超时。
    ``max_tokens``
        单次回复的最大生成 token 数，限制在 1 到 4096。
    ``max_retries``
        额外重试次数，只允许 0 或 1；默认值 1 表示总共最多请求两次。
    ``retry_backoff``
        首次重试的基础等待秒数。当前最多一次重试，但仍采用指数退避公式，便于
        将来扩展；实际等待始终封顶为 2 秒。

    对外接口刻意保持很窄：执行非流式 JSON 模式调用，只返回第一条 assistant
    消息的文本，正好满足各角色 Agent 使用的 ``LLMProvider.complete`` 协议。
    Provider 不直接把文本解析成游戏动作；动作 JSON 的语义校验和本地规划器
    兜底由 Agent 层负责。
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_DEEPSEEK_MODEL
    endpoint: str = DEFAULT_DEEPSEEK_ENDPOINT
    timeout: float = 20.0
    max_tokens: int = 512
    max_retries: int = 1
    retry_backoff: float = 0.25

    def __post_init__(self) -> None:
        """规范化并校验全部服务端配置，不发起任何网络请求。

        构造阶段先失败可以避免带着错误配置运行到对局中途。密钥只检查是否存在
        以及是否含换行符；禁止换行可防止请求头注入。其他数值字段都拒绝 NaN、
        无穷值和越界值。
        """

        # 去掉运维系统注入环境变量时常见的首尾空白。处理后的值只保存在内存中，
        # 本类没有任何写文件或回写环境变量的逻辑。
        self.api_key = str(self.api_key).strip()
        if not self.api_key:
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is not configured")
        if "\r" in self.api_key or "\n" in self.api_key:
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is invalid")

        self.model = str(self.model).strip()
        if not self.model:
            raise DeepSeekConfigurationError("DeepSeek model is not configured")

        self.endpoint = str(self.endpoint).strip()
        parsed = urlsplit(self.endpoint)
        # 除了要求 HTTPS，还拒绝 user:password@host、query 和 fragment，避免凭据
        # 混入 URL，或通过可变查询参数把敏感配置带入代理/访问日志。
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
        ):
            raise DeepSeekConfigurationError("DeepSeek endpoint must be a safe HTTPS URL")

        if not math.isfinite(float(self.timeout)) or float(self.timeout) <= 0:
            raise DeepSeekConfigurationError("DeepSeek timeout must be positive")
        self.timeout = float(self.timeout)

        if isinstance(self.max_tokens, bool) or not 1 <= int(self.max_tokens) <= 4096:
            raise DeepSeekConfigurationError("DeepSeek max_tokens is out of range")
        self.max_tokens = int(self.max_tokens)

        if isinstance(self.max_retries, bool) or int(self.max_retries) not in {0, 1}:
            raise DeepSeekConfigurationError("DeepSeek max_retries must be zero or one")
        self.max_retries = int(self.max_retries)

        if (
            not math.isfinite(float(self.retry_backoff))
            or float(self.retry_backoff) < 0
            or float(self.retry_backoff) > _MAX_RETRY_DELAY_SECONDS
        ):
            raise DeepSeekConfigurationError("DeepSeek retry_backoff is out of range")
        self.retry_backoff = float(self.retry_backoff)

    @classmethod
    def from_env(
        cls,
        environ: Optional[Mapping[str, str]] = None,
        **overrides: Any,
    ) -> "DeepSeekProvider":
        """从服务端环境读取配置并构造 Provider，不序列化 API Key。

        默认读取进程的 :data:`os.environ`；测试或嵌入式调用可以传入一个只读
        ``Mapping``，从而无需修改全局环境。当前读取：

        - ``DEEPSEEK_API_KEY``：必填，仅在服务端内存中使用；
        - ``DEEPSEEK_MODEL``：选填，缺省使用 :data:`DEFAULT_DEEPSEEK_MODEL`。

        ``overrides`` 用于服务器代码覆盖非秘密配置，例如超时、端点或 token
        上限。显式 ``model`` 覆盖环境变量中的模型名。方法不会创建 ``.env``、
        JSON 或其他配置文件，也不会把环境变量内容写回磁盘。
        """

        source = os.environ if environ is None else environ
        api_key = str(source.get("DEEPSEEK_API_KEY", "")).strip()
        if not api_key:
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is not configured")
        model = str(source.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)).strip()
        if "model" in overrides:
            model = str(overrides.pop("model"))
        return cls(api_key=api_key, model=model, **overrides)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        """执行一次 JSON 模式补全并返回 assistant 的文本内容。

        ``messages`` 使用 Chat Completions 的消息数组结构，每项至少包含字符串
        ``role`` 和 ``content``；允许的角色为 ``system``、``user``、
        ``assistant``、``tool``。``temperature`` 必须是 0 到 2 的有限数。

        请求结构固定为：

        - ``model``：当前 Provider 的模型名；
        - ``messages``：规范化后的提示词消息；
        - ``stream=False``：等待完整响应，便于同步状态机消费；
        - ``thinking={"type": "disabled"}``：关闭思考模式，要求直接输出动作；
        - ``response_format={"type": "json_object"}``：要求模型输出 JSON 对象；
        - ``temperature`` 和 ``max_tokens``：采样与长度约束。

        请求体采用 UTF-8 JSON，API Key 只出现在 Authorization 请求头。429、
        500、503 以及网络/超时故障可进行一次短重试；请求参数、认证和余额错误
        不重试。无论哪种失败，都不会把上游错误正文、请求头、原始异常信息或
        凭据拼接到异常中，因为该异常可能最终显示在本地观战界面。
        """

        # 在组装请求之前完成输入检查，非法消息不会触发网络调用。
        normalized_messages = self._normalize_messages(messages)
        try:
            normalized_temperature = float(temperature)
        except (TypeError, ValueError):
            raise DeepSeekProviderError("DeepSeek temperature is invalid") from None
        if (
            not math.isfinite(normalized_temperature)
            or normalized_temperature < 0
            or normalized_temperature > 2
        ):
            raise DeepSeekProviderError("DeepSeek temperature is invalid")

        # Chat Completions 请求体。密钥不在 payload 中，只在稍后的 HTTP 头中。
        payload = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": False,
            "thinking": {"type": "disabled"},
            "temperature": normalized_temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        try:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError):
            raise DeepSeekProviderError("DeepSeek request could not be encoded") from None

        # max_retries 表示“额外尝试次数”，因此循环总次数为 1 + max_retries。
        for attempt in range(self.max_retries + 1):
            request = Request(
                self.endpoint,
                data=encoded,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    # 多读一个字节，既能识别超限，又避免把任意大的响应全部读入
                    # 内存。响应体只在内存中解析，不写入磁盘或事件日志。
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
            except HTTPError as exc:
                status = int(getattr(exc, "code", 0) or 0)
                # 服务端可能通过 Retry-After 指示等待时间；该值仍会被限制为 2 秒，
                # 防止恶意或异常响应长时间卡住当前游戏回合。
                retry_after = self._retry_after_seconds(getattr(exc, "headers", None))
                # 主动关闭 HTTPError 内部的响应流，但绝不读取或转发其正文。
                try:
                    exc.close()
                except Exception:
                    pass
                if status in _RETRYABLE_HTTP_STATUS and attempt < self.max_retries:
                    time.sleep(retry_after if retry_after is not None else self._retry_delay(attempt))
                    continue
                raise self._http_error(status) from None
            except (URLError, TimeoutError, socket.timeout, OSError):
                # 网络、DNS、超时及底层 I/O 故障通常是瞬时的，可按相同策略重试。
                # 原异常可能包含 URL、代理信息等细节，因此不向外传播其文本。
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise DeepSeekProviderError("DeepSeek request failed") from None
            except Exception:
                # 测试替身或不同标准库处理器可能抛出其他异常。这里保持同样的
                # 脱敏边界，避免详细异常文本进入服务端日志或观战 UI。
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise DeepSeekProviderError("DeepSeek request failed") from None

            if len(raw) > _MAX_RESPONSE_BYTES:
                raise DeepSeekProviderError("DeepSeek response is too large")
            return self._parse_response(raw)

        # 按当前循环逻辑一定会 return 或 raise；保留这个脱敏防御分支，避免未来
        # 修改重试边界时意外返回 None 或抛出携带底层信息的异常。
        raise DeepSeekProviderError("DeepSeek request failed")

    @staticmethod
    def _normalize_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        """验证并复制 Chat Completions 消息，返回最小安全结构。

        只保留 ``role`` 和 ``content`` 两个字段，调用方 Mapping 中的其他字段不会
        被透传到供应商。返回新列表还能避免在请求期间引用调用方可变对象。
        """

        if isinstance(messages, (str, bytes)) or not messages:
            raise DeepSeekProviderError("DeepSeek messages are invalid")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise DeepSeekProviderError("DeepSeek messages are invalid")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise DeepSeekProviderError("DeepSeek messages are invalid")
            if role not in {"system", "user", "assistant", "tool"}:
                raise DeepSeekProviderError("DeepSeek messages are invalid")
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _parse_response(raw: bytes) -> str:
        """解析 DeepSeek 响应并提取首个正常结束的 assistant 文本。

        合法响应必须是 UTF-8 JSON 对象，包含非空 ``choices``，第一项必须以
        ``finish_reason == "stop"`` 正常结束，并提供非空字符串 ``message.content``。
        因长度上限结束或结构不完整的内容不会交给 Agent，以免半截 JSON 被当作
        游戏动作。解析失败统一转换成脱敏错误，上游原始响应不会出现在异常中。
        """

        try:
            document = json.loads(raw.decode("utf-8"))
            if not isinstance(document, Mapping):
                raise TypeError
            choices = document.get("choices")
            if not isinstance(choices, list) or not choices:
                raise TypeError
            choice = choices[0]
            if not isinstance(choice, Mapping):
                raise TypeError
            if choice.get("finish_reason") != "stop":
                raise TypeError
            message = choice.get("message")
            if not isinstance(message, Mapping):
                raise TypeError
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise TypeError
            return content
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError):
            raise DeepSeekProviderError("DeepSeek returned an invalid response") from None

    def _retry_delay(self, attempt: int) -> float:
        """计算指数退避时间，并限制到全局最大等待时间。"""

        return min(self.retry_backoff * (2**attempt), _MAX_RETRY_DELAY_SECONDS)

    @staticmethod
    def _retry_after_seconds(headers: Any) -> Optional[float]:
        """安全解析秒数形式的 ``Retry-After``，无效值返回 ``None``。

        HTTP 还允许日期形式的 ``Retry-After``，本实现有意不解析该形式；这样
        不依赖本机时钟，且所有等待都能被严格限制在短时间范围内。
        """

        if headers is None:
            return None
        try:
            value = headers.get("Retry-After")
            if value is None:
                return None
            seconds = float(value)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(seconds):
            return None
        return min(max(seconds, 0.0), _MAX_RETRY_DELAY_SECONDS)

    @staticmethod
    def _http_error(status: int) -> DeepSeekProviderError:
        """把 HTTP 状态码映射成固定、可展示且不含上游正文的错误。"""

        messages = {
            400: "DeepSeek rejected the request",
            401: "DeepSeek authentication failed",
            402: "DeepSeek account balance is insufficient",
            422: "DeepSeek rejected request parameters",
            429: "DeepSeek rate limit was reached",
            500: "DeepSeek service failed",
            503: "DeepSeek service is overloaded",
        }
        return DeepSeekProviderError(
            messages.get(status, "DeepSeek request failed"),
            status_code=status or None,
        )


def build_deepseek_provider_from_env(
    environ: Optional[Mapping[str, str]] = None,
    *,
    required: bool = False,
    **overrides: Any,
) -> Optional[DeepSeekProvider]:
    """按环境变量按需创建 DeepSeek Provider。

    这是应用启动阶段使用的便捷入口。存在 ``DEEPSEEK_API_KEY`` 时，委托
    :meth:`DeepSeekProvider.from_env` 完成读取和校验；未配置时，默认返回
    ``None``，上层即可运行纯本地规划器。若 ``required=True``，未配置密钥会
    立即抛出 :class:`DeepSeekConfigurationError`，适合要求必须启用大模型的
    部署模式。

    与 ``from_env`` 一样，本函数只读取环境变量，不把 API Key 写入源码、配置
    文件、事件日志或任何序列化对象。
    """

    source = os.environ if environ is None else environ
    if not str(source.get("DEEPSEEK_API_KEY", "")).strip():
        if required:
            raise DeepSeekConfigurationError("DEEPSEEK_API_KEY is not configured")
        return None
    return DeepSeekProvider.from_env(source, **overrides)


__all__ = [
    "DEFAULT_DEEPSEEK_ENDPOINT",
    "DEFAULT_DEEPSEEK_MODEL",
    "DeepSeekConfigurationError",
    "DeepSeekProvider",
    "DeepSeekProviderError",
    "build_deepseek_provider_from_env",
]
