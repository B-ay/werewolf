"""Concrete, dependency-free LLM provider adapters.

Only the small :class:`backend.agents.LLMProvider` protocol is implemented
here.  Provider credentials are deliberately server-side configuration: they
must never be accepted from game payloads, exposed to the browser, or included
in logs and exceptions.
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


DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 503})
_MAX_RESPONSE_BYTES = 1_000_000
_MAX_RETRY_DELAY_SECONDS = 2.0


class DeepSeekConfigurationError(ValueError):
    """Raised when local DeepSeek configuration is missing or unsafe."""


class DeepSeekProviderError(RuntimeError):
    """A deliberately redacted DeepSeek request or response failure."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class DeepSeekProvider:
    """Synchronous DeepSeek Chat Completions provider using the stdlib.

    The adapter intentionally has a narrow surface.  It performs non-streamed
    JSON-mode calls and returns only the assistant's final text, matching the
    ``LLMProvider.complete`` protocol consumed by each role agent.  One short
    retry is allowed for documented transient failures; all other failures are
    surfaced as redacted exceptions so the role agent can use its local
    planner fallback.
    """

    api_key: str = field(repr=False)
    model: str = DEFAULT_DEEPSEEK_MODEL
    endpoint: str = DEFAULT_DEEPSEEK_ENDPOINT
    timeout: float = 20.0
    max_tokens: int = 512
    max_retries: int = 1
    retry_backoff: float = 0.25

    def __post_init__(self) -> None:
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
        """Build from server environment without ever serializing the secret."""

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
        """Return one JSON-mode assistant completion.

        No upstream error text, response body, request header, or credential is
        interpolated into an exception.  This is important because caller
        errors may later be shown in the local observer UI.
        """

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
                    raw = response.read(_MAX_RESPONSE_BYTES + 1)
            except HTTPError as exc:
                status = int(getattr(exc, "code", 0) or 0)
                retry_after = self._retry_after_seconds(getattr(exc, "headers", None))
                try:
                    exc.close()
                except Exception:
                    pass
                if status in _RETRYABLE_HTTP_STATUS and attempt < self.max_retries:
                    time.sleep(retry_after if retry_after is not None else self._retry_delay(attempt))
                    continue
                raise self._http_error(status) from None
            except (URLError, TimeoutError, socket.timeout, OSError):
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise DeepSeekProviderError("DeepSeek request failed") from None
            except Exception:
                # Transport doubles and alternate stdlib handlers can raise
                # other exception types.  Never let their possibly verbose
                # messages escape into the server or UI.
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay(attempt))
                    continue
                raise DeepSeekProviderError("DeepSeek request failed") from None

            if len(raw) > _MAX_RESPONSE_BYTES:
                raise DeepSeekProviderError("DeepSeek response is too large")
            return self._parse_response(raw)

        # The loop always returns or raises.  Keep a defensive redacted guard
        # in case its retry bounds are changed in the future.
        raise DeepSeekProviderError("DeepSeek request failed")

    @staticmethod
    def _normalize_messages(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
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
        return min(self.retry_backoff * (2**attempt), _MAX_RETRY_DELAY_SECONDS)

    @staticmethod
    def _retry_after_seconds(headers: Any) -> Optional[float]:
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
    """Return an environment-configured provider, or ``None`` when optional."""

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
