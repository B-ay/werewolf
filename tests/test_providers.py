import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from backend.providers import (
    DEFAULT_DEEPSEEK_ENDPOINT,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekConfigurationError,
    DeepSeekProvider,
    DeepSeekProviderError,
    build_deepseek_provider_from_env,
)


TEST_KEY = "test-only-secret-never-send"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def read(self, limit: int = -1) -> bytes:
        return self.payload if limit < 0 else self.payload[:limit]


def completion_response(
    content: str = '{"action_type":"vote","target_id":"p2"}',
    *,
    finish_reason: str = "stop",
) -> FakeResponse:
    body = {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
            }
        ]
    }
    return FakeResponse(json.dumps(body).encode("utf-8"))


def http_error(status: int, *, body: bytes = b"upstream-private-body", retry_after=None):
    headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    return HTTPError(
        DEFAULT_DEEPSEEK_ENDPOINT,
        status,
        "upstream-private-header",
        headers,
        io.BytesIO(body),
    )


class DeepSeekProviderTests(unittest.TestCase):
    def test_success_uses_official_json_mode_request(self) -> None:
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return completion_response()

        provider = DeepSeekProvider(api_key=TEST_KEY)
        messages = [
            {"role": "system", "content": "Return one JSON object."},
            {"role": "user", "content": "Choose an action."},
        ]
        with patch("backend.providers.urlopen", side_effect=fake_urlopen):
            result = provider.complete(messages, temperature=0.2)

        self.assertEqual(result, '{"action_type":"vote","target_id":"p2"}')
        self.assertEqual(len(calls), 1)
        request, timeout = calls[0]
        self.assertEqual(request.full_url, DEFAULT_DEEPSEEK_ENDPOINT)
        self.assertEqual(timeout, 20.0)
        self.assertEqual(request.get_header("Authorization"), f"Bearer {TEST_KEY}")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], DEFAULT_DEEPSEEK_MODEL)
        self.assertEqual(payload["messages"], messages)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["temperature"], 0.2)
        self.assertNotIn(TEST_KEY, request.data.decode("utf-8"))

    def test_api_key_is_redacted_from_repr_and_errors(self) -> None:
        provider = DeepSeekProvider(api_key=TEST_KEY, max_retries=0)
        self.assertNotIn(TEST_KEY, repr(provider))

        upstream = RuntimeError(f"transport included {TEST_KEY} and private body")
        with patch("backend.providers.urlopen", side_effect=upstream):
            with self.assertRaises(DeepSeekProviderError) as caught:
                provider.complete([{"role": "user", "content": "JSON please"}])
        rendered = f"{caught.exception!r} {caught.exception}"
        self.assertNotIn(TEST_KEY, rendered)
        self.assertNotIn("private body", rendered)
        self.assertNotIn("Authorization", rendered)

    def test_from_env_and_optional_builder(self) -> None:
        provider = DeepSeekProvider.from_env(
            {"DEEPSEEK_API_KEY": TEST_KEY, "DEEPSEEK_MODEL": "deepseek-v4-pro"}
        )
        self.assertEqual(provider.model, "deepseek-v4-pro")
        self.assertNotIn(TEST_KEY, repr(provider))
        self.assertIsNone(build_deepseek_provider_from_env({}))
        with self.assertRaises(DeepSeekConfigurationError):
            build_deepseek_provider_from_env({}, required=True)
        with self.assertRaises(DeepSeekConfigurationError):
            DeepSeekProvider.from_env({})

    def test_429_retries_once_and_honors_capped_retry_after(self) -> None:
        outcomes = [http_error(429, retry_after=99), completion_response()]
        calls = []

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        provider = DeepSeekProvider(api_key=TEST_KEY)
        with (
            patch("backend.providers.urlopen", side_effect=fake_urlopen),
            patch("backend.providers.time.sleep") as sleep,
        ):
            result = provider.complete([{"role": "user", "content": "JSON please"}])
        self.assertTrue(result.startswith("{"))
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(2.0)

    def test_authentication_and_parameter_errors_are_not_retried(self) -> None:
        for status in (400, 401, 402, 422):
            with self.subTest(status=status):
                provider = DeepSeekProvider(api_key=TEST_KEY)
                with (
                    patch("backend.providers.urlopen", side_effect=http_error(status)) as opener,
                    patch("backend.providers.time.sleep") as sleep,
                ):
                    with self.assertRaises(DeepSeekProviderError) as caught:
                        provider.complete([{"role": "user", "content": "JSON please"}])
                self.assertEqual(caught.exception.status_code, status)
                self.assertNotIn(TEST_KEY, str(caught.exception))
                self.assertNotIn("upstream-private", str(caught.exception))
                self.assertEqual(opener.call_count, 1)
                sleep.assert_not_called()

    def test_transient_http_failure_retries_only_once(self) -> None:
        provider = DeepSeekProvider(api_key=TEST_KEY, retry_backoff=0)
        with (
            patch("backend.providers.urlopen", side_effect=[http_error(503), http_error(503)]) as opener,
            patch("backend.providers.time.sleep") as sleep,
        ):
            with self.assertRaises(DeepSeekProviderError) as caught:
                provider.complete([{"role": "user", "content": "JSON please"}])
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(opener.call_count, 2)
        sleep.assert_called_once_with(0.0)

    def test_network_failure_retries_then_raises_redacted_error(self) -> None:
        provider = DeepSeekProvider(api_key=TEST_KEY, retry_backoff=0)
        failure = URLError(f"network details {TEST_KEY}")
        with (
            patch("backend.providers.urlopen", side_effect=[failure, failure]) as opener,
            patch("backend.providers.time.sleep"),
        ):
            with self.assertRaises(DeepSeekProviderError) as caught:
                provider.complete([{"role": "user", "content": "JSON please"}])
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(str(caught.exception), "DeepSeek request failed")
        self.assertNotIn(TEST_KEY, str(caught.exception))

    def test_invalid_or_incomplete_responses_are_rejected(self) -> None:
        invalid_responses = (
            FakeResponse(b"not-json"),
            FakeResponse(b"{}"),
            completion_response(""),
            completion_response("{}", finish_reason="length"),
        )
        for response in invalid_responses:
            with self.subTest(payload=response.payload):
                provider = DeepSeekProvider(api_key=TEST_KEY, max_retries=0)
                with patch("backend.providers.urlopen", return_value=response):
                    with self.assertRaisesRegex(
                        DeepSeekProviderError, "invalid response"
                    ):
                        provider.complete([{"role": "user", "content": "JSON please"}])

    def test_input_and_configuration_validation_happens_before_transport(self) -> None:
        with self.assertRaises(DeepSeekConfigurationError):
            DeepSeekProvider(api_key="")
        with self.assertRaises(DeepSeekConfigurationError):
            DeepSeekProvider(api_key=TEST_KEY, endpoint="http://api.deepseek.com/chat/completions")
        with self.assertRaises(DeepSeekConfigurationError):
            DeepSeekProvider(api_key=TEST_KEY, max_retries=2)

        provider = DeepSeekProvider(api_key=TEST_KEY)
        with patch("backend.providers.urlopen") as opener:
            with self.assertRaises(DeepSeekProviderError):
                provider.complete([])
            with self.assertRaises(DeepSeekProviderError):
                provider.complete([{"role": "user", "content": "ok"}], temperature=float("nan"))
        opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
