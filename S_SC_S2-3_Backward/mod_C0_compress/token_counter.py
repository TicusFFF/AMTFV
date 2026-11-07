from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


TOKEN_COUNT_MISMATCH_TOLERANCE = 100


class TokenCounterError(ValueError):
    """Base error for strict model token counting."""


class TokenCounterConfigurationError(
    TokenCounterError
):
    pass


class TokenCounterUnavailableError(
    TokenCounterError
):
    pass


class TokenCounterMismatchError(
    TokenCounterError
):
    pass


def _llm_timeout_seconds() -> float:
    try:
        return max(
            1.0,
            float(
                os.environ.get(
                    "SPQ_LLM_TIMEOUT_SECONDS",
                    "120",
                )
            ),
        )
    except ValueError:
        return 120.0


def _llm_max_retries() -> int:
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "SPQ_LLM_MAX_RETRIES",
                    "20",
                )
            ),
        )
    except ValueError:
        return 20


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )
    if callable(model_dump):
        try:
            return _json_safe(
                model_dump(mode="json")
            )
        except TypeError:
            return _json_safe(
                model_dump()
            )

    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _expand_environment_value(
    value: str,
    *,
    field_name: str,
) -> str:
    value = str(value or "").strip()
    expanded = os.path.expandvars(value)

    if (
        not expanded
        or ("$" + "{") in expanded
        or "%" in expanded
    ):
        raise (
            TokenCounterConfigurationError(
                f"{field_name} is missing or "
                "contains an unresolved "
                "environment variable: "
                f"{value!r}."
            )
        )

    return expanded


def _require_file_sha256(
    path: Path,
    expected: Any,
    *,
    field_name: str,
) -> str:
    expected = str(
        expected or ""
    ).strip().lower()

    if len(expected) != 64:
        raise (
            TokenCounterConfigurationError(
                f"{field_name} must contain "
                "an explicit SHA-256 digest."
            )
        )

    try:
        int(expected, 16)
    except ValueError as error:
        raise (
            TokenCounterConfigurationError(
                f"{field_name} is not a "
                "valid SHA-256 digest."
            )
        ) from error

    if not path.is_file():
        raise (
            TokenCounterUnavailableError(
                "Cannot find the explicitly "
                "configured tokenizer file: "
                f"{path}"
            )
        )

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    actual = digest.hexdigest()

    if actual != expected:
        raise (
            TokenCounterUnavailableError(
                "The configured tokenizer "
                "resource does not match its "
                "verified SHA-256 digest: "
                f"path={path}, "
                f"expected={expected}, "
                f"actual={actual}."
            )
        )

    return actual


def _vllm_tokenize_url(
    base_url: str,
) -> str:
    base_url = str(
        base_url or ""
    ).rstrip("/")

    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    if not base_url:
        raise (
            TokenCounterConfigurationError(
                "The vLLM tokenizer backend "
                "requires a non-empty base_url."
            )
        )

    return f"{base_url}/tokenize"


@dataclass(frozen=True)
class TokenCounterIdentity:
    profile: str
    api_model: str
    base_url: str
    backend: str
    extra_body_json: str
    counter_config_json: str

    def as_key(self) -> str:
        return _stable_json(
            {
                "profile": self.profile,
                "api_model": self.api_model,
                "base_url": self.base_url,
                "backend": self.backend,
                "extra_body": (
                    self.extra_body_json
                ),
                "counter_config": (
                    self.counter_config_json
                ),
            }
        )


class ModelTokenCounter:
    """Strict, model-scoped token counter."""

    _SUPPORTED_BACKENDS = {
        "vllm_tokenize",
        "huggingface",
        "deepseek_v4",
        "tiktoken",
    }

    def __init__(
        self,
        *,
        model_alias: str,
        api_model: str,
        api_config: dict[str, Any],
        extra_body: (
            dict[str, Any] | None
        ),
        counter_config: dict[str, Any],
        require_verified: bool = True,
    ):
        self.model_alias = str(
            model_alias
        )
        self.api_model = str(
            api_model
        )
        self.api_config = dict(
            api_config
        )
        self.extra_body = dict(
            extra_body or {}
        )
        self.counter_config = dict(
            counter_config or {}
        )

        self.backend = str(
            self.counter_config.get(
                "backend",
                "",
            )
        ).strip()

        if (
            self.backend
            not in self._SUPPORTED_BACKENDS
        ):
            raise (
                TokenCounterConfigurationError(
                    "No valid token counter is "
                    "configured for "
                    f"{self.model_alias!r}. "
                    "A model must explicitly use "
                    "vllm_tokenize, huggingface, "
                    "deepseek_v4, or tiktoken. "
                    "Silent "
                    "tokenizer fallback is "
                    "forbidden."
                )
            )

        if (
            require_verified
            and self.counter_config.get(
                "verified"
            )
            is not True
        ):
            raise (
                TokenCounterConfigurationError(
                    "The token counter for "
                    f"{self.model_alias!r} is not "
                    "verified. Run the read-only "
                    "token-counter verifier before "
                    "the formal experiment."
                )
            )

        self.chat_template_kwargs = dict(
            self.extra_body.get(
                "chat_template_kwargs",
                {},
            )
            or {}
        )

        configured_template_kwargs = (
            self.counter_config.get(
                "chat_template_kwargs"
            )
        )
        if configured_template_kwargs:
            self.chat_template_kwargs.update(
                dict(
                    configured_template_kwargs
                )
            )

        self._lock = threading.RLock()
        self._cache: dict[str, int] = {}
        self.observed_context_window_tokens: int | None = None
        self._cache_limit = max(
            16,
            int(
                self.counter_config.get(
                    "cache_size",
                    512,
                )
            ),
        )

        self._encoding = None
        self._hf_tokenizer = None
        self._deepseek_encode_messages = None
        self._deepseek_thinking_mode = None
        self._deepseek_reasoning_effort = None

        if self.backend == "vllm_tokenize":
            self._initialize_vllm()

        elif self.backend == "huggingface":
            self._initialize_huggingface()

        elif self.backend == "deepseek_v4":
            self._initialize_deepseek_v4()

        elif self.backend == "tiktoken":
            self._initialize_tiktoken()

        identity_config = dict(
            self.counter_config
        )
        identity_config.pop(
            "verified",
            None,
        )

        self.identity = TokenCounterIdentity(
            profile=str(
                self.api_config.get(
                    "profile",
                    "",
                )
            ),
            api_model=self.api_model,
            base_url=str(
                self.api_config.get(
                    "base_url",
                    "",
                )
            ).rstrip("/"),
            backend=self.backend,
            extra_body_json=_stable_json(
                self.extra_body
            ),
            counter_config_json=(
                _stable_json(
                    identity_config
                )
            ),
        )

    @property
    def identity_key(self) -> str:
        return self.identity.as_key()

    @property
    def is_exact(self) -> bool:
        return True

    def _initialize_vllm(self) -> None:
        explicit_url = str(
            self.counter_config.get(
                "tokenize_url",
                "",
            )
        ).strip()

        self._tokenize_url = (
            _expand_environment_value(
                explicit_url,
                field_name="tokenize_url",
            )
            if explicit_url
            else _vllm_tokenize_url(
                self.api_config.get(
                    "base_url",
                    "",
                )
            )
        )
        self._tokenize_timeout = (
            _llm_timeout_seconds()
        )
        self._tokenize_max_retries = (
            _llm_max_retries()
        )

    def _initialize_huggingface(
        self,
    ) -> None:
        tokenizer_name = (
            _expand_environment_value(
                self.counter_config.get(
                    "tokenizer_name_or_path",
                    "",
                ),
                field_name=(
                    "tokenizer_name_or_path"
                ),
            )
        )

        try:
            from transformers import (
                AutoTokenizer,
            )
        except ImportError as error:
            raise (
                TokenCounterUnavailableError(
                    "The explicitly configured "
                    "Hugging Face tokenizer "
                    "requires transformers."
                )
            ) from error

        try:
            self._hf_tokenizer = (
                AutoTokenizer
                .from_pretrained(
                    tokenizer_name,
                    use_fast=bool(
                        self.counter_config.get(
                            "use_fast",
                            True,
                        )
                    ),
                    trust_remote_code=bool(
                        self.counter_config.get(
                            "trust_remote_code",
                            False,
                        )
                    ),
                    local_files_only=bool(
                        self.counter_config.get(
                            "local_files_only",
                            True,
                        )
                    ),
                )
            )
        except Exception as error:
            raise (
                TokenCounterUnavailableError(
                    "Cannot load the explicitly "
                    "configured tokenizer "
                    f"{tokenizer_name!r} for "
                    f"{self.model_alias!r}."
                )
            ) from error

    def _initialize_deepseek_v4(
        self,
    ) -> None:
        # Use the archived tokenizer for the official Flash alias.
        if self.api_model != "deepseek-flash":
            raise (
                TokenCounterConfigurationError(
                    f"Expected the official deepseek-flash alias, got {self.api_model!r}."
                )
            )

        tokenizer_name = (
            _expand_environment_value(
                self.counter_config.get(
                    "tokenizer_name_or_path",
                    "",
                ),
                field_name=(
                    "tokenizer_name_or_path"
                ),
            )
        )
        tokenizer_dir = Path(
            tokenizer_name
        )

        if not tokenizer_dir.is_dir():
            raise (
                TokenCounterUnavailableError(
                    "Cannot find the configured "
                    "DeepSeek-V4 tokenizer "
                    f"directory: {tokenizer_dir}"
                )
            )

        encoding_name = (
            _expand_environment_value(
                self.counter_config.get(
                    "encoding_module_path",
                    "",
                ),
                field_name=(
                    "encoding_module_path"
                ),
            )
        )
        encoding_path = Path(
            encoding_name
        )

        tokenizer_path = (
            tokenizer_dir
            / "tokenizer.json"
        )
        tokenizer_config_path = (
            tokenizer_dir
            / "tokenizer_config.json"
        )

        _require_file_sha256(
            tokenizer_path,
            self.counter_config.get(
                "tokenizer_sha256"
            ),
            field_name="tokenizer_sha256",
        )
        _require_file_sha256(
            tokenizer_config_path,
            self.counter_config.get(
                "tokenizer_config_sha256"
            ),
            field_name=(
                "tokenizer_config_sha256"
            ),
        )
        encoding_digest = (
            _require_file_sha256(
                encoding_path,
                self.counter_config.get(
                    "encoding_sha256"
                ),
                field_name=(
                    "encoding_sha256"
                ),
            )
        )

        self._initialize_huggingface()

        module_name = (
            "_spq_deepseek_v4_encoding_"
            + encoding_digest[:16]
        )

        try:
            spec = (
                importlib.util
                .spec_from_file_location(
                    module_name,
                    encoding_path,
                )
            )

            if (
                spec is None
                or spec.loader is None
            ):
                raise ImportError(
                    "Cannot create the module "
                    "specification."
                )

            module = (
                importlib.util
                .module_from_spec(spec)
            )
            spec.loader.exec_module(module)

            encode_messages = getattr(
                module,
                "encode_messages",
                None,
            )
        except Exception as error:
            raise (
                TokenCounterUnavailableError(
                    "Cannot load the official "
                    "DeepSeek-V4 message encoder "
                    f"from {encoding_path}."
                )
            ) from error

        if not callable(encode_messages):
            raise (
                TokenCounterUnavailableError(
                    "The official DeepSeek-V4 "
                    "encoding module does not "
                    "provide encode_messages()."
                )
            )

        thinking_mode = str(
            self.counter_config.get(
                "thinking_mode",
                "",
            )
        ).strip()

        if thinking_mode not in {
            "chat",
            "thinking",
        }:
            raise (
                TokenCounterConfigurationError(
                    "The deepseek_v4 backend "
                    "requires thinking_mode to "
                    "be 'chat' or 'thinking'."
                )
            )

        thinking = self.extra_body.get(
            "thinking"
        )

        if not isinstance(thinking, dict):
            raise (
                TokenCounterConfigurationError(
                    "The DeepSeek API request "
                    "must explicitly configure "
                    "thinking.type."
                )
            )

        thinking_type = str(
            thinking.get("type") or ""
        ).strip()

        expected_mode = {
            "disabled": "chat",
            "enabled": "thinking",
        }.get(thinking_type)

        if expected_mode is None:
            raise (
                TokenCounterConfigurationError(
                    "DeepSeek thinking.type must "
                    "be 'enabled' or 'disabled'."
                )
            )

        if thinking_mode != expected_mode:
            raise (
                TokenCounterConfigurationError(
                    "The configured DeepSeek "
                    "thinking_mode does not match "
                    "the actual API extra_body: "
                    f"configured={thinking_mode!r}, "
                    f"expected={expected_mode!r}."
                )
            )

        reasoning_effort = (
            self.counter_config.get(
                "encoding_reasoning_effort"
            )
        )

        if reasoning_effort == "":
            reasoning_effort = None

        if reasoning_effort not in {
            None,
            "high",
            "max",
        }:
            raise (
                TokenCounterConfigurationError(
                    "encoding_reasoning_effort "
                    "must be null, 'high', or "
                    f"'max', not "
                    f"{reasoning_effort!r}."
                )
            )

        if (
            thinking_mode == "chat"
            and reasoning_effort is not None
        ):
            raise (
                TokenCounterConfigurationError(
                    "Chat mode must not configure "
                    "encoding_reasoning_effort."
                )
            )

        self._deepseek_encode_messages = (
            encode_messages
        )
        self._deepseek_thinking_mode = (
            thinking_mode
        )
        self._deepseek_reasoning_effort = (
            reasoning_effort
        )

    def _initialize_tiktoken(
        self,
    ) -> None:
        encoding_name = str(
            self.counter_config.get(
                "encoding_name",
                "",
            )
        ).strip()

        if not encoding_name:
            raise (
                TokenCounterConfigurationError(
                    "The tiktoken backend "
                    "requires an explicit "
                    "encoding_name. Automatic "
                    "model lookup and fallback "
                    "are forbidden."
                )
            )

        for required_key in (
            "tokens_per_message",
            "tokens_per_name",
            "reply_primer_tokens",
        ):
            if (
                required_key
                not in self.counter_config
            ):
                raise (
                    TokenCounterConfigurationError(
                        "The tiktoken backend "
                        "requires an explicit "
                        f"{required_key} value."
                    )
                )

        try:
            import tiktoken
        except ImportError as error:
            raise (
                TokenCounterUnavailableError(
                    "The explicitly configured "
                    "tiktoken backend requires "
                    "tiktoken."
                )
            ) from error

        try:
            self._encoding = (
                tiktoken.get_encoding(
                    encoding_name
                )
            )
        except Exception as error:
            raise (
                TokenCounterUnavailableError(
                    "Cannot load the explicit "
                    "tiktoken encoding "
                    f"{encoding_name!r}."
                )
            ) from error

    def _cache_get(
        self,
        key: str,
    ) -> int | None:
        with self._lock:
            return self._cache.get(key)

    def _cache_set(
        self,
        key: str,
        value: int,
    ) -> int:
        with self._lock:
            if (
                len(self._cache)
                >= self._cache_limit
            ):
                self._cache.clear()

            self._cache[key] = int(value)

        return int(value)

    def _request_vllm_count(
        self,
        payload: dict[str, Any],
    ) -> int:
        headers = {
            "Content-Type": (
                "application/json"
            ),
        }

        api_key = str(
            self.api_config.get(
                "api_key",
                "",
            )
        ).strip()

        if api_key:
            headers[
                "Authorization"
            ] = f"Bearer {api_key}"

        try:
            response = requests.post(
                self._tokenize_url,
                headers=headers,
                json=payload,
                timeout=(
                    self._tokenize_timeout
                ),
            )
        except Exception as error:
            raise (
                TokenCounterUnavailableError(
                    "The configured vLLM "
                    "tokenizer endpoint is "
                    "unavailable: "
                    f"{self._tokenize_url}"
                )
            ) from error

        if response.status_code != 200:
            raise (
                TokenCounterUnavailableError(
                    "The configured vLLM "
                    "tokenizer endpoint returned "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )
            )

        try:
            data = response.json()
            count = data.get("count")
            if count is None:
                count = len(
                    data["tokens"]
                )
            count = int(count)
        except Exception as error:
            raise (
                TokenCounterUnavailableError(
                    "The vLLM /tokenize "
                    "response does not contain "
                    "a valid token count."
                )
            ) from error

        if count < 0:
            raise (
                TokenCounterUnavailableError(
                    "The vLLM tokenizer "
                    "returned a negative count."
                )
            )

        service_limit = data.get("max_model_len")
        if (
            isinstance(service_limit, int)
            and not isinstance(service_limit, bool)
            and service_limit > 0
        ):
            with self._lock:
                previous_limit = self.observed_context_window_tokens
                self.observed_context_window_tokens = (
                    service_limit
                    if previous_limit is None
                    else min(previous_limit, service_limit)
                )

        return count


    def _vllm_count(
        self,
        payload: dict[str, Any],
    ) -> int:
        cache_key = (
            "vllm:"
            + hashlib.sha256(
                _stable_json(payload).encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        cached = self._cache_get(
            cache_key
        )

        if cached is not None:
            return cached

        retries = 0

        while True:
            try:
                count = (
                    self._request_vllm_count(
                        payload
                    )
                )

                return self._cache_set(
                    cache_key,
                    count,
                )

            except TokenCounterUnavailableError as error:
                retries += 1

                if (
                    retries
                    >= self._tokenize_max_retries
                ):
                    raise

                wait_seconds = (
                    retries - 1
                ) * 10

                print(
                    "[C0] vLLM tokenizer request failed: "
                    f"{error}. Retrying in "
                    f"{wait_seconds} seconds "
                    f"({retries + 1}/"
                    f"{self._tokenize_max_retries})...",
                    flush=True,
                )

                time.sleep(
                    wait_seconds
                )

    def count_text_raw(
        self,
        text: Any,
    ) -> int:
        text = str(text or "")

        if self.backend == "vllm_tokenize":
            return self._vllm_count(
                {
                    "model": self.api_model,
                    "prompt": text,
                    "add_special_tokens": (
                        False
                    ),
                }
            )

        if self.backend in {
            "huggingface",
            "deepseek_v4",
        }:
            assert (
                self._hf_tokenizer
                is not None
            )
            return len(
                self._hf_tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )
            )

        assert self._encoding is not None
        return len(
            self._encoding.encode(
                text,
                disallowed_special=(),
            )
        )

    def count_text(
        self,
        text: Any,
    ) -> int:
        return self.count_text_raw(text)

    def count_request(
        self,
        request_payload: dict[str, Any],
    ) -> int:
        """Count the final Chat Completions input using the native counter."""
        messages = request_payload.get("messages")
        if not isinstance(messages, list):
            raise TokenCounterConfigurationError(
                "Chat request messages must be a list."
            )

        template_keys = (
            "add_generation_prompt",
            "continue_final_message",
            "add_special_tokens",
            "chat_template",
            "chat_template_kwargs",
            "media_io_kwargs",
            "mm_processor_kwargs",
            "tools",
        )

        if self.backend == "vllm_tokenize":
            payload = {
                "model": self.api_model,
                "messages": _json_safe(messages),
                "add_generation_prompt": True,
                "continue_final_message": False,
            }
            if self.chat_template_kwargs:
                payload["chat_template_kwargs"] = dict(
                    self.chat_template_kwargs
                )
            for key in template_keys:
                if key in request_payload:
                    payload[key] = _json_safe(request_payload[key])

            return self._vllm_count(payload)

        if any(
            key in request_payload and request_payload[key] is not None
            for key in template_keys
        ):
            raise TokenCounterConfigurationError(
                "The configured native counter does not support "
                "these request-level template/tool options."
            )

        return self.count_messages(messages)

    def count_messages_raw(
        self,
        messages: list[
            dict[str, Any]
        ],
    ) -> int:
        messages = [
            dict(message)
            for message in messages
        ]

        if self.counter_config.get("verified_message_format") == "text_chat":
            if any(
                message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or set(message) - {"role", "content"}
                for message in messages
            ):
                raise TokenCounterConfigurationError(
                    "This counter is verified only for text system/user/assistant messages."
                )
            max_system = self.counter_config.get("max_system_messages")
            if max_system is not None and sum(
                message.get("role") == "system" for message in messages
            ) > int(max_system):
                raise TokenCounterConfigurationError(
                    "Multiple system messages are outside this counter's verified scope."
                )
            if self.counter_config.get("reject_consecutive_assistant") and any(
                left.get("role") == right.get("role") == "assistant"
                for left, right in zip(messages, messages[1:])
            ):
                raise TokenCounterConfigurationError(
                    "Consecutive assistant messages are outside this counter's verified scope."
                )

        if self.backend == "vllm_tokenize":
            payload: dict[str, Any] = {
                "model": self.api_model,
                "messages": _json_safe(
                    messages
                ),
                "add_generation_prompt": (
                    True
                ),
                "continue_final_message": (
                    False
                ),
            }

            if self.chat_template_kwargs:
                payload[
                    "chat_template_kwargs"
                ] = dict(
                    self.chat_template_kwargs
                )

            return self._vllm_count(payload)

        if self.backend == "deepseek_v4":
            assert (
                self._hf_tokenizer
                is not None
            )
            assert (
                self._deepseek_encode_messages
                is not None
            )
            assert (
                self._deepseek_thinking_mode
                is not None
            )

            try:
                prompt = (
                    self
                    ._deepseek_encode_messages(
                        _json_safe(messages),
                        thinking_mode=(
                            self
                            ._deepseek_thinking_mode
                        ),
                        drop_thinking=True,
                        add_default_bos_token=(
                            True
                        ),
                        reasoning_effort=(
                            self
                            ._deepseek_reasoning_effort
                        ),
                    )
                )
            except Exception as error:
                raise (
                    TokenCounterUnavailableError(
                        "The official DeepSeek-V4 "
                        "message encoding failed "
                        f"for {self.model_alias!r}."
                    )
                ) from error

            if not isinstance(prompt, str):
                raise (
                    TokenCounterUnavailableError(
                        "The official DeepSeek-V4 "
                        "encoder did not return "
                        "text."
                    )
                )

            # Include the provider generation header for a trailing assistant message.
            header = self.counter_config.get("assistant_tail_generation_header", "")
            if header and messages and messages[-1].get("role") == "assistant":
                last = messages[-1]
                if (
                    self._deepseek_thinking_mode != "chat"
                    or last.get("prefix")
                    or last.get("wo_eos")
                    or last.get("tool_calls")
                ):
                    raise TokenCounterUnavailableError(
                        "The assistant-tail adapter is verified only for ordinary non-thinking chat."
                    )
                prompt += header

            return len(
                self._hf_tokenizer.encode(
                    prompt,
                    add_special_tokens=False,
                )
            )

        if self.backend == "huggingface":
            assert (
                self._hf_tokenizer
                is not None
            )

            try:
                token_ids = (
                    self._hf_tokenizer
                    .apply_chat_template(
                        _json_safe(messages),
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                        **self.chat_template_kwargs,
                    )
                )
            except Exception as error:
                raise (
                    TokenCounterUnavailableError(
                        "The explicitly "
                        "configured Hugging Face "
                        "chat template failed."
                    )
                ) from error

            return len(token_ids)

        assert self._encoding is not None

        tokens_per_message = int(
            self.counter_config[
                "tokens_per_message"
            ]
        )
        tokens_per_name = int(
            self.counter_config[
                "tokens_per_name"
            ]
        )
        total = int(
            self.counter_config[
                "reply_primer_tokens"
            ]
        )

        for message in messages:
            total += tokens_per_message
            total += int(
                self.counter_config.get(
                    "tokens_per_role",
                    {},
                ).get(message.get("role"), 0)
            )

            for key, value in (
                message.items()
            ):
                if value is None:
                    continue

                if isinstance(
                    value,
                    (dict, list, tuple),
                ):
                    value = _stable_json(
                        value
                    )

                total += self.count_text_raw(
                    value
                )

                if key == "name":
                    total += (
                        tokens_per_name
                    )

        return max(total, 0)

    def count_messages(
        self,
        messages: list[
            dict[str, Any]
        ],
    ) -> int:
        return self.count_messages_raw(
            messages
        )

    @property
    def token_safety_margin(self) -> int:
        # Keep the model-specific allowance and record any provider token-count differences.
        if (
            self.model_alias == "qdd_GPT56Luna"
            and self.api_model == "gpt-5.6-luna"
        ):
            return 512
        return 0

    def observe(
        self,
        messages: list[
            dict[str, Any]
        ],
        prompt_tokens: int,
    ) -> dict[str, Any]:
        try:
            prompt_tokens = int(
                prompt_tokens
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise (
                TokenCounterMismatchError(
                    "Provider prompt_tokens "
                    "must be a positive integer."
                )
            ) from error

        if prompt_tokens <= 0:
            raise (
                TokenCounterMismatchError(
                    "Provider prompt_tokens "
                    "must be positive."
                )
            )

        expected_tokens = (
            self.count_messages_raw(
                messages
            )
        )
        token_difference = (
            expected_tokens
            - prompt_tokens
        )

        tolerance = max(
            TOKEN_COUNT_MISMATCH_TOLERANCE,
            self.token_safety_margin,
        )
        if (
            abs(token_difference)
            > tolerance
        ):
            raise (
                TokenCounterMismatchError(
                    "Tokenizer mismatch exceeds "
                    "the allowed tolerance "
                    f"for {self.model_alias!r}: "
                    f"local={expected_tokens}, "
                    f"provider={prompt_tokens}, "
                    f"difference(local-provider)="
                    f"{token_difference}, "
                    f"tolerance="
                    f"{tolerance}. "
                    "Check the model identity, "
                    "chat template, and "
                    "chat_template_kwargs."
                )
            )

        if token_difference != 0:
            warnings.warn(
                (
                    "Tokenizer mismatch is within "
                    "the allowed tolerance for "
                    f"{self.model_alias!r}: "
                    f"local={expected_tokens}, "
                    f"provider={prompt_tokens}, "
                    f"difference(local-provider)="
                    f"{token_difference}, "
                    f"tolerance="
                    f"{tolerance}. "
                    "Using provider prompt_tokens."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        return {
            "backend": self.backend,
            "changed": False,
            "prompt_tokens": (
                prompt_tokens
            ),
            "estimated_tokens": (
                expected_tokens
            ),
            "token_difference": (
                token_difference
            ),
            "tolerance": (
                tolerance
            ),
        }


def build_token_counter(
    *,
    model_alias: str,
    api_model: str,
    api_config: dict[str, Any],
    extra_body: (
        dict[str, Any] | None
    ),
    counter_config: dict[str, Any],
    require_verified: bool = True,
) -> ModelTokenCounter:
    return ModelTokenCounter(
        model_alias=model_alias,
        api_model=api_model,
        api_config=api_config,
        extra_body=extra_body,
        counter_config=counter_config,
        require_verified=require_verified,
    )
