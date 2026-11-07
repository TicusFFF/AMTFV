from __future__ import annotations

import itertools
import json
import math
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from openai import OpenAI

from .capacity import load_effective_capacity
from .request_guard import configure_request_guard
from .token_counter import (
    ModelTokenCounter,
    TokenCounterError,
    build_token_counter,
)


def _consume_sample_llm_call() -> None:
    try:
        from utils.sample_budget import consume_llm_call
    except ImportError:
        return
    consume_llm_call()


def _gpu_queue_client_kwargs(base_url: str) -> dict:
    try:
        from utils.sample_budget import gpu_queue_client_kwargs
    except ImportError:
        return {}
    return gpu_queue_client_kwargs(base_url)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PUBLIC_CAPACITY_RELATIVE_PATH = (
    Path("0-4_API")
    / "2_model_capacity"
    / "model_capacities.json"
)

DEFAULT_AUTO_TRIGGER_TOKENS = 131_072
_COMPRESSOR_CONTEXT_SAFETY_TOKENS = 128
_WORK_OUTPUT_RESERVE_TOKENS = 8_192
_WORK_CONTEXT_SAFETY_TOKENS = 8_192

_TRACE_PATH = ContextVar("c0_trace_path", default=None)
_SAMPLE_ID = ContextVar("c0_sample_id", default="")

_RUNTIME_LOCK = threading.RLock()
_TRACE_LOCK = threading.Lock()
_SEQUENCE = itertools.count(1)

_RUNTIME: "_ContextRuntime | None" = None


class ContextWindowError(RuntimeError):
    pass


class _CompressionInputTooLongError(
    ContextWindowError
):
    pass


def _build_compression_retry_messages(
    original_messages: list[dict[str, str]],
    retry_requirements: list[str],
) -> list[dict[str, str]]:
    """Append cumulative requirements only after content validation failures."""
    if not retry_requirements:
        return original_messages

    retry_messages = deepcopy(
        original_messages
    )
    retry_messages[-1]["content"] = (
        f"{retry_messages[-1]['content']}\n\n"
        + "\n\n".join(
            retry_requirements
        )
    )
    return retry_messages


@dataclass(frozen=True)
class _ModelSpec:
    alias: str
    api_model: str
    api_config: dict[str, Any]
    extra_body: dict[str, Any] | None
    token_counting: dict[str, Any]
    context_window_tokens: int | None
    max_output_tokens: int | None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": str(value),
        }

    if is_dataclass(value):
        return _json_safe(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"))
        except TypeError:
            return _json_safe(model_dump())

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    return repr(value)


_CONTEXT_LENGTH_ERROR_MARKERS = (
    "context_length_exceeded",
    "context window exceeded",
    "context window is exceeded",
    "maximum context length",
    "maximum context window",
    "maximum sequence length",
    "maximum model length",
    "max_model_len",
    "prompt is too long",
    "prompt too long",
    "input is too long",
    "input too long",
    "too many input tokens",
    "too many tokens",
    "token limit exceeded",
    "input tokens exceed",
    "exceeds the maximum number of tokens",
)

_EVENT_BOUNDARY_RE = re.compile(
    r"(?=^\[(?:system|user|assistant|tool|"
    r"Timing Event|PythonTool Event|"
    r"Verification Result|Stop Reason|"
    r"LLM Error|Skip Reason)\][ \t]*$)",
    re.MULTILINE | re.IGNORECASE,
)


def _is_context_length_error(
    error: Exception,
) -> bool:
    if isinstance(
        error,
        _CompressionInputTooLongError,
    ):
        return True

    values: list[Any] = [
        str(error),
        getattr(error, "code", ""),
        getattr(error, "type", ""),
        getattr(error, "body", ""),
    ]

    text = " ".join(
        json.dumps(
            _json_safe(value),
            ensure_ascii=False,
        )
        if isinstance(
            value,
            (dict, list, tuple),
        )
        else str(value or "")
        for value in values
    ).lower()

    return any(
        marker in text
        for marker in (
            _CONTEXT_LENGTH_ERROR_MARKERS
        )
    )


def _split_event_units(
    text: str,
) -> list[str]:
    text = str(text or "")

    units = [
        part
        for part in (
            _EVENT_BOUNDARY_RE.split(text)
        )
        if part
    ]

    return units or [text]


def _split_text_to_token_limit(
    text: str,
    *,
    counter: ModelTokenCounter,
    token_limit: int,
) -> list[str]:
    text = str(text or "")
    token_limit = max(
        1,
        int(token_limit),
    )

    if (
        counter.count_text(text)
        <= token_limit
    ):
        return [text]

    pieces: list[str] = []
    remaining = text

    while (
        remaining
        and counter.count_text(remaining)
        > token_limit
    ):
        low = 1
        high = len(remaining)
        best = 0

        while low <= high:
            middle = (
                low + high
            ) // 2

            if (
                counter.count_text(
                    remaining[:middle]
                )
                <= token_limit
            ):
                best = middle
                low = middle + 1
            else:
                high = middle - 1

        if best <= 0:
            best = 1

        newline = remaining.rfind(
            "\n",
            0,
            best + 1,
        )

        if newline >= max(
            1,
            best // 2,
        ):
            best = newline + 1

        pieces.append(
            remaining[:best]
        )
        remaining = remaining[best:]

    if remaining:
        pieces.append(remaining)

    return pieces


def _pack_source_units(
    units: list[str],
    *,
    counter: ModelTokenCounter,
    token_limit: int,
) -> list[str]:
    token_limit = max(
        1,
        int(token_limit),
    )

    chunks: list[str] = []
    current = ""

    for unit in units:
        for piece in (
            _split_text_to_token_limit(
                unit,
                counter=counter,
                token_limit=token_limit,
            )
        ):
            candidate = (
                current + piece
            )

            if (
                current
                and counter.count_text(
                    candidate
                )
                > token_limit
            ):
                chunks.append(current)
                current = piece
            else:
                current = candidate

    if current:
        chunks.append(current)

    return chunks


class _ContextRuntime:
    def __init__(
        self,
        *,
        enabled: bool,
        trace_enabled: bool,
        trace_root: Path,
        system_prompt_template: str,
        user_prompt_template: str,
        trigger_tokens: int | None,
        work_model: _ModelSpec | None,
        compressor_model: _ModelSpec | None,
    ):
        self.enabled = enabled
        self.trace_enabled = trace_enabled
        self.trace_root = trace_root
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.trigger_tokens = trigger_tokens
        self.work_model = work_model
        self.compressor_model = compressor_model

        self.work_counter = (
            build_token_counter(
                model_alias=work_model.alias,
                api_model=work_model.api_model,
                api_config=work_model.api_config,
                extra_body=work_model.extra_body,
                counter_config=(
                    work_model.token_counting
                ),
            )
            if work_model is not None
            else None
        )

        compressor_counter = (
            build_token_counter(
                model_alias=compressor_model.alias,
                api_model=compressor_model.api_model,
                api_config=(
                    compressor_model.api_config
                ),
                extra_body=(
                    compressor_model.extra_body
                ),
                counter_config=(
                    compressor_model
                    .token_counting
                ),
            )
            if compressor_model is not None
            else None
        )

        if (
            self.work_counter is not None
            and compressor_counter is not None
            and self.work_counter.identity_key
            == compressor_counter.identity_key
        ):
            self.compressor_counter = (
                self.work_counter
            )
        else:
            self.compressor_counter = (
                compressor_counter
            )

        self.client = None

        if compressor_model is not None:
            self.client = OpenAI(
                api_key=compressor_model.api_config["api_key"],
                base_url=compressor_model.api_config["base_url"],
                timeout=_compressor_timeout_seconds(),
                max_retries=0,
                default_headers={
                    "X-SPQ-Context-Compression": "1",
                },
                **_gpu_queue_client_kwargs(compressor_model.api_config["base_url"]),
            )

    def _same_token_unit(self) -> bool:
        return (
            self.work_counter is not None
            and self.compressor_counter
            is not None
            and self.work_counter.identity_key
            == self.compressor_counter
            .identity_key
        )

    def record(
        self,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.trace_enabled:
            return

        trace_path = _TRACE_PATH.get()
        if trace_path is None:
            trace_path = self.trace_root / "raw_trajectory.jsonl"

        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "sequence": next(_SEQUENCE),
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "sample_id": _SAMPLE_ID.get(),
            "event_type": event_type,
            "payload": _json_safe(payload),
        }

        line = json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with _TRACE_LOCK:
            with trace_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(line)
                handle.write("\n")
                handle.flush()

    def should_compress(
        self,
        messages: list[dict[str, Any]],
    ) -> bool:
        if (
            not self.enabled
            or self.trigger_tokens is None
            or self.work_counter is None
        ):
            return False

        input_tokens = self.work_counter.count_messages(messages)

        return input_tokens >= self.trigger_tokens

    def safe_input_tokens(self) -> int | None:
        if (
            self.work_model is None
            or self.work_model.context_window_tokens is None
        ):
            return None

        return max(
            1,
            int(self.work_model.context_window_tokens)
            - _WORK_OUTPUT_RESERVE_TOKENS
            - _WORK_CONTEXT_SAFETY_TOKENS,
        )

    def _build_compression_messages(
        self,
        source: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    self.system_prompt_template
                ),
            },
            {
                "role": "user",
                "content": (
                    self.user_prompt_template
                    .replace(
                        "{context}",
                        source,
                    )
                ),
            },
        ]

    def _source_chunk_token_limit(
        self,
    ) -> int:
        if (
            self.compressor_model is None
            or self.compressor_counter
            is None
        ):
            raise ContextWindowError(
                "Compressor model is "
                "unavailable."
            )

        token_limit = int(
            self.trigger_tokens
            or DEFAULT_AUTO_TRIGGER_TOKENS
        )

        if (
            self.compressor_model
            .context_window_tokens
            is not None
        ):
            prompt_overhead = (
                self.compressor_counter
                .count_messages(
                    self
                    ._build_compression_messages(
                        ""
                    )
                )
            )

            available = (
                int(
                    self.compressor_model
                    .context_window_tokens
                )
                - prompt_overhead
                - 1
            )

            if available <= 0:
                raise ContextWindowError(
                    "The compression prompt "
                    "already reaches the "
                    "compressor model context "
                    "window."
                )

            token_limit = min(
                token_limit,
                available,
            )

        return max(
            1,
            token_limit,
        )

    def compress_source_chunked(
        self,
        *,
        stage: str,
        source: str,
        summary_cap: int,
        summary_validator: (
            Callable[[str], bool] | None
        ) = None,
        source_units: (
            list[str] | None
        ) = None,
    ) -> str:
        input_error: (
            _CompressionInputTooLongError
            | None
        ) = None

        try:
            return self.compress_source(
                stage=stage,
                source=source,
                summary_cap=summary_cap,
                summary_validator=(
                    summary_validator
                ),
            )

        except TokenCounterError:
            raise

        except (
            _CompressionInputTooLongError
        ) as error:
            input_error = error

        assert input_error is not None

        if self.compressor_counter is None:
            raise ContextWindowError(
                "Compressor model is "
                "unavailable."
            )

        units = list(
            source_units
            or _split_event_units(source)
        )

        if "".join(units) != source:
            units = [source]

        chunk_token_limit = (
            self._source_chunk_token_limit()
        )

        pending = _pack_source_units(
            units,
            counter=(
                self.compressor_counter
            ),
            token_limit=(
                chunk_token_limit
            ),
        )

        if len(pending) <= 1:
            raise input_error

        self.record(
            "compression.chunking",
            {
                "stage": stage,
                "source": source,
                "chunk_count": len(
                    pending
                ),
                "chunk_token_limit": (
                    chunk_token_limit
                ),
                "reason": input_error,
            },
        )

        summaries: list[str] = []
        successful_chunks = 0

        while pending:
            chunk = pending.pop(0)

            try:
                summary = (
                    self.compress_source(
                        stage=(
                            f"{stage}.chunk_"
                            f"{successful_chunks + 1}"
                        ),
                        source=chunk,
                        summary_cap=(
                            summary_cap
                        ),
                    )
                )

            except TokenCounterError:
                raise

            except (
                _CompressionInputTooLongError
            ):
                chunk_tokens = (
                    self.compressor_counter
                    .count_text(chunk)
                )

                smaller_limit = max(
                    1,
                    min(
                        chunk_token_limit,
                        chunk_tokens // 2,
                    ),
                )

                smaller_chunks = (
                    _pack_source_units(
                        [chunk],
                        counter=(
                            self
                            .compressor_counter
                        ),
                        token_limit=(
                            smaller_limit
                        ),
                    )
                )

                if (
                    len(smaller_chunks)
                    <= 1
                ):
                    raise

                pending[0:0] = (
                    smaller_chunks
                )
                continue

            summaries.append(summary)
            successful_chunks += 1

        merged = "\n\n".join(
            summaries
        ).strip()

        merged_tokens = (
            self.work_counter
            .count_text(merged)
        )
        source_tokens = (
            self.work_counter
            .count_text(source)
        )

        merged_fits = (
            bool(merged)
            and merged_tokens
            < source_tokens
            and merged_tokens
            <= summary_cap
            and (
                summary_validator is None
                or summary_validator(
                    merged
                )
            )
        )

        if merged_fits:
            self.record(
                "compression.chunked_success",
                {
                    "stage": stage,
                    "source": source,
                    "source_tokens": (
                        source_tokens
                    ),
                    "summary": merged,
                    "summary_tokens": (
                        merged_tokens
                    ),
                    "chunk_count": (
                        successful_chunks
                    ),
                },
            )
            return merged

        if (
            not merged
            or merged_tokens
            >= source_tokens
        ):
            raise RuntimeError(
                "Chunked compression did "
                "not reduce the context."
            )

        merged_units = [
            summary
            if index == 0
            else "\n\n" + summary
            for index, summary
            in enumerate(summaries)
        ]

        return (
            self.compress_source_chunked(
                stage=f"{stage}.merge",
                source=merged,
                summary_cap=summary_cap,
                summary_validator=(
                    summary_validator
                ),
                source_units=merged_units,
            )
        )

    def compress_source(
        self,
        *,
        stage: str,
        source: str,
        summary_cap: int,
        summary_validator: (
            Callable[[str], bool] | None
        ) = None,
    ) -> str:
        if (
            self.compressor_model is None
            or self.compressor_counter
            is None
            or self.work_counter is None
            or self.client is None
        ):
            raise ContextWindowError(
                "Compressor model is "
                "unavailable."
            )

        compression_messages = (
            self
            ._build_compression_messages(
                source
            )
        )

        compressor_input_tokens = (
            self.compressor_counter
            .count_messages(
                compression_messages
            )
        )

        compressor_remaining = None

        if (
            self.compressor_model
            .context_window_tokens
            is not None
        ):
            compressor_remaining = (
                self.compressor_model
                .context_window_tokens
                - compressor_input_tokens
                - max(
                    _COMPRESSOR_CONTEXT_SAFETY_TOKENS,
                    self.compressor_counter.token_safety_margin,
                )
            )

        if (
            compressor_remaining is not None
            and compressor_remaining <= 0
        ):
            raise (
                _CompressionInputTooLongError(
                    "The compression input "
                    "itself exceeds the "
                    "compressor model context "
                    "window."
                )
            )

        if summary_cap <= 0:
            raise ContextWindowError(
                "The compression output token "
                "budget is empty."
            )

        request_kwargs: dict[str, Any] = {
            "model": (
                self.compressor_model
                .api_model
            ),
            "messages": (
                compression_messages
            ),
        }

        temperature = os.environ.get(
            "SKILLOPT_OPENAI_TEMPERATURE",
            "",
        ).strip()

        if temperature:
            request_kwargs["temperature"] = float(
                temperature
            )

        # Bound compression by model capacity, remaining context, and the summary target.
        max_token_limits: list[int] = []

        if self._same_token_unit():
            max_token_limits.append(
                summary_cap
            )

        if (
            self.compressor_model
            .max_output_tokens
            is not None
        ):
            max_token_limits.append(
                self.compressor_model
                .max_output_tokens
            )

        if (
            compressor_remaining
            is not None
        ):
            max_token_limits.append(
                compressor_remaining
            )

        if not max_token_limits:
            raise ContextWindowError(
                "The compressor output capacity "
                "is unknown."
            )

        max_tokens = min(max_token_limits)

        if max_tokens <= 0:
            raise ContextWindowError(
                "The compression output "
                "token budget is empty."
            )

        request_kwargs[
            "max_tokens"
        ] = int(max_tokens)

        current_request_max_tokens = int(
            max_tokens
        )

        if (
            self.compressor_model
            .extra_body
        ):
            request_kwargs[
                "extra_body"
            ] = (
                self.compressor_model
                .extra_body
            )

        max_attempts = (
            _llm_max_attempts()
        )

        source_tokens = (
            self.work_counter
            .count_text(source)
        )

        self.record(
            "compression.start",
            {
                "stage": stage,
                "model": (
                    self.compressor_model
                    .alias
                ),
                "source": source,
                "source_tokens": (
                    source_tokens
                ),
                "summary_cap": (
                    summary_cap
                ),
                "compressor_input_tokens": (
                    compressor_input_tokens
                ),
                "request_max_tokens": (
                    max_tokens
                ),
                "max_attempts": (
                    max_attempts
                ),
            },
        )

        last_error: (
            Exception | None
        ) = None

        retry_requirements: list[str] = []

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                attempt_messages = (
                    _build_compression_retry_messages(
                        compression_messages,
                        retry_requirements,
                    )
                )

                attempt_input_tokens = (
                    self.compressor_counter
                    .count_messages(
                        attempt_messages
                    )
                )
                attempt_max_tokens = (
                    current_request_max_tokens
                )

                if (
                    self.compressor_model
                    .context_window_tokens
                    is not None
                ):
                    attempt_remaining = (
                        self.compressor_model
                        .context_window_tokens
                        - attempt_input_tokens
                        - max(
                            _COMPRESSOR_CONTEXT_SAFETY_TOKENS,
                            self.compressor_counter.token_safety_margin,
                        )
                    )

                    if attempt_remaining <= 0:
                        raise (
                            _CompressionInputTooLongError(
                                "The compression input "
                                "itself exceeds the "
                                "compressor model context "
                                "window."
                            )
                        )

                    attempt_max_tokens = min(
                        attempt_max_tokens,
                        attempt_remaining,
                    )

                if attempt_max_tokens <= 0:
                    raise ContextWindowError(
                        "The compression output "
                        "token budget is empty."
                    )

                attempt_request_kwargs = dict(
                    request_kwargs
                )
                attempt_request_kwargs[
                    "messages"
                ] = attempt_messages
                attempt_request_kwargs[
                    "max_tokens"
                ] = int(attempt_max_tokens)

                _consume_sample_llm_call()
                response = (
                    _create_without_spq_override(
                        self.client,
                        **attempt_request_kwargs,
                    )
                )

                from api_profiles import record_response_metadata
                record_response_metadata(response, stage="compression", model=attempt_request_kwargs["model"])

                message = (
                    response
                    .choices[0]
                    .message
                )

                summary = str(
                    getattr(
                        message,
                        "content",
                        "",
                    )
                    or ""
                ).strip()

                usage = _usage_dict(
                    response
                )

                if (
                    usage["prompt_tokens"]
                    > 0
                ):
                    validation = (
                        self.compressor_counter
                        .observe(
                            attempt_messages,
                            usage[
                                "prompt_tokens"
                            ],
                        )
                    )
                    self.record(
                        "token_counter.validated",
                        {
                            "stage": stage,
                            "model_kind": (
                                "compressor"
                            ),
                            "validation": validation,
                        },
                    )

                if not summary:
                    retry_requirements.append(
                        "## Additional Compression Requirement\n\n"
                        "The previous response was empty. Use the complete "
                        "original context and preservation requirements above, "
                        "and return non-empty compressed text only."
                    )
                    raise RuntimeError(
                        "The compression model "
                        "returned empty content."
                    )

                summary_tokens = (
                    self.work_counter
                    .count_text(summary)
                )

                if (
                    summary_tokens
                    >= source_tokens
                ):
                    retry_requirements.append(
                        "## Additional Compression Requirement\n\n"
                        "The previous response did not shorten the context. "
                        "Use the complete original context and preservation "
                        "requirements above, compress it substantially, and "
                        "return only the compressed text."
                    )
                    raise RuntimeError(
                        "The compression result "
                        "did not reduce the "
                        "context."
                    )

                if (
                    summary_tokens
                    > summary_cap
                ):
                    if not self._same_token_unit():
                        completion_tokens = int(
                            usage.get(
                                "completion_tokens",
                                0,
                            )
                            or self.compressor_counter
                            .count_text(summary)
                        )
                        scaled_max_tokens = max(
                            1,
                            math.floor(
                                completion_tokens
                                * summary_cap
                                / summary_tokens
                            ),
                        )
                        current_request_max_tokens = min(
                            max(
                                1,
                                current_request_max_tokens
                                - 1,
                            ),
                            scaled_max_tokens,
                        )

                    retry_requirements.append(
                        "## Additional Compression Requirement\n\n"
                        "The previous response exceeded the available output "
                        f"budget. Keep the compressed text safely below "
                        f"{summary_cap} tokens while preserving the required "
                        "information, and output only the compressed text."
                    )
                    raise RuntimeError(
                        "The compression result "
                        "exceeds the available "
                        "summary token budget."
                    )

                if (
                    summary_validator
                    is not None
                    and not summary_validator(
                        summary
                    )
                ):
                    retry_requirements.append(
                        "## Additional Compression Requirement\n\n"
                        "The previous response did not sufficiently reduce the "
                        "resulting working context. Compress more aggressively "
                        "while preserving the required information, and output "
                        "only the shorter compressed text."
                    )
                    raise RuntimeError(
                        "The compressed working context was not reduced "
                        "sufficiently."
                    )

                self.record(
                    "compression.success",
                    {
                        "stage": stage,
                        "model": (
                            self
                            .compressor_model
                            .alias
                        ),
                        "attempt": attempt,
                        "max_attempts": (
                            max_attempts
                        ),
                        "source": source,
                        "summary": summary,
                        "usage": usage,
                        "raw_response": (
                            response
                        ),
                    },
                )

                return summary

            except TokenCounterError:
                raise

            except Exception as error:
                context_length_error = (
                    _is_context_length_error(
                        error
                    )
                )

                will_retry = (
                    not context_length_error
                    and attempt
                    < max_attempts
                )

                last_error = error

                self.record(
                    "compression.error",
                    {
                        "stage": stage,
                        "model": (
                            self
                            .compressor_model
                            .alias
                        ),
                        "attempt": attempt,
                        "max_attempts": (
                            max_attempts
                        ),
                        "will_retry": (
                            will_retry
                        ),
                        "source": source,
                        "error": error,
                    },
                )

                if context_length_error:
                    if isinstance(
                        error,
                        _CompressionInputTooLongError,
                    ):
                        raise

                    raise (
                        _CompressionInputTooLongError(
                            str(error)
                        )
                    ) from error

                if will_retry:
                    time.sleep(
                        min(
                            2 ** (
                                attempt - 1
                            ),
                            30,
                        )
                    )

        if last_error is None:
            raise RuntimeError(
                "Context compression failed."
            )

        raise last_error


def _timeout_seconds() -> float:
    try:
        return max(
            1.0,
            float(os.environ.get("SPQ_LLM_TIMEOUT_SECONDS", "120")),
        )
    except ValueError:
        return 120.0


def _compressor_timeout_seconds() -> float:
    """Reuse the project's existing long-request timeout for C0 calls."""
    try:
        long_timeout = max(
            1.0,
            float(
                os.environ.get(
                    "SPQ_DEEPSEEK_LONG_TIMEOUT_SECONDS",
                    "2400",
                )
            ),
        )
    except ValueError:
        long_timeout = 2400.0

    return max(_timeout_seconds(), long_timeout)


def _llm_max_attempts() -> int:
    try:
        return max(
            2,
            int(os.environ.get("SPQ_LLM_MAX_RETRIES", "5")),
        )
    except ValueError:
        return 5


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(
            getattr(usage, "completion_tokens", 0) or 0
        ),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _create_without_spq_override(
    client: OpenAI,
    **kwargs: Any,
) -> Any:
    create = client.chat.completions.create
    function = getattr(create, "__func__", None)
    original = getattr(function, "_spq_original", None)

    if callable(original):
        return original(create.__self__, **kwargs)

    return create(**kwargs)


def _resolve_path(
    value: str | Path,
    default: Path,
) -> Path:
    text = str(value or "").strip()
    path = Path(text) if text else default

    if not path.is_absolute():
        path = _PROJECT_ROOT / path

    return path.resolve()


def _find_public_capacity_path() -> Path:
    for root in (
        _PROJECT_ROOT,
        *_PROJECT_ROOT.parents,
    ):
        candidate = (
            root
            / _PUBLIC_CAPACITY_RELATIVE_PATH
        )

        if candidate.is_file():
            return candidate.resolve()

    return (
        _PROJECT_ROOT.parent
        / _PUBLIC_CAPACITY_RELATIVE_PATH
    ).resolve()


def _capacity_error(
    *,
    error: Exception,
    capacities_path: Path,
    profile: str,
    model_alias: str,
) -> RuntimeError:
    public_path = (
        _find_public_capacity_path()
    )
    tool_path = (
        public_path.parent
        / "model_capacity_tool.py"
    )

    return RuntimeError(
        f"{error}\n"
        "Public model-capacity registry:\n"
        f"{capacities_path}\n"
        "Inspect the shared registry:\n"
        f'python "{tool_path}" show '
        f'--profile "{profile}" '
        f'--model "{model_alias}"\n'
        "If official values are known, use "
        "the declare command. Otherwise use "
        "the probe command before the formal "
        "experiment."
    )


def _resolve_extra_body(
    api_profiles: Any,
    alias: str,
    api_config: dict[str, Any],
    api_model: str,
) -> dict[str, Any] | None:
    try:
        value = api_profiles.extra_body_for_model(
            alias,
            api_config,
            api_model=api_model,
        )
    except TypeError:
        value = api_profiles.extra_body_for_model(alias, api_config)

    if value is None:
        return None

    if not isinstance(value, dict):
        raise TypeError(
            "extra_body_for_model() must return dict or None."
        )

    return dict(value)


def _resolve_trigger_tokens(
    value: Any,
    *,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "compression.trigger_tokens must be "
            "a positive integer or 'auto'."
        )

    if str(value).strip().lower() == "auto":
        if (
            context_window_tokens is None
            or max_output_tokens is None
        ):
            raise ValueError(
                "Cannot resolve compression.trigger_tokens=auto: "
                "the work model context window and maximum "
                "output must be known."
            )

        usable_input_tokens = (
            int(context_window_tokens)
            - int(max_output_tokens)
        )

        if usable_input_tokens <= 0:
            raise ValueError(
                "Cannot resolve compression.trigger_tokens=auto: "
                "context_window_tokens must be greater than "
                "max_output_tokens."
            )

        return min(
            DEFAULT_AUTO_TRIGGER_TOKENS,
            usable_input_tokens,
        )

    try:
        trigger_tokens = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "compression.trigger_tokens must be "
            "a positive integer or 'auto'."
        ) from error

    if trigger_tokens <= 0:
        raise ValueError(
            "compression.trigger_tokens must be positive."
        )

    if (
        context_window_tokens is not None
        and trigger_tokens >= int(context_window_tokens)
    ):
        raise ValueError(
            "compression.trigger_tokens must be smaller "
            "than the work model context window."
        )

    return trigger_tokens


def configure_context_runtime(
    *,
    config: dict[str, Any] | None,
    output_root: str | Path,
    api_profiles: Any,
    work_model_alias: str,
    work_api_model: str,
    api_config: dict[str, Any],
    work_extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    global _RUNTIME

    config = dict(config or {})
    enabled = bool(
        config.get(
            "enabled",
            False,
        )
    )
    trace_enabled = bool(
        config.get(
            "trace_enabled",
            False,
        )
    )
    trigger_setting = config.get(
        "trigger_tokens",
        DEFAULT_AUTO_TRIGGER_TOKENS,
    )

    capacities_path = _resolve_path(
        config.get(
            "capacities_path",
            "",
        ),
        _find_public_capacity_path(),
    )
    prompt_path = _resolve_path(
        config.get(
            "prompt_path",
            "",
        ),
        _PROJECT_ROOT
        / "mod_C0_compress"
        / "prompts"
        / "context_compress.md",
    )
    user_prompt_path = _resolve_path(
        config.get(
            "user_prompt_path",
            "",
        ),
        _PROJECT_ROOT
        / "mod_C0_compress"
        / "prompts"
        / "context_compress_user.md",
    )

    work_model = None
    compressor_model = None
    resolved_trigger_tokens = None
    system_prompt_template = ""
    user_prompt_template = ""

    if enabled:
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Cannot find C0 system prompt: "
                f"{prompt_path}"
            )

        if not user_prompt_path.is_file():
            raise FileNotFoundError(
                f"Cannot find C0 user prompt: "
                f"{user_prompt_path}"
            )

        system_prompt_template = (
            prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        )
        user_prompt_template = (
            user_prompt_path.read_text(
                encoding="utf-8"
            ).strip()
        )

        if not system_prompt_template:
            raise ValueError(
                "The C0 system prompt must not be empty."
            )

        if "{context}" in system_prompt_template:
            raise ValueError(
                "{context} must not appear in the "
                "C0 system prompt."
            )

        if user_prompt_template.count(
            "{context}"
        ) != 1:
            raise ValueError(
                "The C0 user prompt must contain "
                "exactly one {context} placeholder."
            )

        profile = str(api_config.get("profile") or "")
        base_url = str(api_config.get("base_url") or "")

        try:
            work_capacity = load_effective_capacity(
                capacities_path,
                profile=profile,
                model_alias=work_model_alias,
                api_model=work_api_model,
                base_url=base_url,
                extra_body=work_extra_body,
            )
        except (
            FileNotFoundError,
            KeyError,
            ValueError,
        ) as error:
            raise _capacity_error(
                error=error,
                capacities_path=capacities_path,
                profile=profile,
                model_alias=work_model_alias,
            ) from error

        work_model = _ModelSpec(
            alias=work_model_alias,
            api_model=work_api_model,
            api_config=dict(api_config),
            extra_body=(
                dict(work_extra_body)
                if work_extra_body
                else None
            ),
            token_counting=dict(
                work_capacity[
                    "token_counting"
                ]
            ),
            context_window_tokens=work_capacity["context_window_tokens"],
            max_output_tokens=work_capacity["max_output_tokens"],
        )

        resolved_trigger_tokens = (
            _resolve_trigger_tokens(
                trigger_setting,
                context_window_tokens=(
                    work_model.context_window_tokens
                ),
                max_output_tokens=(
                    work_model.max_output_tokens
                ),
            )
        )

        compressor_alias = str(
            config.get("model") or work_model_alias
        ).strip()
        compressor_api_model = str(
            api_profiles.api_model_name(compressor_alias, api_config)
        )
        compressor_extra_body = _resolve_extra_body(
            api_profiles,
            compressor_alias,
            api_config,
            compressor_api_model,
        )

        try:
            compressor_capacity = load_effective_capacity(
                capacities_path,
                profile=profile,
                model_alias=compressor_alias,
                api_model=compressor_api_model,
                base_url=base_url,
                extra_body=compressor_extra_body,
            )
        except (
            FileNotFoundError,
            KeyError,
            ValueError,
        ) as error:
            raise _capacity_error(
                error=error,
                capacities_path=capacities_path,
                profile=profile,
                model_alias=compressor_alias,
            ) from error

        compressor_model = _ModelSpec(
            alias=compressor_alias,
            api_model=compressor_api_model,
            api_config=dict(api_config),
            extra_body=compressor_extra_body,
            token_counting=dict(
                compressor_capacity[
                    "token_counting"
                ]
            ),
            context_window_tokens=compressor_capacity[
                "context_window_tokens"
            ],
            max_output_tokens=compressor_capacity["max_output_tokens"],
        )

    configure_request_guard(
        capacities_path=capacities_path,
        api_profiles=api_profiles,
        api_config=api_config,
        work_model_alias=work_model_alias,
    )

    runtime = _ContextRuntime(
        enabled=enabled,
        trace_enabled=trace_enabled,
        trace_root=Path(output_root),
        system_prompt_template=(
            system_prompt_template
        ),
        user_prompt_template=(
            user_prompt_template
        ),
        trigger_tokens=resolved_trigger_tokens,
        work_model=work_model,
        compressor_model=compressor_model,
    )

    with _RUNTIME_LOCK:
        _RUNTIME = runtime

    public_config: dict[str, Any] = {
        "enabled": enabled,
        "trace_enabled": trace_enabled,
        "requested_trigger_tokens": (
            trigger_setting
        ),
        "resolved_trigger_tokens": (
            resolved_trigger_tokens
        ),
        "safe_input_tokens": (
            runtime.safe_input_tokens()
        ),
        "capacities_path": capacities_path.name,
        "prompt_path": prompt_path.name,
        "user_prompt_path": user_prompt_path.name,
    }

    if work_model is not None:
        public_config["work_model"] = {
            "alias": work_model.alias,
            "api_model": work_model.api_model,
            "token_counting_backend": (
                work_model.token_counting.get(
                    "backend"
                )
            ),
            "context_window_tokens": work_model.context_window_tokens,
            "max_output_tokens": work_model.max_output_tokens,
        }

    if compressor_model is not None:
        public_config["compressor_model"] = {
            "alias": compressor_model.alias,
            "api_model": compressor_model.api_model,
            "token_counting_backend": (
                compressor_model.token_counting.get(
                    "backend"
                )
            ),
            "context_window_tokens": compressor_model.context_window_tokens,
            "max_output_tokens": compressor_model.max_output_tokens,
        }

    runtime.record("context_runtime.configured", public_config)
    return public_config


def _get_runtime() -> _ContextRuntime | None:
    with _RUNTIME_LOCK:
        return _RUNTIME


def _messages_with_appended_text(
    fixed_messages: list[dict[str, Any]],
    text: str,
) -> list[dict[str, Any]]:
    messages = deepcopy(fixed_messages)

    if not messages:
        return [{"role": "user", "content": text}]

    last = messages[-1]
    last["content"] = str(last.get("content") or "") + text
    return messages


def maybe_compress_text(
    *,
    stage: str,
    text: str,
    fixed_messages: list[dict[str, Any]],
) -> str:
    runtime = _get_runtime()

    if (
        runtime is None
        or not runtime.enabled
        or runtime.trigger_tokens is None
        or runtime.work_counter is None
    ):
        return text

    full_messages = _messages_with_appended_text(
        fixed_messages,
        text,
    )
    before_tokens = (
        runtime.work_counter.count_messages(
            full_messages
        )
    )

    if before_tokens < runtime.trigger_tokens:
        return text

    source_tokens = (
        runtime.work_counter.count_text(
            text
        )
    )
    safe_input_tokens = (
        runtime.safe_input_tokens()
    )

    accepted_input_limit = (
        before_tokens - 1
    )

    if safe_input_tokens is not None:
        accepted_input_limit = min(
            accepted_input_limit,
            safe_input_tokens,
        )

    fixed_tokens = (
        runtime.work_counter.count_messages(
            fixed_messages
        )
    )
    summary_cap = min(
        source_tokens - 1,
        accepted_input_limit
        - fixed_tokens
        - 1,
    )

    runtime.record(
        "compression.detected",
        {
            "stage": stage,
            "before_tokens": before_tokens,
            "trigger_tokens": (
                runtime.trigger_tokens
            ),
            "safe_input_tokens": (
                safe_input_tokens
            ),
            "accepted_input_limit": (
                accepted_input_limit
            ),
            "source": text,
        },
    )

    if summary_cap <= 0:
        runtime.record(
            "compression.fallback",
            {
                "stage": stage,
                "reason": (
                    "The fixed prompt leaves no room "
                    "for a shorter compressed result "
                    "within the accepted input limit."
                ),
                "source": text,
            },
        )
        return text

    def summary_fits(summary: str) -> bool:
        result_messages = (
            _messages_with_appended_text(
                fixed_messages,
                summary,
            )
        )
        after_tokens = (
            runtime.work_counter
            .count_messages(
                result_messages
            )
        )

        return (
            after_tokens
            <= accepted_input_limit
        )

    try:
        summary = runtime.compress_source_chunked(
            stage=stage,
            source=text,
            summary_cap=summary_cap,
            summary_validator=summary_fits,
            source_units=_split_event_units(text),
        )
    except TokenCounterError:
        raise

    except Exception as error:
        runtime.record(
            "compression.fallback",
            {
                "stage": stage,
                "reason": (
                    "Compression failed after retries."
                ),
                "source": text,
                "error": error,
            },
        )
        print(
            f"[C0] compression failed after retries "
            f"at {stage}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return text

    result_messages = (
        _messages_with_appended_text(
            fixed_messages,
            summary,
        )
    )
    after_tokens = (
        runtime.work_counter.count_messages(
            result_messages
        )
    )

    runtime.record(
        "working_context.replaced",
        {
            "stage": stage,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "safe_input_tokens": (
                safe_input_tokens
            ),
            "accepted_input_limit": (
                accepted_input_limit
            ),
            "source": text,
            "summary": summary,
        },
    )

    print(
        f"[C0] compressed text context at {stage}: "
        f"{before_tokens} -> {after_tokens} tokens",
        flush=True,
    )

    return summary


def _serialize_message(
    message: dict[str, Any],
) -> str:
    role = str(
        message.get("role")
        or "unknown"
    )

    content = message.get(
        "content"
    )

    if content is None:
        content = json.dumps(
            _json_safe(message),
            ensure_ascii=False,
            sort_keys=True,
        )

    return f"[{role}]\n{content}"


def _serialize_message_units(
    messages: list[
        dict[str, Any]
    ],
) -> list[str]:
    return [
        serialized
        if index == 0
        else "\n\n" + serialized
        for index, serialized
        in enumerate(
            _serialize_message(
                message
            )
            for message in messages
        )
    ]


def _serialize_messages(
    messages: list[
        dict[str, Any]
    ],
) -> str:
    return "".join(
        _serialize_message_units(
            messages
        )
    )


def _tail_start(
    messages: list[dict[str, Any]],
    pinned_prefix_count: int,
) -> int:
    for index in range(
        len(messages) - 1,
        pinned_prefix_count - 1,
        -1,
    ):
        if messages[index].get("role") == "assistant":
            return index

    return max(pinned_prefix_count, len(messages) - 2)


def maybe_compress_messages(
    *,
    stage: str,
    messages: list[dict[str, Any]],
    pinned_prefix_count: int,
) -> list[dict[str, Any]]:
    runtime = _get_runtime()

    if (
        runtime is None
        or not runtime.enabled
        or runtime.trigger_tokens is None
        or runtime.work_counter is None
    ):
        return messages

    if not runtime.should_compress(
        messages
    ):
        return messages

    before_tokens = (
        runtime.work_counter.count_messages(
            messages
        )
    )

    pinned_prefix_count = max(
        0,
        min(int(pinned_prefix_count), len(messages)),
    )
    tail_start = _tail_start(messages, pinned_prefix_count)

    prefix = deepcopy(messages[:pinned_prefix_count])
    older = deepcopy(messages[pinned_prefix_count:tail_start])
    tail = deepcopy(messages[tail_start:])

    if not older:
        runtime.record(
            "compression.fallback",
            {
                "stage": stage,
                "reason": (
                    "There is no older message history "
                    "that can be compressed."
                ),
                "before_messages": messages,
            },
        )
        return messages

    marker = "[Compressed Earlier Context]\n"

    base_messages = (
        prefix
        + [
            {
                "role": "user",
                "content": marker,
            }
        ]
        + tail
    )

    source_units = (
        _serialize_message_units(
            older
        )
    )
    source = "".join(
        source_units
    )
    source_tokens = (
        runtime.work_counter.count_text(
            source
        )
    )
    safe_input_tokens = (
        runtime.safe_input_tokens()
    )

    accepted_input_limit = (
        before_tokens - 1
    )

    if safe_input_tokens is not None:
        accepted_input_limit = min(
            accepted_input_limit,
            safe_input_tokens,
        )

    base_tokens = (
        runtime.work_counter.count_messages(
            base_messages
        )
    )
    summary_cap = min(
        source_tokens - 1,
        accepted_input_limit
        - base_tokens
        - 1,
    )

    if summary_cap <= 0:
        runtime.record(
            "compression.fallback",
            {
                "stage": stage,
                "reason": (
                    "The fixed prompt and latest turn "
                    "leave no room for a shorter "
                    "compressed result within the "
                    "accepted input limit."
                ),
                "before_messages": messages,
            },
        )
        return messages

    def build_result(
        summary: str,
    ) -> list[dict[str, Any]]:
        return (
            prefix
            + [
                {
                    "role": "user",
                    "content": (
                        marker + summary
                    ),
                }
            ]
            + tail
        )

    def summary_fits(
        summary: str,
    ) -> bool:
        after_tokens = (
            runtime.work_counter
            .count_messages(
                build_result(
                    summary
                )
            )
        )

        return (
            after_tokens
            <= accepted_input_limit
        )

    try:
        summary = runtime.compress_source_chunked(
            stage=stage,
            source=source,
            summary_cap=summary_cap,
            summary_validator=summary_fits,
            source_units=source_units,
        )
    except TokenCounterError:
        raise

    except Exception as error:
        runtime.record(
            "compression.fallback",
            {
                "stage": stage,
                "reason": (
                    "Compression failed after retries."
                ),
                "before_messages": messages,
                "source": source,
                "error": error,
            },
        )
        print(
            f"[C0] compression failed after retries "
            f"at {stage}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return messages

    result = build_result(
        summary
    )
    after_tokens = (
        runtime.work_counter.count_messages(
            result
        )
    )

    runtime.record(
        "working_context.replaced",
        {
            "stage": stage,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "safe_input_tokens": (
                safe_input_tokens
            ),
            "accepted_input_limit": (
                accepted_input_limit
            ),
            "before_messages": messages,
            "compressed_older_messages": older,
            "summary": summary,
            "after_messages": result,
        },
    )

    print(
        f"[C0] compressed message history at {stage}: "
        f"{before_tokens} -> {after_tokens} tokens",
        flush=True,
    )

    return result


def observe_prompt_usage(
    *,
    messages: list[dict[str, Any]],
    prompt_tokens: int,
    model_kind: str = "work",
) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return

    if model_kind not in {
        "work",
        "compressor",
    }:
        raise ValueError(
            "model_kind must be 'work' "
            "or 'compressor'."
        )

    counter = (
        runtime.compressor_counter
        if model_kind == "compressor"
        else runtime.work_counter
    )

    if counter is not None:
        validation = counter.observe(
            messages,
            prompt_tokens,
        )
        runtime.record(
            "token_counter.validated",
            {
                "model_kind": model_kind,
                "validation": validation,
            },
        )


def record_model_event(
    *,
    stage: str,
    model: str,
    messages: list[dict[str, Any]],
    status: str,
    response: Any = None,
    raw_response: Any = None,
    usage: dict[str, Any] | None = None,
    error: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return

    runtime.record(
        "model_call",
        {
            "stage": stage,
            "model": model,
            "status": status,
            "messages": messages,
            "response": response,
            "raw_response": raw_response,
            "usage": usage or {},
            "error": error,
            "metadata": metadata or {},
        },
    )


def record_tool_event(
    *,
    stage: str,
    tool_name: str,
    tool_input: Any,
    rewritten_input: Any,
    tool_output: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return

    runtime.record(
        "tool_call",
        {
            "stage": stage,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "rewritten_input": rewritten_input,
            "tool_output": tool_output,
            "metadata": metadata or {},
        },
    )


def record_state_event(
    stage: str,
    payload: Any,
) -> None:
    runtime = _get_runtime()
    if runtime is None:
        return

    runtime.record(
        "state",
        {
            "stage": stage,
            "value": payload,
        },
    )


@contextmanager
def trace_scope(
    *,
    sample_id: str,
    trace_path: str | Path,
) -> Iterator[None]:
    runtime = _get_runtime()

    if runtime is None:
        yield
        return

    sample_token = _SAMPLE_ID.set(str(sample_id))
    path_token = _TRACE_PATH.set(Path(trace_path))
    runtime.record("sample.start", {"sample_id": sample_id})

    try:
        yield

    except BaseException as error:
        runtime.record(
            "sample.error",
            {
                "sample_id": sample_id,
                "error": error,
            },
        )
        raise

    else:
        runtime.record("sample.end", {"sample_id": sample_id})

    finally:
        _TRACE_PATH.reset(path_token)
        _SAMPLE_ID.reset(sample_token)
