"""Local, opt-in transport diagnostics without changing evaluation semantics.

``install(module, stage)`` replaces only that module's ``urllib`` binding.
``with capture(context, sink=None) as events: ...`` records safe HTTP metadata;
the optional synchronous sink receives one independent dictionary per request.
No request body, URL, header, response text, exception message, or reasoning
text is retained.  Captures must be entered in each worker thread (ContextVar
contexts are not automatically inherited by ThreadPoolExecutor workers).

The wrapped evaluators read non-streaming JSON with ``response.read()``.  Sized
reads are supported by collecting a bounded inspection copy until EOF/close;
the original bytes and all transport arguments are always passed unchanged.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from http.client import IncompleteRead, RemoteDisconnected
import json
import re
import ssl
from threading import RLock
from time import perf_counter
from types import SimpleNamespace
import urllib.error


_ACTIVE = ContextVar("unified_evaluation_http_diagnostics", default=None)
_INSTALL_LOCK = RLock()
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,95}\Z")
_CONTEXT_KEYS = frozenset({
    "job_key", "case_id", "run_id", "run_signature", "input_sha256",
    "response_sha256", "source_sha256",
})
_MODELS = frozenset({
    "deepseek-chat", "deepseek-reasoner", "deepseek-flash", "deepseek-pro",
    "deepseek-v4-flash", "deepseek-v4-pro",
})
_FINISH_REASONS = frozenset({
    "stop", "end_turn", "length", "max_tokens", "tool_calls", "function_call",
    "content_filter", "insufficient_system_resource", "sensitive",
})
_USAGE_KEYS = frozenset({
    "prompt_tokens", "completion_tokens", "total_tokens", "input_tokens",
    "output_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
    "reasoning_tokens", "cached_tokens",
})
_USAGE_DETAILS = {
    "prompt_tokens_details": frozenset({"cached_tokens", "audio_tokens"}),
    "completion_tokens_details": frozenset({
        "reasoning_tokens", "audio_tokens", "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    }),
    "input_tokens_details": frozenset({"cached_tokens"}),
    "output_tokens_details": frozenset({"reasoning_tokens"}),
}
_MAX_INSPECTION_BYTES = 4 * 1024 * 1024


class AuditStorageError(RuntimeError):
    """The requested audit sink failed; the surrounding batch must stop."""


def _type_name(value):
    name = type(value).__name__
    return name if _TYPE_NAME.fullmatch(name) else "UnrecognizedExceptionType"


def _stage(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError("diagnostic stage must be a short safe identifier")
    return value


def _http_status(value):
    return value if type(value) is int and 100 <= value <= 599 else None


def describe_exception(exc, stage):
    """Return a safe exception description, never ``str(exc)`` or its args.

    This is also suitable for exceptions outside the HTTP transport.  Previous
    transport failures swallowed/replaced by legacy evaluators remain visible
    in the capture's event list; this helper does not invent their causes.
    """
    result = {"stage": _stage(stage), "error_type": _type_name(exc),
              "kind": "exception", "retryable": False}
    status = _http_status(getattr(exc, "code", None))
    if status is not None:
        result["http_status"] = status
        result.update(kind="http_error", retryable=status in {408, 429, 500, 502, 503, 504})
    elif isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError,
                          IncompleteRead, RemoteDisconnected, ssl.SSLError)):
        result.update(kind="network_error", retryable=True)
    elif isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
        result.update(kind="response_decode_error", retryable=True)
    elif _type_name(exc) == "JudgeOutputError":
        result.update(kind="invalid_judge_output", retryable=True)
    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException):
        result["cause_type"] = _type_name(reason)
    return result


class _Capture:
    def __init__(self, context, sink):
        self.context = {
            key: value for key, value in dict(context or {}).items()
            if key in _CONTEXT_KEYS and isinstance(value, str)
            and _IDENTIFIER.fullmatch(value)
        }
        self.sink = sink
        self.events = []
        self.attempts = {}
        self.lock = RLock()
        self.sink_error_type = None

    def check_storage(self):
        if self.sink_error_type is not None:
            error = AuditStorageError("diagnostic sink failed; evaluation stopped")
            error.sink_error_type = self.sink_error_type
            raise error from None

    def begin(self, stage):
        with self.lock:
            # Do not send another request after a swallowed storage failure.
            self.check_storage()
            attempt = self.attempts.get(stage, 0) + 1
            self.attempts[stage] = attempt
            return {
                "context": dict(self.context), "stage": stage,
                "attempt": attempt, "http_status": None,
                "finish_reason": None, "returned_model": None,
                "usage": {}, "content_length": None, "content_empty": None,
                "error_type": None, "kind": "http_response", "retryable": False,
            }

    def publish(self, event, *, preserve_exception=False):
        with self.lock:
            self.events.append(deepcopy(event))
            if self.sink is not None and self.sink_error_type is None:
                try:
                    self.sink(deepcopy(event))
                except Exception as exc:
                    # Legacy evaluators catch Exception.  Remember the failure
                    # and rethrow at capture exit even if they swallow it.
                    self.sink_error_type = _type_name(exc)
            if not preserve_exception:
                self.check_storage()


@contextmanager
def capture(context=None, sink=None):
    """Yield a per-context event list, optionally persisting each event.

    Allowed context keys are job_key, case_id, run_id, run_signature,
    input_sha256, response_sha256 and source_sha256.  Values must be short
    ASCII identifiers; unknown keys/unsafe values are discarded.  Nested
    captures are independent.  A sink failure ALWAYS escapes this context as
    AuditStorageError, including when the wrapped evaluator caught it.
    """
    if sink is not None and not callable(sink):
        raise TypeError("diagnostic sink must be callable")
    state = _Capture(context, sink)
    token = _ACTIVE.set(state)
    try:
        yield state.events
    finally:
        _ACTIVE.reset(token)
        state.check_storage()


def _usage(raw):
    if not isinstance(raw, dict):
        return {}
    result = {
        key: value for key, value in raw.items()
        if key in _USAGE_KEYS and type(value) is int and value >= 0
    }
    for key, allowed in _USAGE_DETAILS.items():
        detail = raw.get(key)
        if isinstance(detail, dict):
            clean = {
                field: value for field, value in detail.items()
                if field in allowed and type(value) is int and value >= 0
            }
            if clean:
                result[key] = clean
    return result


def _inspect_body(body, event):
    try:
        raw = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        event["response_decode_error_type"] = _type_name(exc)
        event.update(kind="response_decode_error", retryable=True)
        return
    if not isinstance(raw, dict):
        event["response_shape"] = "non_object"
        event.update(kind="invalid_response_shape", retryable=True)
        return
    event["response_shape"] = "object"
    model = raw.get("model")
    if isinstance(model, str) and model in _MODELS:
        event["returned_model"] = model
    elif model is not None:
        event["returned_model_unrecognized"] = True
    event["usage"] = _usage(raw.get("usage"))
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        event["choice_present"] = False
        event.update(kind="invalid_response_shape", retryable=True)
        return
    event["choice_present"] = True
    choice = choices[0]
    finish = choice.get("finish_reason")
    if isinstance(finish, str) and finish in _FINISH_REASONS:
        event["finish_reason"] = finish
    elif finish is not None:
        event["finish_reason_unrecognized"] = True
    if finish in ("length", "max_tokens", "insufficient_system_resource"):
        event.update(kind="incomplete_response", retryable=True)
    elif finish in ("content_filter", "sensitive"):
        event.update(kind="provider_rejection", retryable=False)
    elif finish not in ("stop", "end_turn"):
        event.update(kind="invalid_finish_reason", retryable=finish is None)
    message = choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        event["content_kind"] = "missing"
        if event["kind"] == "http_response":
            event.update(kind="invalid_response_shape", retryable=True)
        return
    content = message["content"]
    if isinstance(content, str):
        event["content_kind"] = "text"
        event["content_length"] = len(content)
        event["content_empty"] = not bool(content.strip())
    elif content is None:
        event["content_kind"] = "null"
        event["content_empty"] = True
    else:
        event["content_kind"] = "non_text"
        if event["kind"] == "http_response":
            event.update(kind="invalid_response_shape", retryable=True)
    if event["content_empty"] is True and event["kind"] == "http_response":
        event.update(kind="empty_response", retryable=True)


class _ResponseProxy:
    def __init__(self, response, state, event, start):
        self._response = response
        self._context_manager = response
        self._state = state
        self._event = event
        self._start = start
        self._published = False
        self._parts = []
        self._bytes = 0
        self._read_started = False
        self._event["http_status"] = _http_status(getattr(response, "status", None))

    def __getattr__(self, name):
        return getattr(self._response, name)

    def _finish(self, exc=None):
        if self._published:
            return
        self._published = True
        if self._read_started:
            self._event["response_bytes"] = self._bytes
            if self._bytes <= _MAX_INSPECTION_BYTES:
                _inspect_body(b"".join(self._parts), self._event)
            else:
                self._event["inspection_truncated"] = True
        else:
            self._event["response_not_read"] = True
        self._parts.clear()
        if exc is not None:
            self._event.update(describe_exception(exc, self._event["stage"]))
        elif self._event["http_status"] is not None and self._event["http_status"] >= 400:
            status = self._event["http_status"]
            self._event.update(kind="http_error", retryable=status in {408, 429, 500, 502, 503, 504})
        self._event["elapsed_seconds"] = round(perf_counter() - self._start, 6)
        self._state.publish(self._event, preserve_exception=exc is not None)

    def read(self, *args, **kwargs):
        try:
            result = self._response.read(*args, **kwargs)
        except BaseException as exc:
            self._finish(exc)
            raise
        self._read_started = True
        if isinstance(result, bytes):
            self._bytes += len(result)
            if self._bytes <= _MAX_INSPECTION_BYTES:
                self._parts.append(result)
            else:
                self._parts.clear()
        # HTTPResponse.read() and read(None/-1) consume the entire body.
        amount = args[0] if args else kwargs.get("amt", kwargs.get("size"))
        if amount is None or (isinstance(amount, int) and amount < 0) or (result == b"" and amount != 0):
            self._finish()
        return result

    def __enter__(self):
        try:
            entered = self._context_manager.__enter__()
        except BaseException as exc:
            self._finish(exc)
            raise
        self._response = entered
        self._event["http_status"] = _http_status(getattr(entered, "status", None))
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            result = self._context_manager.__exit__(exc_type, exc, traceback)
        except BaseException as exit_exc:
            self._finish(exit_exc)
            raise
        self._finish(exc)
        return result

    def close(self):
        try:
            result = self._response.close()
        except BaseException as exc:
            self._finish(exc)
            raise
        self._finish()
        return result


class _RequestProxy:
    def __init__(self, original, stage):
        self._original = original
        self._stage = stage
        self.Request = original.Request

    def __getattr__(self, name):
        return getattr(self._original, name)

    def urlopen(self, *args, **kwargs):
        state = _ACTIVE.get()
        if state is None:
            return self._original.urlopen(*args, **kwargs)
        event = state.begin(self._stage)
        start = perf_counter()
        try:
            response = self._original.urlopen(*args, **kwargs)
        except BaseException as exc:
            event.update(describe_exception(exc, self._stage))
            event["elapsed_seconds"] = round(perf_counter() - start, 6)
            state.publish(event, preserve_exception=True)
            raise
        return _ResponseProxy(response, state, event, start)


def install(module, stage):
    """Idempotently instrument this module's urllib binding, not global urllib.

    No request is sent by installation.  Request construction, URL, payload,
    headers, timeouts, transport exceptions, and evaluator retries stay owned
    by the original code.  A conflicting second stage is an explicit error.
    Returns the installed local namespace.
    """
    stage = _stage(stage)
    with _INSTALL_LOCK:
        current = module.urllib
        request = current.request
        if isinstance(request, _RequestProxy):
            if request._stage != stage:
                raise ValueError("module already instrumented for a different stage")
            return current
        local = SimpleNamespace(
            error=current.error, request=_RequestProxy(request, stage),
        )
        module.urllib = local
        return local


__all__ = ["AuditStorageError", "capture", "describe_exception", "install"]
