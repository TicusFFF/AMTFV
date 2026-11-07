import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar


DEFAULT_MAX_LLM_CALLS = 3200
DEFAULT_MAX_PYTHON_TOOL_CALLS = 1600
DEFAULT_MAX_WALL_SECONDS = 21600

_CURRENT_SAMPLE_ID = ContextVar(
    "spq_sample_budget_sample_id",
    default=None,
)
_CURRENT_SAMPLE_STARTED_AT = ContextVar(
    "spq_sample_budget_started_at",
    default=None,
)
_REGISTRY = {}
_REGISTRY_LOCK = threading.Lock()


def _positive_env_int(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _new_budget_state():
    return {
        "llm_calls": 0,
        "python_tool_calls": 0,
        "max_llm_calls": _positive_env_int(
            "SPQ_SAMPLE_MAX_LLM_CALLS",
            DEFAULT_MAX_LLM_CALLS,
        ),
        "max_python_tool_calls": _positive_env_int(
            "SPQ_SAMPLE_MAX_PYTHON_TOOL_CALLS",
            DEFAULT_MAX_PYTHON_TOOL_CALLS,
        ),
        "max_wall_seconds": _positive_env_int(
            "SPQ_SAMPLE_MAX_WALL_SECONDS",
            DEFAULT_MAX_WALL_SECONDS,
        ),
    }


def _elapsed_wall_seconds():
    started_at = _CURRENT_SAMPLE_STARTED_AT.get()
    if started_at is None:
        return 0.0
    return max(0.0, time.monotonic() - started_at)


def _stop_reason(operation):
    return {
        "llm": "sample_llm_call_limit_reached",
        "python_tool": "sample_python_tool_call_limit_reached",
        "wall_time": "sample_wall_time_limit_reached",
    }.get(operation, "sample_budget_exceeded")


def _state_snapshot(state):
    snapshot = dict(state)
    snapshot["elapsed_wall_seconds"] = round(
        _elapsed_wall_seconds(),
        3,
    )
    return snapshot


class SampleBudgetExceeded(BaseException):
    """Non-retriable signal that stops only the active sample."""

    def __init__(
        self,
        sample_id,
        operation,
        limit,
        attempted_count,
        state,
    ):
        self.sample_id = str(sample_id)
        self.operation = str(operation)
        self.limit = int(limit)
        self.attempted_count = int(attempted_count)
        self.state = dict(state)
        self.stop_reason = _stop_reason(self.operation)

        super().__init__(
            f"Sample {self.sample_id} reached the "
            f"{self.operation} limit ({self.limit})."
        )

    def to_dict(self):
        return {
            "sample_id": self.sample_id,
            "operation": self.operation,
            "limit": self.limit,
            "attempted_count": self.attempted_count,
            "llm_calls": int(
                self.state.get("llm_calls", 0)
            ),
            "python_tool_calls": int(
                self.state.get("python_tool_calls", 0)
            ),
            "max_llm_calls": int(
                self.state.get(
                    "max_llm_calls",
                    DEFAULT_MAX_LLM_CALLS,
                )
            ),
            "max_python_tool_calls": int(
                self.state.get(
                    "max_python_tool_calls",
                    DEFAULT_MAX_PYTHON_TOOL_CALLS,
                )
            ),
            "elapsed_wall_seconds": float(
                self.state.get(
                    "elapsed_wall_seconds",
                    0.0,
                )
            ),
            "max_wall_seconds": int(
                self.state.get(
                    "max_wall_seconds",
                    DEFAULT_MAX_WALL_SECONDS,
                )
            ),
            "stop_reason": self.stop_reason,
        }


def _ensure_sample(sample_id):
    with _REGISTRY_LOCK:
        return _REGISTRY.setdefault(
            sample_id,
            _new_budget_state(),
        )


def check_sample_budget():
    sample_id = _CURRENT_SAMPLE_ID.get()
    if sample_id is None:
        return

    with _REGISTRY_LOCK:
        state = _REGISTRY.setdefault(
            sample_id,
            _new_budget_state(),
        )
        elapsed = _elapsed_wall_seconds()
        limit = int(state["max_wall_seconds"])

        if elapsed < limit:
            return

        snapshot = _state_snapshot(state)

    raise SampleBudgetExceeded(
        sample_id=sample_id,
        operation="wall_time",
        limit=limit,
        attempted_count=int(elapsed),
        state=snapshot,
    )


@contextmanager
def sample_budget_scope(sample_id):
    sample_key = str(sample_id)
    _ensure_sample(sample_key)

    sample_token = _CURRENT_SAMPLE_ID.set(sample_key)
    started_token = _CURRENT_SAMPLE_STARTED_AT.set(
        time.monotonic()
    )

    try:
        check_sample_budget()
        yield sample_key
        check_sample_budget()
    finally:
        _CURRENT_SAMPLE_STARTED_AT.reset(started_token)
        _CURRENT_SAMPLE_ID.reset(sample_token)


def _consume(operation, count_key, limit_key):
    sample_id = _CURRENT_SAMPLE_ID.get()
    if sample_id is None:
        return None

    check_sample_budget()

    exceeded = None
    attempted_count = None

    with _REGISTRY_LOCK:
        state = _REGISTRY.setdefault(
            sample_id,
            _new_budget_state(),
        )
        attempted_count = int(state[count_key]) + 1
        limit = int(state[limit_key])

        if attempted_count > limit:
            exceeded = SampleBudgetExceeded(
                sample_id=sample_id,
                operation=operation,
                limit=limit,
                attempted_count=attempted_count,
                state=_state_snapshot(state),
            )
        else:
            state[count_key] = attempted_count

    if exceeded is not None:
        raise exceeded

    return attempted_count


def consume_llm():
    return _consume(
        "llm",
        "llm_calls",
        "max_llm_calls",
    )


def consume_python():
    return _consume(
        "python_tool",
        "python_tool_calls",
        "max_python_tool_calls",
    )


# C0 uses this name; retain the existing per-sample LLM budget.
consume_llm_call = consume_llm
