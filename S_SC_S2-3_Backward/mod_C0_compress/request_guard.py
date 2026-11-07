from __future__ import annotations

import logging
import sys
import threading
from copy import deepcopy

from .capacity import (
    get_entry,
    load_effective_capacity,
    load_registry,
)
from .token_counter import build_token_counter


_CONTEXT_SAFETY_TOKENS = 128
_LOCK = threading.RLock()
_STATE = None
_RESOURCES = {}
_LOGGER = logging.getLogger(__name__)


class RequestBudgetError(ValueError):
    """A request cannot be safely sent with its current token budget."""


def configure_request_guard(
    *,
    capacities_path,
    api_profiles,
    api_config,
    work_model_alias,
):
    global _STATE

    with _LOCK:
        _STATE = (
            capacities_path,
            api_profiles,
            dict(api_config),
            work_model_alias,
        )
        _RESOURCES.clear()


def _merge_body(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_body(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def _effective_payload(client, request_kwargs):
    payload = deepcopy(request_kwargs)
    create = client.chat.completions.create
    function = getattr(create, "__func__", create)
    seen = set()
    qwen_patch_installed = False

    while callable(function) and id(function) not in seen:
        seen.add(id(function))
        if getattr(function, "_spq_qwen36_output_request_patch", False):
            qwen_patch_installed = True
        function = getattr(function, "_spq_original", None)

    if qwen_patch_installed:
        patch = sys.modules.get("spq_qwen36_output_request_patch")
        apply_options = getattr(
            patch, "apply_qwen36_output_request_options", None
        )
        if not callable(apply_options):
            raise RequestBudgetError(
                "Cannot resolve the installed output-budget adapter."
            )
        payload = apply_options(payload)

    extra_body = payload.pop("extra_body", None)
    if extra_body:
        _merge_body(payload, extra_body)

    return payload


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _get_resource(client, payload):
    with _LOCK:
        if _STATE is None:
            raise RequestBudgetError(
                "Configure the context runtime before sending LLM requests."
            )

        path, profiles, config, work_alias = _STATE
        model = str(payload.get("model") or "")
        matches = []

        for alias, api_model in config.get("model_map", {}).items():
            if str(api_model) != model:
                continue

            extra_body = profiles.extra_body_for_model(
                alias, config, api_model
            ) or {}

            if all(payload.get(key) == value for key, value in extra_body.items()):
                matches.append((alias, extra_body))

        preferred = [item for item in matches if item[0] == work_alias]
        if preferred:
            matches = preferred

        if len(matches) != 1:
            raise RequestBudgetError(
                f"Cannot uniquely match request model/mode: {model!r}."
            )

        alias, extra_body = matches[0]
        base_url = str(client.base_url).rstrip("/")
        key = (alias, model, base_url)

        if key not in _RESOURCES:
            actual_config = dict(config)
            actual_config["base_url"] = base_url
            actual_config["api_key"] = getattr(
                client, "api_key", config.get("api_key", "")
            )
            profile = str(config.get("profile") or "")

            capacity = load_effective_capacity(
                path,
                profile=profile,
                model_alias=alias,
                api_model=model,
                base_url=base_url,
                extra_body=extra_body,
            )
            counter = build_token_counter(
                model_alias=alias,
                api_model=model,
                api_config=actual_config,
                extra_body=extra_body,
                counter_config=capacity["token_counting"],
            )
            entry = get_entry(
                load_registry(path),
                profile=profile,
                model_alias=alias,
            ) or {}
            default_output = entry.get("measured", {}).get(
                "observed_default_output_limit_tokens"
            )
            _RESOURCES[key] = (capacity, counter, default_output)

        return _RESOURCES[key]


def check_chat_request_budget(client, request_kwargs):
    """Inspect only; never truncate content or alter the generation budget."""
    payload = _effective_payload(client, request_kwargs)
    capacity, counter, default_output = _get_resource(client, payload)

    output_values = [
        payload[key]
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens")
        if payload.get(key) is not None
    ]
    if len(output_values) > 1:
        raise RequestBudgetError(
            "Multiple output-budget fields are present in one request."
        )

    output_tokens = _positive_int(
        output_values[0] if output_values else default_output
    )
    if output_tokens is None:
        raise RequestBudgetError(
            "The effective output budget is unknown; refusing to guess."
        )

    context_limit = _positive_int(capacity.get("context_window_tokens"))
    if context_limit is None:
        raise RequestBudgetError("The context capacity is unknown.")

    input_tokens = counter.count_request(payload)
    service_limit = _positive_int(
        getattr(counter, "observed_context_window_tokens", None)
    )
    if service_limit is not None:
        context_limit = min(context_limit, service_limit)

    output_limit = _positive_int(capacity.get("max_output_tokens"))
    safety_tokens = max(_CONTEXT_SAFETY_TOKENS, counter.token_safety_margin)
    required_tokens = input_tokens + output_tokens + safety_tokens

    if required_tokens > context_limit or (
        output_limit is not None and output_tokens > output_limit
    ):
        message = (
            f"Request token budget exceeded: input={input_tokens}, "
            f"output={output_tokens}, safety={safety_tokens}, "
            f"context_limit={context_limit}, output_limit={output_limit}. "
            "Generation request was not sent."
        )
        _LOGGER.error(message)
        raise RequestBudgetError(message)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "safety_tokens": safety_tokens,
        "context_limit": context_limit,
    }
