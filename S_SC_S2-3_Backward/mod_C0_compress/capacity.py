from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _positive_int(
    value: Any,
) -> int | None:
    if isinstance(value, bool):
        return None

    try:
        value = int(value)
    except (TypeError, ValueError):
        return None

    return value if value > 0 else None


def _normalize_base_url(
    value: str,
) -> str:
    return (
        str(value or "")
        .split("?", 1)[0]
        .rstrip("/")
    )


def public_identity(
    *,
    profile: str,
    model_alias: str,
    api_model: str,
    base_url: str,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "profile": str(profile),
        "model_alias": str(model_alias),
        "api_model": str(api_model),
        "base_url": _normalize_base_url(
            base_url
        ),
        "extra_body": dict(
            extra_body or {}
        ),
    }


def load_registry(
    path: str | Path,
) -> dict[str, Any]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            "Cannot find public "
            f"model-capacity registry: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if (
        data.get("schema_version")
        != SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported model-capacity schema "
            f"in {path}: "
            f"{data.get('schema_version')!r}"
        )

    if not isinstance(
        data.get("profiles"),
        dict,
    ):
        raise ValueError(
            "Invalid model-capacity registry: "
            f"{path}"
        )

    return data


def get_entry(
    registry: dict[str, Any],
    *,
    profile: str,
    model_alias: str,
) -> dict[str, Any] | None:
    profile_data = (
        registry
        .get("profiles", {})
        .get(str(profile), {})
    )

    if not isinstance(profile_data, dict):
        return None

    entry = profile_data.get(
        str(model_alias)
    )

    return (
        entry
        if isinstance(entry, dict)
        else None
    )


def _validate_identity(
    entry: dict[str, Any],
    current_identity: dict[str, Any],
) -> None:
    saved = entry.get("identity")

    if not isinstance(saved, dict):
        return

    for key in (
        "profile",
        "model_alias",
        "api_model",
        "base_url",
        "extra_body",
    ):
        if key not in saved:
            continue

        if saved[key] != current_identity[key]:
            raise ValueError(
                "The saved model capacity "
                "does not match the current "
                "API configuration: "
                f"field={key!r}, "
                f"saved={saved[key]!r}, "
                f"current="
                f"{current_identity[key]!r}."
            )


def _select_limit(
    *,
    declared: dict[str, Any],
    measured: dict[str, Any],
    value_key: str,
    exact_key: str,
) -> tuple[int | None, str]:
    measured_value = _positive_int(
        measured.get(value_key)
    )
    declared_value = _positive_int(
        declared.get(value_key)
    )

    if (
        measured_value
        and measured.get(exact_key) is True
    ):
        return (
            measured_value,
            "measured_exact",
        )

    if declared_value:
        return declared_value, "declared"

    if measured_value:
        return (
            measured_value,
            "measured_safe_bound",
        )

    return None, ""


def resolve_token_counting_paths(
    registry_path: str | Path,
    token_counting: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(
        token_counting or {}
    )

    if (
        resolved.get(
            "paths_relative_to_registry"
        )
        is not True
    ):
        return resolved

    base_dir = (
        Path(registry_path)
        .resolve()
        .parent
    )

    for key in (
        "tokenizer_name_or_path",
        "encoding_module_path",
    ):
        value = str(
            resolved.get(key) or ""
        ).strip()

        if not value:
            continue

        path = Path(value)

        if not path.is_absolute():
            resolved[key] = str(
                (
                    base_dir
                    / path
                ).resolve()
            )

    return resolved


def load_effective_capacity(
    path: str | Path,
    *,
    profile: str,
    model_alias: str,
    api_model: str,
    base_url: str,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    registry = load_registry(path)
    entry = get_entry(
        registry,
        profile=profile,
        model_alias=model_alias,
    )

    if entry is None:
        entry = {}

    identity = public_identity(
        profile=profile,
        model_alias=model_alias,
        api_model=api_model,
        base_url=base_url,
        extra_body=extra_body,
    )
    _validate_identity(
        entry,
        identity,
    )

    declared = (
        entry.get("declared")
        if isinstance(
            entry.get("declared"),
            dict,
        )
        else {}
    )
    measured = (
        entry.get("measured")
        if isinstance(
            entry.get("measured"),
            dict,
        )
        else {}
    )
    token_counting = (
        entry.get("token_counting")
        if isinstance(
            entry.get("token_counting"),
            dict,
        )
        else {}
    )

    context_window, context_source = (
        _select_limit(
            declared=declared,
            measured=measured,
            value_key=(
                "context_window_tokens"
            ),
            exact_key="context_is_exact",
        )
    )
    max_output, output_source = (
        _select_limit(
            declared=declared,
            measured=measured,
            value_key="max_output_tokens",
            exact_key="output_is_exact",
        )
    )

    return {
        "profile": profile,
        "model_alias": model_alias,
        "api_model": api_model,
        "context_window_tokens": (
            context_window
        ),
        "max_output_tokens": max_output,
        "context_source": context_source,
        "output_source": output_source,
        "context_status": measured.get(
            "context_status",
            "",
        ),
        "output_status": measured.get(
            "output_status",
            "",
        ),
        "token_counting": (
            resolve_token_counting_paths(
                path,
                token_counting,
            )
        ),
    }
