# -*- coding: utf-8 -*-
# Shared final-answer equivalence judge for all evaluated methods.
import hashlib
import importlib.util
import json
import logging
import os
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_scoring_profile():
    path = Path(__file__).with_name("scoring_profile.py")
    spec = importlib.util.spec_from_file_location("unified_scoring_profile", path)
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    return profile


DEFAULT_JUDGE_API_PROFILE = "deepseek_official"
DEFAULT_JUDGE_MODEL = "dso_DSv41Flash"
API_PROFILE = DEFAULT_JUDGE_API_PROFILE
MODEL = DEFAULT_JUDGE_MODEL
API_CONFIG, API_MODEL, EXTRA_BODY = load_scoring_profile().configuration(API_PROFILE, MODEL)
API_KEY = API_CONFIG["api_key"]
BASE_URL = API_CONFIG["base_url"].rstrip("/")
TEMPERATURE = 0.0
TIMEOUT = 960.0
MAX_TOKENS = 8192
VERDICT_PARSER_VERSION = 1
# The protocol identity isolates caches across judge versions.
JUDGE_PROTOCOL_VERSION = "three_step_decimal_transport_v2_20261022"
JUDGE_IDENTITY = {
    "protocol": JUDGE_PROTOCOL_VERSION, "api_profile": API_PROFILE,
    "model": MODEL, "api_model": API_MODEL, "temperature": TEMPERATURE,
    "extra_body": EXTRA_BODY, "verdict_parser_version": VERDICT_PARSER_VERSION,
    "max_tokens": MAX_TOKENS,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
JUDGE_SIGNATURE = hashlib.sha256(
    json.dumps(JUDGE_IDENTITY, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()[:16]
CACHE_NAMESPACE = f"judge_{JUDGE_SIGNATURE}"
DEFAULT_CACHE_PATH = None  # The caller must choose a run-specific cache directory.

_CACHE = {}
_CACHE_LOADED = set()
_LOCK = threading.Lock()


class JudgeOutputError(ValueError):
    """Unresolved judge output; scoring fallback is handled by the caller."""


def _parse_verdict(output):
    # Only read an explicit standalone final verdict, never a word in reasoning.
    def standalone(line):
        line = line.strip().rstrip('.。').strip()
        for wrapper in ('**', '__', '`', '"', "'"):
            if line.startswith(wrapper) and line.endswith(wrapper):
                line = line[len(wrapper):-len(wrapper)].strip()
        line = line.rstrip('.。').strip().lower()
        return {'true': True, 'false': False}.get(line)

    if not isinstance(output, str):
        raise JudgeOutputError('Judge reply is not text; evaluation remains unresolved.')
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    verdict = standalone(lines[-1]) if lines else None
    if verdict is None:
        raise JudgeOutputError('No explicit final True/False; evaluation remains unresolved.')
    for line in lines[:-1]:
        earlier = standalone(line)
        if earlier is not None and earlier != verdict:
            raise JudgeOutputError('Conflicting True/False verdicts; evaluation remains unresolved.')
    return verdict


def _text(x):
    return "" if x is None else str(x).strip()


def _key(answer_1, answer_2):
    payload = json.dumps([JUDGE_SIGNATURE, _text(answer_1), _text(answer_2)], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def versioned_cache_path(path):
    """Isolate even explicitly supplied cache files; never consume legacy rows."""
    if path is None:
        return None
    path = Path(path)
    if path.stem.endswith("_" + CACHE_NAMESPACE):
        return path
    return path.with_name(f"{path.stem}_{CACHE_NAMESPACE}{path.suffix}")


def _load_cache(path):
    if path is None:
        return
    path = versioned_cache_path(path)
    with _LOCK:
        if path in _CACHE_LOADED:
            return
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("judge_signature") != JUDGE_SIGNATURE:
                        continue
                    _CACHE[row["key"]] = row
        _CACHE_LOADED.add(path)


def _save_cache(path, row):
    if path is None:
        return
    path = versioned_cache_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        if _CACHE.get(row["key"]) == row:
            return
        _CACHE[row["key"]] = row
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _float_equal(a, b):
    # Same strict 1e-3 tolerance as S-series, without IEEE float rounding huge
    # unequal integers to the same value, or treating the list "1,2" as 12.
    number = re.compile(r'^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
    try:
        if not all(number.fullmatch(_text(v)) for v in (a, b)):
            return False
        x, y = (Decimal(_text(v).replace(',', '')) for v in (a, b))
        return x.is_finite() and y.is_finite() and abs(x - y) < Decimal('0.001')
    except (InvalidOperation, ValueError, OverflowError):
        return False


def _parse_exact_scalar(value):
    """Parse only a complete rational scalar, never an arbitrary math expression.

    Presentation wrappers are removed only around the entire answer. Lists,
    variables, conditions, units and incomplete LaTeX stay with the LLM path.
    Fraction avoids both binary floating point and large-number rounding.
    """
    text = _text(value)
    if not text or len(text) > 4096:
        return None
    for _ in range(8):
        previous = text
        for left, right in ((r'\[', r'\]'), (r'\(', r'\)'), ('$$', '$$'), ('$', '$')):
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left):-len(right)].strip()
                break
        else:
            wrapper = re.fullmatch(r'\\(?:boxed|fbox)\s*\{([\s\S]*)\}', text)
            if wrapper:
                body = wrapper.group(1)
                depth = 0
                for char in body:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                    if depth < 0:
                        return None
                if depth:
                    return None
                text = body.strip()
        if text == previous:
            break
    number = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

    def rational(token):
        if not re.fullmatch(number, token):
            return None
        try:
            decimal = Decimal(token.replace(',', ''))
            if not decimal.is_finite() or abs(decimal.as_tuple().exponent) > 1000:
                return None
            return Fraction(decimal)
        except (InvalidOperation, ValueError, OverflowError):
            return None

    scalar = rational(text)
    if scalar is not None:
        return scalar
    frac = re.fullmatch(r'([+-]?)\\(?:frac|dfrac|tfrac)\s*\{\s*(' + number + r')\s*\}\s*\{\s*(' + number + r')\s*\}', text)
    if frac:
        sign, numerator, denominator = frac.groups()
    else:
        frac = re.fullmatch(r'(' + number + r')\s*/\s*(' + number + r')', text)
        if not frac:
            return None
        sign, (numerator, denominator) = '', frac.groups()
    a, b = rational(numerator), rational(denominator)
    if a is None or b is None or b == 0:
        return None
    return (-1 if sign == '-' else 1) * a / b


def _call_llm(prompt, retries=3):
    if not API_KEY:
        raise RuntimeError("Set DEEPSEEK_API_KEY to use the official equivalence judge.")
    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    payload.update(EXTRA_BODY)
    last = None
    for i in range(retries + 1):
        data = json.dumps(payload).encode("utf-8")
        output = None
        finish_reason = None
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                obj = json.loads(r.read().decode("utf-8"))
                finish_reason = obj["choices"][0].get("finish_reason")
                output = obj["choices"][0]["message"]["content"]
                if finish_reason not in ('stop', 'end_turn'):
                    raise JudgeOutputError('Incomplete judge response; not a reliable verdict.')
                _parse_verdict(output)
                return output.strip()
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, TypeError, json.JSONDecodeError, JudgeOutputError) as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise  # Never retry provider rejection/authentication or rewrite blocked data.
            if isinstance(exc, JudgeOutputError):
                logging.warning(
                    "Judge format failure attempt=%d/%d finish_reason=%r reply=%r: %s",
                    i + 1, retries + 1, finish_reason,
                    output[:1000] if isinstance(output, str) else output, exc,
                )
                # A format retry uses a fresh verdict prompt without the invalid earlier reply.
                payload["messages"] = [
                    {"role": "system", "content": (
                        "Your previous reply did not satisfy the required output format. "
                        "Judge only whether the two final answers are mathematically equivalent. "
                        "Do not solve the original problem. Follow the equivalence rules in the user message. "
                        "Reply with exactly one word: True or False. "
                        "Do not include explanations, labels, punctuation, or Markdown."
                    )},
                    {"role": "user", "content": prompt},
                ]
            if i < retries:
                time.sleep(min(20, 2 * (i + 1)))
    if isinstance(last, JudgeOutputError):
        raise last
    raise RuntimeError(str(last))


def judge_answer_equivalence(answer_1, answer_2, cache_path=DEFAULT_CACHE_PATH):
    a = _text(answer_1)
    b = _text(answer_2)
    # Empty answers are incorrect. The numeric answer zero is nonempty.
    if not a or not b:
        return {"equivalent": False, "method": "empty_answer", "answer_1": a, "answer_2": b}
    if a == b:
        return {"equivalent": True, "method": "exact", "answer_1": a, "answer_2": b}
    if _float_equal(a, b):
        return {"equivalent": True, "method": "float", "answer_1": a, "answer_2": b}

    scalar_a, scalar_b = _parse_exact_scalar(a), _parse_exact_scalar(b)
    if scalar_a is not None and scalar_b is not None:
        # The original absolute tolerance remains strict. Numerical inequality
        # is conclusive, so an LLM cannot accept close-looking large numbers.
        difference = abs(scalar_a - scalar_b)
        return {"equivalent": difference < Fraction(1, 1000), "method": "numeric_exact",
                "answer_1": a, "answer_2": b, "absolute_difference": str(difference),
                "tolerance": "strictly_less_than_1/1000"}

    cache_key = _key(a, b)
    _load_cache(cache_path)
    if cache_key in _CACHE:
        cached = dict(_CACHE[cache_key])
        try:
            cached['equivalent'] = _parse_verdict(cached.get('llm_output'))
        except JudgeOutputError:
            # Old malformed/missing replies must be judged again, not trusted as False.
            with _LOCK:
                _CACHE.pop(cache_key, None)
        else:
            cached['verdict_parser_version'] = VERDICT_PARSER_VERSION
            _save_cache(cache_path, cached)
            return cached

    prompt = (
        "Determine whether the following two mathematical final answers are equivalent.\n\n"
        "Judge only the equivalence of the two final answers. Do not solve any original problem. "
        "Do not infer missing information beyond the two answers.\n\n"
        "Equivalence rules:\n"
        "1. Ignore purely cosmetic formatting differences, such as spaces, commas in large numbers, "
        "\\boxed{} wrapping, or simple labels like x = when the value is otherwise the same.\n"
        "2. Treat mathematically equal numeric forms as equivalent, such as 1/2, \\frac{1}{2}, and 0.5.\n"
        "3. For multiple answers or solution sets, the two answers are equivalent only if they contain the same complete set of values. "
        "Order does not matter for sets or lists of solutions.\n"
        "4. If one answer gives only part of a solution set and the other gives the complete set, output False.\n"
        "5. For ordered pairs, vectors, matrices, or sequences where order is part of the answer, order matters.\n"
        "6. If the mathematical meaning is unclear or cannot be reliably determined, output False.\n\n"
        "Examples:\n"
        "Answer 1: 1/2\n"
        "Answer 2: 0.5\n"
        "Output: True\n\n"
        "Answer 1: {1, 2}\n"
        "Answer 2: {1}\n"
        "Output: False\n\n"
        "Now judge the following two answers:\n\n"
        f"Answer 1: {a}\n\n"
        f"Answer 2: {b}\n\n"
        'Output only one word: "True" or "False".'
    )

    try:
        output = _call_llm(prompt)
    except JudgeOutputError as exc:
        # Candidate policy: judge/service errors are unresolved, not evidence that
        # a generated answer is wrong. Missing extracted answers still count wrong.
        logging.warning(
            "Judge output unresolved; retained pending. key=%s cache=%s error=%s",
            cache_key, cache_path, exc,
        )
        return {
            "key": cache_key,
            "equivalent": None,
            "verdict_parser_version": VERDICT_PARSER_VERSION,
            "method": "judge_error_pending",
            "answer_1": a,
            "answer_2": b,
            "llm_output": None,
            "judge_error": str(exc),
        }
    row = {
        "key": cache_key,
        "judge_signature": JUDGE_SIGNATURE,
        "judge_config": JUDGE_IDENTITY,
        "equivalent": _parse_verdict(output),
        "verdict_parser_version": VERDICT_PARSER_VERSION,
        "method": "llm",
        "answer_1": a,
        "answer_2": b,
        "llm_output": output,
    }
    _save_cache(cache_path, row)
    return row


def judge_many(pairs, workers=1, cache_path=DEFAULT_CACHE_PATH):
    pairs = list(pairs)
    if workers <= 1:
        return [judge_answer_equivalence(a, b, cache_path=cache_path) for a, b in pairs]
    results = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(judge_answer_equivalence, a, b, cache_path): i
            for i, (a, b) in enumerate(pairs)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results
