# -*- coding: utf-8 -*-
# Evaluation rule: this shared 0-2_step_judge package is the only hard-test evaluator.
# Do not create project-local evaluation methods or metrics.
import hashlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


API_PROFILE = "deepseek_official"
MODEL = "dso_DSv41Flash"
API_MODEL = "deepseek-flash"
EXTRA_BODY = {"thinking": {"type": "disabled"}}
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 0.0
TIMEOUT = 60.0
VERDICT_PARSER_VERSION = 1
JUDGE_PROTOCOL_VERSION = "three_step_shared_v1"
JUDGE_IDENTITY = {
    "protocol": JUDGE_PROTOCOL_VERSION, "api_profile": API_PROFILE,
    "model": MODEL, "api_model": API_MODEL, "temperature": TEMPERATURE,
    "extra_body": EXTRA_BODY, "verdict_parser_version": VERDICT_PARSER_VERSION,
}
JUDGE_SIGNATURE = hashlib.sha256(
    json.dumps(JUDGE_IDENTITY, ensure_ascii=False, sort_keys=True).encode("utf-8")
).hexdigest()[:16]
CACHE_NAMESPACE = f"judge_{JUDGE_SIGNATURE}"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "judge_cache" / f"{MODEL}_{CACHE_NAMESPACE}.jsonl"

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
    try:
        return abs(float(_text(a).replace(",", "")) - float(_text(b).replace(",", ""))) < 1e-3
    except Exception:
        return False


def _call_llm(prompt, retries=3):
    payload = {
        "model": API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
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
                _parse_verdict(output)
                return output.strip()
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, JudgeOutputError) as exc:
            last = exc
            if isinstance(exc, JudgeOutputError):
                logging.warning(
                    "Judge format failure attempt=%d/%d finish_reason=%r reply=%r: %s",
                    i + 1, retries + 1, finish_reason,
                    output[:1000] if isinstance(output, str) else output, exc,
                )
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
            time.sleep(min(20, 2 * (i + 1)))
    if isinstance(last, JudgeOutputError):
        raise last
    raise RuntimeError(str(last))


def judge_answer_equivalence(answer_1, answer_2, cache_path=DEFAULT_CACHE_PATH):
    a = _text(answer_1)
    b = _text(answer_2)
    if not a or not b:
        return {"equivalent": False, "method": "empty_answer", "answer_1": a, "answer_2": b}
    if a == b:
        return {"equivalent": True, "method": "exact", "answer_1": a, "answer_2": b}
    if _float_equal(a, b):
        return {"equivalent": True, "method": "float", "answer_1": a, "answer_2": b}

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
        logging.warning(
            "Judge output unresolved; counted as incorrect by policy. key=%s cache=%s error=%s",
            cache_key, cache_path, exc,
        )
        return {
            "key": cache_key,
            "equivalent": False,
            "verdict_parser_version": VERDICT_PARSER_VERSION,
            "method": "judge_error_as_incorrect",
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
