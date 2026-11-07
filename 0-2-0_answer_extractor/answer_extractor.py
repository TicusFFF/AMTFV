"""Shared, reporting-only A8 extraction. Never a solver or voting component.

The prompt and _a8_extract_answer body are copied unchanged from S7's A8
upstream/progco/MATH/extract_answer.py (2026-09-28, SHA256
965f212ad5fab68c70f1bd96ac5050b42595b045e2c7e49f1989c1f385646abf).
Only the transport, audit/cache wrapper and failure status are centralized.
Do not import this into training, intervention, or candidate selection.
"""
import hashlib
import os
import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import sqlite3
import threading
import time
from types import FunctionType
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'shared_a8_quote_v1'
# FAIRNESS: configure here only. No per-method/model/environment overrides.
API_PROFILE = 'deepseek_official'
MODEL = 'dso_DSv41Flash'
TEMPERATURE = 0
MAX_TOKENS = 8192
TIMEOUT_SECONDS = 960
NETWORK_RETRIES = 3  # Additional attempts; never modify the payload on retry.
_CALL_LOCK = threading.Lock()  # Serial extraction per process; never calls Luna.

extract_prompt = """Your task is to extract the final answer explicitly stated in the given solution response.

Important notes:
1. Do not solve the problem, continue the reasoning, infer an unstated answer, or guess an answer.
2. Select only the final-answer expression actually stated as the solution's conclusion. Do not select a tentative guess, an intermediate value, or an example from the problem.
3. Copy the selected expression from the solution response. Preserve its mathematical value and original LaTeX; do not rewrite it into a different expression.
4. If the response does not explicitly state a final answer, return null for answer_quote.
5. Output exactly one JSON object with the single key answer_quote. Do not output explanations or additional fields. Escape backslashes correctly in JSON.

Examples:
Solution: The final answer is 42.
Output: {{"answer_quote": "42"}}

Solution: Let us next consider the remaining case.
Output: {{"answer_quote": null}}

Here is the problem:
{query}

Here is the solution response:
{response}
"""


def _a8_extract_answer(query, response, model, max_retries=2, retry_sleep_seconds=3):
    source_response = str(response or "")
    if not source_response.strip():
        return ""

    prompt = extract_prompt.format(query=query, response=source_response)
    messages = [{"role": "user", "content": prompt}]
    normalized_source = " ".join(source_response.split())
    last_error = ""

    for attempt in range(1, max_retries + 1):
        result = call_llm(model=model, messages=messages, temperature=0)
        try:
            text = str(result or "").strip()
            if text.startswith("```") and text.endswith("```"):
                lines = text.splitlines()
                if len(lines) >= 3:
                    text = "\n".join(lines[1:-1]).strip()
            payload = json.loads(text)
            if not isinstance(payload, dict) or set(payload) != {"answer_quote"}:
                raise ValueError("expected only the answer_quote field")
            quote = payload["answer_quote"]
            if quote is None:
                return ""
            if not isinstance(quote, str) or not quote.strip():
                raise ValueError("answer_quote must be a nonempty string or null")
            quote = quote.strip()
            if " ".join(quote.split()) not in normalized_source:
                raise ValueError("answer_quote is not present in the solution response")
            return quote
        except (ValueError, TypeError) as exc:
            last_error = str(exc)

        print(
            f"[extract_answer] invalid extraction attempt {attempt}/{max_retries}, "
            f"model={model}, reason={last_error}",
            flush=True,
        )
        if attempt < max_retries:
            time.sleep(retry_sleep_seconds * attempt)

    # Keep the failure local: no trusted answer was extracted.
    # The complete-solution controller requests correction/completion.
    print(
        f"[extract_answer] no trusted final answer after {max_retries} attempts, "
        f"model={model}, reason={last_error}",
        flush=True,
    )
    return ""


@lru_cache(maxsize=1)
def _api():
    config = {"api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
              "base_url": "https://api.deepseek.com"}
    return config, "deepseek-flash", {"thinking": {"type": "disabled"}}


def identity():
    _, api_model, extra = _api()
    return {'protocol': PROTOCOL, 'api_profile': API_PROFILE, 'model': MODEL,
            'api_model': api_model, 'temperature': TEMPERATURE, 'max_tokens': MAX_TOKENS,
            'extra_body': extra, 'format_attempts': 2, 'network_retries': NETWORK_RETRIES,
            'prompt_sha256': hashlib.sha256(extract_prompt.encode('utf-8')).hexdigest()}


def call_llm(model, messages, temperature):
    """Transport only; the A8 prompt/validation/retry loop above is unchanged."""
    if model != MODEL or temperature != TEMPERATURE:
        raise ValueError('Public A8 configuration must not be overridden.')
    config, api_model, extra = _api()
    payload = {'model': api_model, 'messages': messages, 'temperature': TEMPERATURE,
               **extra, 'max_tokens': MAX_TOKENS, 'stream': False}
    data = json.dumps(payload).encode('utf-8')
    for attempt in range(NETWORK_RETRIES + 1):
        try:
            # Queue time is outside the HTTP timeout. Release before backoff.
            with _CALL_LOCK:
                request = urllib.request.Request(
                    config['base_url'].rstrip('/') + '/chat/completions', data=data,
                    headers={'Content-Type': 'application/json',
                             'Authorization': 'Bearer ' + config['api_key']}, method='POST')
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    result = json.loads(response.read().decode('utf-8'))
            choice = result['choices'][0]
            if choice.get('finish_reason') not in ('stop', 'end_turn'):
                raise ValueError('Incomplete extraction output; needs review.')
            return choice['message']['content']
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (408, 429, 500, 502, 503, 504):
                raise  # No repeat attempts for invalid_prompt/authentication failures.
            if attempt == NETWORK_RETRIES:
                raise
            time.sleep(2 ** (attempt + 1))


@contextmanager
def _cache_connection(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    try:
        with connection:
            connection.execute('CREATE TABLE IF NOT EXISTS a8_cache (key TEXT PRIMARY KEY, record TEXT NOT NULL)')
            yield connection
    finally:
        connection.close()


def extract_answer(query, response, *, cache_path=None):
    """Return ok/no_answer/pending + audit. No gold or algorithm identity input.

    Cache identity includes question, full response, prompt and model settings.
    A transport/JSON/quote-validation failure is never cached as a wrong answer.
    """
    config = identity()
    payload = json.dumps([config, query, response], ensure_ascii=False, sort_keys=True)
    key = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    if cache_path is not None:
        with _cache_connection(cache_path) as connection:
            cached = connection.execute('SELECT record FROM a8_cache WHERE key=?', (key,)).fetchone()
        if cached:
            return {**json.loads(cached[0]), 'cache_hit': True}
    record = {'key': key, 'extraction_config': config, 'status': 'pending',
              'answer': None, 'cache_hit': False, 'attempts': [],
              'response_sha256': hashlib.sha256(str(response).encode('utf-8')).hexdigest()}
    if not isinstance(response, str) or not isinstance(query, str) or not query.strip():
        return {**record, 'reason': 'missing_question_or_full_response'}

    def tracked_call(**kwargs):
        reply = call_llm(**kwargs)
        record['attempts'].append({'reply': reply})
        return reply

    # Bind only transport per invocation; the copied A8 body is not modified.
    # Fresh globals avoid cross-thread changes to the original A8 or this module.
    a8 = FunctionType(_a8_extract_answer.__code__,
                      {**globals(), 'call_llm': tracked_call}, argdefs=_a8_extract_answer.__defaults__)
    try:
        answer = a8(query, response, MODEL)
        if answer in ('**', '__', '`', '```', '$', '$$', r'\[', r'\]', r'\(', r'\)'):
            record['reason'] = 'formatting_only_quote_needs_review'
        elif answer:
            record.update(status='ok', answer=answer)
        elif not response.strip():
            record.update(status='no_answer', answer='', reason='empty_source_response')
        else:
            reply = str(record['attempts'][-1]['reply']).strip() if record['attempts'] else ''
            if reply.startswith('```') and reply.endswith('```'):
                reply = '\n'.join(reply.splitlines()[1:-1]).strip()
            try:
                no_answer = json.loads(reply) == {'answer_quote': None}
            except (ValueError, TypeError):
                no_answer = False
            if no_answer:
                record.update(status='no_answer', answer='', reason='no_explicit_final_answer')
            else:
                record['reason'] = 'invalid_a8_output_or_quote_after_two_attempts'
    except Exception as exc:
        # Retain original generated response upstream; do not expose credentials
        # or retry blocked prompts here. This is not a valid False judgement.
        record['reason'] = 'extraction_request_failed:' + type(exc).__name__
    if cache_path is not None and record['status'] != 'pending':
        with _cache_connection(cache_path) as connection:
            connection.execute('INSERT OR IGNORE INTO a8_cache (key, record) VALUES (?, ?)',
                               (key, json.dumps(record, ensure_ascii=False)))
            # Concurrent identical queries use the first recorded result.
            record = json.loads(connection.execute('SELECT record FROM a8_cache WHERE key=?', (key,)).fetchone()[0])
    return record
