"""
Stream call to OpenAI API and determine whether to stop through string matching
Note: The stop parameter in OpenAI API does not work with Claude model
https://github.com/MacPaw/OpenAI/issues/198 
https://community.openai.com/t/interrupting-completion-stream-in-python/30628/7
"""
import time
import os
import threading
from openai import OpenAI
from typing import Optional, List

from utils.sample_budget import consume_llm


import api_profiles as _api_profiles
_api_config = _api_profiles.get_api_config()
_CALL_STATE = threading.local()


def _set_last_call_stats(stats):
    _CALL_STATE.last_call_stats = dict(stats)


def get_last_call_stats():
    return dict(getattr(_CALL_STATE, "last_call_stats", {}))


def _is_timeout_error(exc):
    text = f"{type(exc).__name__}: {str(exc)}".lower()
    return "timeout" in text or "timed out" in text


def _llm_timeout_seconds():
    try:
        return max(1.0, float(os.environ.get("SPQ_LLM_TIMEOUT_SECONDS", "120")))
    except ValueError:
        return 120.0


def _llm_stop_max_retries():
    try:
        value = os.environ.get("SPQ_LLM_STOP_MAX_RETRIES") or os.environ.get("SPQ_LLM_MAX_RETRIES", "30")
        return max(1, int(value))
    except ValueError:
        return 30


def _apply_stop_words(text, stop_words):
    if not stop_words:
        return text, False, None

    for stop_word in stop_words:
        if stop_word and stop_word in text:
            return text.split(stop_word)[0] + stop_word, True, stop_word

    return text, False, None

client = OpenAI(
    api_key=_api_config["api_key"],
    base_url=_api_config["base_url"],
    timeout=_llm_timeout_seconds(),
)

def stream_openai(model,messages,temperature,stop_words,max_tokens):
    api_model = _api_profiles.api_model_name(model, _api_config)
    request_kwargs = dict(
        model=api_model,
        messages=messages,
        stream=True,
        temperature=temperature,
    )
    if max_tokens is not None and max_tokens > 0:
        request_kwargs["max_tokens"] = max_tokens
    extra_body = _api_profiles.extra_body_for_model(model, _api_config, api_model=api_model)
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    consume_llm()
    response = client.chat.completions.create(**request_kwargs)
    is_stop=False
    gen_stop_word=None

    content_text = ""
    try:
        for index, chunk in enumerate(response):
            if index == 0:
                _api_profiles.record_response_metadata(chunk, stage="work_stream", model=api_model)
            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)

            if content is not None:
                content_text += content
                content_text, is_stop, gen_stop_word = _apply_stop_words(content_text, stop_words)
                if is_stop:
                    break
    finally:
        response.close()

    if content_text.strip():
        return content_text,is_stop,gen_stop_word

    raise RuntimeError("empty content")

def call_llm_with_stop_words(
            model: str, 
            messages: List[dict],
            temperature: float = None, 
            stop_strs: Optional[List[str]] = None,
            max_tokens: int = None):
    if _api_profiles.is_openai_compatible_model(model):
        retries = 0
        max_retries = _llm_stop_max_retries()
        stats = {"error_count": 0, "timeout_count": 0, "last_error": ""}
        _set_last_call_stats(stats)
        while retries < max_retries:
            try:
                response,is_stop,gen_stop_word = stream_openai(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stop_words=stop_strs,
                    max_tokens=max_tokens,
                )
                _set_last_call_stats(stats)
                return response,is_stop,gen_stop_word
            except KeyboardInterrupt:
                print("Operation canceled by user.")
                return ''
            except ValueError:
                raise
            except Exception as e:
                stats["error_count"] += 1
                stats["last_error"] = f"{type(e).__name__} (status {getattr(e, 'status_code', 'unknown')})"
                if _is_timeout_error(e):
                    stats["timeout_count"] += 1
                print(f"Model request failed ({stats['last_error']}); retrying.")
                time.sleep(retries*10) 
                retries += 1
        _set_last_call_stats(stats)
        return '',False,None
    else:
        raise

if __name__ == "__main__":
    MESSAGES = [
        {"role": "user", "content": "Write an essay about summer"},
    ]
    STOP_WORD = ["time","hot"]  # Set your stop words
    response,is_stop,gen_stop_word = call_llm_with_stop_words(model="qdd_GPT4o",messages=MESSAGES,temperature=0,stop_strs=STOP_WORD)
    print(response)
    print(is_stop)
    print(gen_stop_word)
