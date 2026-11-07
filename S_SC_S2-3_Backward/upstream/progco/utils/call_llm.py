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


def _llm_max_retries():
    try:
        return max(1, int(os.environ.get("SPQ_LLM_MAX_RETRIES", "20")))
    except ValueError:
        return 20


def _message_content(message):
    content = getattr(message, "content", None) or ""
    if not isinstance(content, str):
        content = str(content)
    return content

def huggingface_llm_inference(
                        model: str, 
                        messages: List[dict],
                        temperature: float = None, 
                        stop_strs: Optional[List[str]] = None,
                        max_tokens: int = None):
    raise NotImplementedError

def openai_llm_inference(
                        model: str, 
                        messages: List[dict],
                        temperature: float = None, 
                        stop_strs: Optional[List[str]] = None,
                        max_tokens: int = None):
    client = OpenAI(
        api_key=_api_config["api_key"],
        base_url=_api_config["base_url"],
        timeout=_llm_timeout_seconds(),
    )
    retries = 0
    max_retries = _llm_max_retries()
    stats = {"error_count": 0, "timeout_count": 0, "last_error": ""}
    _set_last_call_stats(stats)
    while retries < max_retries:
        try:
            api_model = _api_profiles.api_model_name(model, _api_config)
            request_kwargs = dict(
                model=api_model,
                messages=messages,
                stop=stop_strs,
                temperature=temperature,
            )
            if max_tokens is not None and max_tokens > 0:
                request_kwargs["max_tokens"] = max_tokens
            extra_body = _api_profiles.extra_body_for_model(model, _api_config, api_model=api_model)
            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body

            consume_llm()
            response = client.chat.completions.create(**request_kwargs)
            _api_profiles.record_response_metadata(response, stage="work", model=api_model)

            message = response.choices[0].message
            output = _message_content(message)
            stats.update({
                "content_chars": len(output or ""),
                "finish_reason": getattr(response.choices[0], "finish_reason", ""),
            })
            if not output.strip():
                raise RuntimeError("empty content")

            _set_last_call_stats(stats)
            return output
        except KeyboardInterrupt:
            print("Operation canceled by user.")
            break
        except ValueError:
            raise
        except Exception as e:
            stats["error_count"] += 1
            stats["last_error"] = f"{type(e).__name__} (status {getattr(e, 'status_code', 'unknown')})"
            if _is_timeout_error(e):
                stats["timeout_count"] += 1
            print(f"Model request failed ({stats['last_error']}); retrying.")
            time.sleep(retries*10) 
            # gpt-3.5-16k does not output certain data points in MATH dataset at temperature 0
            if model=='gpt-3.5-16K' and retries>3:
                temperature=0.5
            retries += 1
    _set_last_call_stats(stats)
    return ''


def call_llm(
            model: str, 
            messages: List[dict],
            temperature: float = None, 
            stop_strs: Optional[List[str]] = None,
            max_tokens: int = None):
    if _api_profiles.is_openai_compatible_model(model):
        response=openai_llm_inference(model,messages,temperature,stop_strs,max_tokens)
    else:
        response=huggingface_llm_inference(model,messages,temperature,stop_strs,max_tokens)
    return response

if __name__=="__main__":
    messages=[{'role': 'user', 'content': "write a essay about time"}]
    stop_strs=["time"]
    response=call_llm(model="gpt-3.5-16K",messages=messages,temperature=0,stop_strs=stop_strs)
    print(response)
    
