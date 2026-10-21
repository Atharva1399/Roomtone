"""Model providers for the LLM step: a local model through Ollama, Hugging Face
Inference Providers, or Claude.

All take the same input (a system prompt, Anthropic-style content blocks of
text and base64 images, and a JSON schema) and return a dict that matches the
schema. Pick one with LLM_PROVIDER = "ollama" | "hf" | "anthropic" | "auto".
Auto prefers the local model when Ollama is running and has the model, then
Hugging Face (HF_TOKEN), then Claude (Anthropic credentials).
"""

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

HF_MODEL = os.environ.get("HF_MODEL", "google/gemma-4-31B-it")
HF_FALLBACK_MODEL = os.environ.get("HF_FALLBACK_MODEL", "Qwen/Qwen3.8-27B")
ANTHROPIC_MODEL = "claude-opus-5-5"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
# A 4B model on a laptop GPU (partly on CPU) takes seconds to tens of seconds per answer.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "180"))
# Claude's server-side refusal fallback (opt-in): a declined request is retried
# on a suitable fallback model inside the same call.
_ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    """No credentials, a network failure, or no usable answer; callers fall back."""


# ---------- provider selection ----------

def _hf_token() -> str | None:
    """HF_TOKEN from the environment, or the token saved by `hf auth login`."""
    from huggingface_hub import get_token
    return get_token()


def _has_hf() -> bool:
    return bool(_hf_token())


def _has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE"))


_ollama_seen: tuple[float, bool] = (-1e9, False)


def _has_ollama() -> bool:
    """Ollama is running locally and has OLLAMA_MODEL. Cached for 30 s, so the
    check costs nothing on a hot path; a down server answers in milliseconds."""
    global _ollama_seen
    checked, ok = _ollama_seen
    if time.monotonic() - checked < 30:
        return ok
    try:
        import httpx
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.5).json()
        names = {m.get("name", "") for m in tags.get("models", [])}
        want = OLLAMA_MODEL if ":" in OLLAMA_MODEL else f"{OLLAMA_MODEL}:latest"
        ok = want in names
    except Exception:
        ok = False
    _ollama_seen = (time.monotonic(), ok)
    return ok


def active_provider() -> str | None:
    choice = os.environ.get("LLM_PROVIDER", "auto").lower()
    if choice == "ollama":
        return "ollama" if _has_ollama() else None
    if choice == "hf":
        return "hf" if _has_hf() else None
    if choice == "anthropic":
        return "anthropic" if _has_anthropic() else None
    if _has_ollama():
        return "ollama"
    return "hf" if _has_hf() else "anthropic" if _has_anthropic() else None


def provider_label() -> str:
    p = active_provider()
    return {"ollama": f"local:{OLLAMA_MODEL}", "hf": f"huggingface:{HF_MODEL}",
            "anthropic": f"claude:{ANTHROPIC_MODEL}"}.get(p, "none")


# One model call at a time: a local model serves one request at a time anyway, and a
# queue on our side lets a waiting call see that it is no longer wanted and step aside.
_LLM_LOCK = threading.Lock()


def call(system: str, content: list, schema: dict, max_tokens: int = 8000,
         still_wanted: Callable[[], bool] | None = None) -> dict:
    with _LLM_LOCK:
        if still_wanted is not None and not still_wanted():
            raise LLMUnavailable("superseded by a newer request")
        return _call_provider(system, content, schema, max_tokens)


def _call_provider(system: str, content: list, schema: dict, max_tokens: int) -> dict:
    p = active_provider()
    if p is None:
        raise LLMUnavailable(f"no LLM available (start Ollama with {OLLAMA_MODEL}, "
                             "or set HF_TOKEN / ANTHROPIC_API_KEY)")
    if p == "ollama":
        raw = _call_ollama(system, content, schema, max_tokens)
    elif p == "hf":
        raw = _call_hf(system, content, schema, max_tokens)
    else:
        raw = _call_anthropic(system, content, schema, max_tokens)
    return conform(raw, schema)


# ---------- schema conformance ----------

def conform(data, schema: dict):
    """Validate a model's JSON against our (simple) schemas, fixing what is safe to
    fix: enum case/spacing, numbers given as strings. Raises ValueError otherwise.
    Open models don't always honour response_format, so we never trust it blindly."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(data, dict):
            raise ValueError(f"expected object, got {type(data).__name__}")
        out = {}
        for key in schema.get("required", []):
            if key not in data:
                raise ValueError(f"missing field '{key}'")
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                out[key] = conform(data[key], sub)
        return out
    if t == "array":
        if not isinstance(data, list):
            raise ValueError("expected array")
        return [conform(x, schema["items"]) for x in data]
    if t == "boolean":
        if isinstance(data, str) and data.lower() in ("true", "false"):
            data = data.lower() == "true"
        if not isinstance(data, bool):
            raise ValueError("expected boolean")
        return data
    if t == "integer":
        if isinstance(data, str) and data.strip().lstrip("-").isdigit():
            data = int(data)
        if not isinstance(data, int) or isinstance(data, bool):
            raise ValueError("expected integer")
        if "minimum" in schema:
            data = max(schema["minimum"], data)
        if "maximum" in schema:
            data = min(schema["maximum"], data)
        return data
    if t == "string":
        if not isinstance(data, str):
            data = str(data)
        if "enum" in schema:
            norm = {e.lower().replace("-", " "): e for e in schema["enum"]}
            key = data.strip().lower().replace("-", " ").replace("_", " ")
            if key in norm:
                return norm[key]
            if "other" in schema["enum"]:
                return "other"
            raise ValueError(f"'{data}' not one of {schema['enum']}")
        return data.strip()
    return data


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _parse_json_text(text: str):
    """Pull a JSON object out of a model reply that may wrap it in thinking tags,
    code fences or a sentence of preamble."""
    text = _THINK.sub("", text).strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


# ---------- Local model through Ollama ----------

def _call_ollama(system: str, content: list, schema: dict, max_tokens: int) -> dict:
    """Ollama's chat API with the JSON schema as `format`, so the model is held to
    the schema while it generates; images go in as base64."""
    import httpx

    text = "\n\n".join(b["text"] for b in content if b["type"] == "text")
    images = [b["source"]["data"] for b in content if b["type"] == "image"]
    user = {"role": "user", "content": text}
    if images:
        user["images"] = images
    body = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": system}, user],
        "format": schema,
        "stream": False,
        "keep_alive": "30m",    # stay loaded between requests during a demo
        "options": {"temperature": 0.3, "num_predict": min(max_tokens, 2048), "num_ctx": 4096},
    }
    errors = []
    for attempt in range(2):   # one retry: a small model occasionally emits invalid JSON
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=OLLAMA_TIMEOUT)
        except httpx.TimeoutException as e:
            raise LLMUnavailable(f"the local model took longer than {OLLAMA_TIMEOUT:.0f} s") from e
        except httpx.HTTPError as e:
            raise LLMUnavailable("could not reach Ollama; is it running?") from e
        if r.status_code == 404:
            raise LLMUnavailable(f"Ollama has no model '{OLLAMA_MODEL}' (run: ollama pull {OLLAMA_MODEL})")
        if r.status_code >= 400:
            raise LLMUnavailable(f"Ollama error {r.status_code}: {r.text[:120]}")
        out = r.json()
        if out.get("done_reason") == "length":
            errors.append("answer cut off")
            continue
        try:
            return conform(_parse_json_text(out.get("message", {}).get("content", "")), schema)
        except (ValueError, json.JSONDecodeError) as e:
            errors.append(f"unusable answer ({e})")
            log.warning("Ollama attempt %d failed: %s", attempt + 1, errors[-1])
    raise LLMUnavailable("the local model gave no usable answer: " + "; ".join(errors))


# ---------- Hugging Face Inference Providers ----------

def _to_openai_content(content: list) -> list:
    out = []
    for b in content:
        if b["type"] == "image":
            src = b["source"]
            out.append({"type": "image_url",
                        "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
        elif b["type"] == "text":
            out.append({"type": "text", "text": b["text"]})
    return out


def _call_hf(system: str, content: list, schema: dict, max_tokens: int) -> dict:
    from huggingface_hub import InferenceClient
    from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

    client = InferenceClient(provider="auto", api_key=_hf_token(), timeout=120)
    system_full = (system + "\n\nReply with only a JSON object that matches this JSON Schema, "
                   "with no other text:\n" + json.dumps(schema))
    messages = [{"role": "system", "content": system_full},
                {"role": "user", "content": _to_openai_content(content)}]
    response_format = {"type": "json_schema",
                       "json_schema": {"name": "answer", "schema": schema, "strict": True}}

    # Strict schema mode on the main model first; some hosts reject response_format,
    # so retry with the schema only in the prompt, then try the fallback model.
    attempts = [(HF_MODEL, response_format), (HF_MODEL, None), (HF_FALLBACK_MODEL, None)]
    errors = []
    for model, rf in attempts:
        try:
            out = client.chat_completion(messages=messages, model=model, max_tokens=max_tokens,
                                         temperature=0.3, response_format=rf)
            choice = out.choices[0]
            if choice.finish_reason == "length":
                raise ValueError("answer cut off (max_tokens)")
            data = _parse_json_text(choice.message.content or "")
            return conform(data, schema)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (401, 403):
                raise LLMUnavailable("Hugging Face token rejected (needs the "
                                     "'Make calls to Inference Providers' permission)") from e
            if status == 402:
                raise LLMUnavailable("Hugging Face inference credits used up for this month") from e
            errors.append(f"{model}: HTTP {status}")
        except InferenceTimeoutError:
            errors.append(f"{model}: timed out")
        except (ValueError, json.JSONDecodeError) as e:
            errors.append(f"{model}: unusable answer ({e})")
        except Exception as e:  # network errors surface as requests exceptions
            errors.append(f"{model}: {type(e).__name__}")
        log.warning("HF attempt failed: %s", errors[-1])
    raise LLMUnavailable("Hugging Face models gave no usable answer: " + "; ".join(errors))


# ---------- Claude ----------

_anthropic_client = None


def _call_anthropic(system: str, content: list, schema: dict, max_tokens: int) -> dict:
    import anthropic

    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(timeout=120.0, max_retries=2)
    try:
        response = _anthropic_client.beta.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            betas=[_ANTHROPIC_FALLBACK_BETA],
            fallbacks="default",
            system=system,
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable("Anthropic credentials were rejected") from e
    except anthropic.RateLimitError as e:
        raise LLMUnavailable("Claude is rate limited; try again shortly") from e
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable("could not reach the Anthropic API") from e
    except anthropic.APIStatusError as e:
        log.warning("Claude API error %s", e.status_code)
        raise LLMUnavailable(f"Claude API error {e.status_code}") from e

    if response.stop_reason == "refusal":
        raise LLMUnavailable("Claude declined this request")
    if response.stop_reason == "max_tokens":
        raise LLMUnavailable("Claude's answer was cut off")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise LLMUnavailable("Claude returned no text")
    return json.loads(text)
