"""LLM provider tests: Hugging Face with a fake InferenceClient and Ollama with a
fake HTTP server (no network)."""

import json
import os
from types import SimpleNamespace

import pytest

from app import llm, llm_providers
from app.llm_providers import LLMUnavailable, conform

# Captured at import, before conftest's autouse fixture hides Ollama.
REAL_HAS_OLLAMA = llm_providers._has_ollama

STYLE_ANSWER = {
    "room_type": "Living Room", "style": "Scandinavian", "secondary_style": "minimal",
    "style_confidence": "HIGH", "mood": " airy ", "key_materials": ["oak", "linen"],
    "fixed_elements": [{"item": "oak floor", "colour": "honey"}],
    "natural_light": "abundant", "suggested_goal": "calm",
    "design_notes": "Soft neutrals.", "avoid": [],
}


def _reply(text, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=text))])


class FakeHTTPError(Exception):
    def __init__(self, status):
        self.response = SimpleNamespace(status_code=status)


@pytest.fixture
def hf_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("LLM_PROVIDER", "hf")
    # Make the provider's `except HfHubHTTPError` catch our fake error.
    import huggingface_hub.errors as hub_errors
    monkeypatch.setattr(hub_errors, "HfHubHTTPError", FakeHTTPError)


def fake_client(monkeypatch, replies):
    """Install a fake InferenceClient that returns/raises `replies` in order."""
    calls = []

    class Client:
        def __init__(self, **kw):
            pass

        def chat_completion(self, **kw):
            calls.append(kw)
            r = replies[len(calls) - 1]
            if isinstance(r, Exception):
                raise r
            return r

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "InferenceClient", Client)
    return calls


def test_provider_selection(monkeypatch):
    for v in ("HF_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "LLM_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(llm_providers, "_hf_token", lambda: os.environ.get("HF_TOKEN"))
    assert llm_providers.active_provider() is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm_providers.active_provider() == "anthropic"
    monkeypatch.setenv("HF_TOKEN", "y")
    assert llm_providers.active_provider() == "hf"          # auto prefers HF
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert llm_providers.active_provider() == "anthropic"


def test_conform_normalises_and_rejects():
    out = conform(STYLE_ANSWER, llm.STYLE_SCHEMA)
    assert out["room_type"] == "living room" and out["style"] == "scandinavian"
    assert out["style_confidence"] == "high" and out["mood"] == "airy"
    assert conform({"style": "Art Deco"}, {"type": "object", "properties": {
        "style": {"type": "string", "enum": ["modern", "other"]}}})["style"] == "other"
    with pytest.raises(ValueError):
        conform({"room_type": "living room"}, llm.STYLE_SCHEMA)        # missing fields
    with pytest.raises(ValueError):
        conform({"index": "two"}, {"type": "object", "properties": {"index": {"type": "integer"}},
                                   "required": ["index"]})


def test_hf_parses_messy_reply_and_sends_image(monkeypatch, hf_env):
    messy = "<think>looking at the room...</think>\nHere you go:\n```json\n" + json.dumps(STYLE_ANSWER) + "\n```"
    calls = fake_client(monkeypatch, [_reply(messy)])
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
               {"type": "text", "text": "Analyse this room."}]
    out = llm_providers.call("sys", content, llm.STYLE_SCHEMA)
    assert out["style"] == "scandinavian"

    sent = calls[0]
    assert sent["model"] == llm_providers.HF_MODEL
    assert sent["response_format"]["type"] == "json_schema"
    user = sent["messages"][1]["content"]
    assert user[0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}
    assert "JSON Schema" in sent["messages"][0]["content"]


def test_hf_retries_without_strict_mode_then_fallback_model(monkeypatch, hf_env):
    calls = fake_client(monkeypatch, [
        FakeHTTPError(422),                       # host rejects response_format
        _reply("sorry, I can't format that"),     # no JSON at all
        _reply(json.dumps(STYLE_ANSWER)),         # fallback model succeeds
    ])
    out = llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA)
    assert out["room_type"] == "living room"
    assert [c["response_format"] is not None for c in calls] == [True, False, False]
    assert calls[2]["model"] == llm_providers.HF_FALLBACK_MODEL


@pytest.mark.parametrize("status,needle", [(401, "token rejected"), (402, "credits")])
def test_hf_auth_and_credit_errors_stop_immediately(monkeypatch, hf_env, status, needle):
    calls = fake_client(monkeypatch, [FakeHTTPError(status)])
    with pytest.raises(LLMUnavailable, match=needle):
        llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA)
    assert len(calls) == 1


def test_hf_all_attempts_fail(monkeypatch, hf_env):
    fake_client(monkeypatch, [_reply("{bad json"), _reply("{}"), _reply("", finish="length")])
    with pytest.raises(LLMUnavailable, match="no usable answer"):
        llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA)


# ---------- local model through Ollama (fake server, no network) ----------

class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


def fake_ollama(monkeypatch, replies):
    """Make httpx.post return `replies` in order; record what was sent."""
    import httpx
    sent = []

    def post(url, json=None, timeout=None):
        sent.append({"url": url, "body": json, "timeout": timeout})
        r = replies[len(sent) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(llm_providers, "_has_ollama", lambda: True)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    return sent


def test_auto_prefers_local_model(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("HF_TOKEN", "y")
    monkeypatch.setattr(llm_providers, "_hf_token", lambda: "y")
    monkeypatch.setattr(llm_providers, "_has_ollama", lambda: True)
    assert llm_providers.active_provider() == "ollama"
    assert llm_providers.provider_label() == f"local:{llm_providers.OLLAMA_MODEL}"
    monkeypatch.setattr(llm_providers, "_has_ollama", lambda: False)
    assert llm_providers.active_provider() == "hf"           # falls back when Ollama is down


def test_ollama_sends_schema_and_image(monkeypatch):
    sent = fake_ollama(monkeypatch, [FakeResponse(200, {"message": {"content": json.dumps(STYLE_ANSWER)},
                                                         "done_reason": "stop"})])
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}},
               {"type": "text", "text": "Analyse this room."}]
    out = llm_providers.call("sys", content, llm.STYLE_SCHEMA)
    assert out["style"] == "scandinavian"
    body = sent[0]["body"]
    assert sent[0]["url"].endswith("/api/chat")
    assert body["format"] is llm.STYLE_SCHEMA and body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1]["images"] == ["QUJD"]
    assert body["messages"][1]["content"] == "Analyse this room."


def test_ollama_retries_once_on_bad_json(monkeypatch):
    sent = fake_ollama(monkeypatch, [
        FakeResponse(200, {"message": {"content": "{not json"}, "done_reason": "stop"}),
        FakeResponse(200, {"message": {"content": json.dumps(STYLE_ANSWER)}, "done_reason": "stop"}),
    ])
    assert llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA)["style"] == "scandinavian"
    assert len(sent) == 2


def test_ollama_missing_model_and_timeout(monkeypatch):
    import httpx
    fake_ollama(monkeypatch, [FakeResponse(404, {"error": "model not found"})])
    with pytest.raises(LLMUnavailable, match="ollama pull"):
        llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA)
    fake_ollama(monkeypatch, [httpx.ReadTimeout("slow")])
    with pytest.raises(LLMUnavailable, match="longer than"):
        llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA)


def test_ollama_detection_reads_tags(monkeypatch):
    import httpx
    tags = {"models": [{"name": "llama3.2:latest"}]}
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: FakeResponse(200, tags))
    monkeypatch.setattr(llm_providers, "_ollama_seen", (-1e9, False))
    assert REAL_HAS_OLLAMA() is False                       # wrong model installed
    tags["models"].append({"name": llm_providers.OLLAMA_MODEL})
    monkeypatch.setattr(llm_providers, "_ollama_seen", (-1e9, False))
    assert REAL_HAS_OLLAMA() is True

    def down(url, timeout=None):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", down)
    monkeypatch.setattr(llm_providers, "_ollama_seen", (-1e9, False))
    assert REAL_HAS_OLLAMA() is False                       # server not running


def test_queued_call_that_is_no_longer_wanted_is_skipped(monkeypatch):
    sent = fake_ollama(monkeypatch, [FakeResponse(200, {"message": {"content": json.dumps(STYLE_ANSWER)},
                                                         "done_reason": "stop"})])
    with pytest.raises(LLMUnavailable, match="superseded"):
        llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA, still_wanted=lambda: False)
    assert sent == []                                       # the model was never asked
    assert llm_providers.call("sys", [{"type": "text", "text": "x"}], llm.STYLE_SCHEMA,
                              still_wanted=lambda: True)["style"] == "scandinavian"
