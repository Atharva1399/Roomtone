"""LLM step tests. The model is mocked (no network, no cost); the offline
fallback is tested with credentials removed."""

import base64
import json

import pytest

from app import llm
from tests.conftest import upload

EXPLAIN_FIELDS = {"name", "summary", "why_walls", "why_accent", "where_to_use", "watch_out", "source"}

FAKE_STYLE = {
    "room_type": "living room", "style": "minimalist", "secondary_style": "contemporary",
    "style_confidence": "high", "mood": "airy and calm",
    "key_materials": ["polished marble", "linen", "boucle"],
    "fixed_elements": [{"item": "marble floor", "colour": "warm beige"},
                       {"item": "sectional sofa", "colour": "sage green"}],
    "natural_light": "abundant", "suggested_goal": "calm",
    "design_notes": "Soft warm neutrals would keep the calm feel.", "avoid": ["saturated reds"],
}


@pytest.fixture
def no_credentials(monkeypatch):
    for var in ("HF_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("app.llm_providers._hf_token", lambda: None)   # ignore a saved HF login


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the API call; record what would have been sent."""
    calls = []

    def fake_call(system, content, schema, max_tokens=8000, still_wanted=None):
        calls.append({"system": system, "content": content, "schema": schema})
        if schema is llm.STYLE_SCHEMA:
            return dict(FAKE_STYLE)
        n = len(json.loads(content[-1]["text"])["palettes"])
        return {"palettes": [{"index": i, "name": f"Palette {i}", "summary": "s", "why_walls": "w",
                              "why_accent": "a", "where_to_use": "u", "watch_out": "c"}
                             for i in range(n)]}

    monkeypatch.setattr(llm, "_call", fake_call)
    monkeypatch.setattr(llm, "provider_label", lambda: "fake")
    monkeypatch.setattr("app.main.provider_label", lambda: "fake")
    return calls


def test_style_without_credentials_is_503(client, room, no_credentials):
    r = client.post(f"/api/rooms/{room['room_id']}/style")
    assert r.status_code == 503
    assert "HF_TOKEN" in r.json()["detail"]


def test_explanations_fall_back_to_templates(client, room, no_credentials):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"goal": "calm", "n": 2, "explain": True}).json()
    assert body["explanation_source"] == "template"
    for p in body["palettes"]:
        assert set(p["explanation"]) == EXPLAIN_FIELDS
        assert p["explanation"]["source"] == "template"
        assert p["roles"]["dominant"]["hex"] in p["explanation"]["where_to_use"]


def test_style_is_requested_with_photo_and_cached(client, fake_llm):
    rid = upload(client)["room_id"]      # fresh room: no cached style
    r = client.post(f"/api/rooms/{rid}/style")
    assert r.status_code == 200
    assert r.json()["style"] == "minimalist" and r.json()["source"] == "fake"

    sent = fake_llm[0]
    image = next(b for b in sent["content"] if b["type"] == "image")
    assert image["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(image["source"]["data"])[:3] == b"\xff\xd8\xff"
    assert "lighting" in sent["content"][-1]["text"]     # measured facts go along

    client.post(f"/api/rooms/{rid}/style")               # cached: no second call
    assert len(fake_llm) == 1


def test_style_steers_goal_and_ranking(client, fake_llm):
    rid = upload(client)["room_id"]
    client.post(f"/api/rooms/{rid}/style")
    body = client.post(f"/api/rooms/{rid}/palettes", json={"n": 4}).json()
    assert body["goal"] == "calm" and body["goal_source"] == "style"
    assert body["style"] == "minimalist"
    assert any("suits the room's minimalist style" in f for p in body["palettes"] for f in p["facts"])

    # An explicit goal wins over the style's suggestion; use_style=false ignores style.
    body = client.post(f"/api/rooms/{rid}/palettes",
                       json={"goal": "bold", "use_style": False}).json()
    assert body["goal"] == "bold" and body["style"] is None


def test_explanations_from_llm(client, fake_llm):
    rid = upload(client)["room_id"]
    body = client.post(f"/api/rooms/{rid}/palettes", json={"n": 3, "explain": True}).json()
    assert body["explanation_source"] == "fake"
    for i, p in enumerate(body["palettes"]):
        assert p["explanation"]["name"] == f"Palette {i}"
        assert p["explanation"]["source"] == "fake"

    brief = json.loads(fake_llm[-1]["content"][-1]["text"])
    assert brief["palettes"][0]["roles"]["dominant"]["paint"]     # real paint names reach the model
    assert brief["palettes"][0]["facts"]
