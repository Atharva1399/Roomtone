"""Mood input: the vocabulary parser and its effect through the palettes API."""

import pytest

from app import llm
from app.pipeline.mood import parse_mood


@pytest.mark.parametrize("text, goal, hues", [
    ("calm coastal morning", "calm", ["aqua", "sand"]),
    ("moody library", "bold", ["forest", "burgundy"]),
    ("fresh sage green nursery", "bright", ["sage", "green"]),
    ("Cozy cabin evening", "cosy", ["rust", "olive"]),
    ("scandinavian minimal", "bright", []),
])
def test_parse_reads_goal_and_hues(text, goal, hues):
    r = parse_mood(text)
    assert r.goal == goal
    assert [h["word"] for h in r.hues] == hues
    assert r.source == "keywords"


def test_named_colour_outranks_theme_colours():
    r = parse_mood("library but in navy")
    assert r.hues[0]["word"] == "navy"


def test_negation_and_intensifiers():
    assert parse_mood("bright but not cold").adjust["warmth"] > 0
    r = parse_mood("not too dark, earthy")
    assert r.adjust.get("lightness", 0) > 0          # "not too dark" lightens
    assert r.goal != "bold"


def test_steps_are_clamped_and_neutral_yields_to_colour():
    r = parse_mood("warm warm warm warm sunny cosy")
    assert r.adjust["warmth"] == 3
    assert parse_mood("neutral sage").neutral is False
    assert parse_mood("neutral calm").neutral is True


def test_unknown_words_are_listed_not_guessed():
    r = parse_mood("an afternoon in Kyoto")
    assert r.empty
    assert "kyoto" in r.ignored
    assert r.summary() == ""


def test_hue_angles_are_measured():
    hues = {h["word"]: h["hue"] for h in parse_mood("sage navy terracotta").hues}
    assert 120 < hues["sage"] < 150 and 250 < hues["navy"] < 275


# ---------- through the API ----------

@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr("app.main.llm_configured", lambda: False)


def test_api_mood_sets_goal_adjust_and_matches(client, room, no_llm):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"mood": "calm coastal morning", "n": 3}).json()
    assert body["goal"] == "calm" and body["goal_source"] == "mood"
    assert body["mood"]["summary"].startswith("calm")
    assert body["adjust"]["warmth"] < 0
    assert any("matches your mood" in f for p in body["palettes"] for f in p["facts"])


def test_api_explicit_goal_beats_mood_goal(client, room, no_llm):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"mood": "moody library", "goal": "bright"}).json()
    assert body["goal"] == "bright" and body["goal_source"] == "request"
    assert body["mood"]["hues"][0]["word"] == "forest"


def test_api_mood_adds_to_manual_adjust(client, room, no_llm):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"mood": "warm", "adjust": {"warmth": 2, "lightness": 0, "saturation": 0}}).json()
    assert body["adjust"]["warmth"] == 3                 # 2 + 2, clamped


def test_api_unknown_mood_uses_llm_when_configured(client, room, monkeypatch):
    monkeypatch.setattr("app.main.llm_configured", lambda: True)
    monkeypatch.setattr(llm, "_call", lambda *a, **k: {"goal": "calm", "warmth": -1, "lightness": 1,
                                                       "saturation": -1, "hue_words": ["sky"], "neutral": False})
    monkeypatch.setattr(llm, "provider_label", lambda: "fake")
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"mood": "an afternoon in Kyoto"}).json()
    assert body["mood"]["source"] == "fake"
    assert body["goal"] == "calm"
    assert body["mood"]["hues"][0]["word"] == "sky"


def test_api_unknown_mood_without_llm_changes_nothing(client, room, no_llm):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes", json={"mood": "an afternoon in Kyoto"}).json()
    assert body["goal"] == "balanced" and body["goal_source"] in ("default", "style")
    assert body["mood"]["summary"] == ""


def test_api_mood_length_limit(client, room):
    r = client.post(f"/api/rooms/{room['room_id']}/palettes", json={"mood": "x" * 201})
    assert r.status_code == 422
