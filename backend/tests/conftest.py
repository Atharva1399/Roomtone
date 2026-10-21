from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "living_kolkata.png"


@pytest.fixture(autouse=True)
def no_local_llm_by_default(monkeypatch):
    """Tests never reach a real local model: Ollama counts as absent unless a test
    says otherwise (test_llm_providers turns it on with a fake server)."""
    monkeypatch.setattr("app.llm_providers._has_ollama", lambda: False)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:      # runs the startup warm-up
        yield c


def upload(client, path: Path = SAMPLE) -> dict:
    with open(path, "rb") as f:
        r = client.post("/api/rooms", files={"file": (path.name, f, "image/png")})
    assert r.status_code == 201, r.text
    return r.json()


@pytest.fixture(scope="session")
def room(client):
    return upload(client)
