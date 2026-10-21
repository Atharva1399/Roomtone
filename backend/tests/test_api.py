"""End-to-end API tests on a real sample photo (needs the segmentation model;
runs on GPU if available, CPU otherwise)."""

import io
import time

import numpy as np
import pytest
from PIL import Image


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["paints"] > 1000


def test_create_room_returns_analysis(room):
    assert len(room["room_id"]) == 32
    assert room["paintable"]["wall"] > 0.05
    a = room["analysis"]
    assert {"regions", "overall_palette", "current_roles", "lighting"} <= a.keys()
    assert a["current_roles"]["dominant"]["source"] == "wall"


def test_get_room_and_images(client, room):
    rid = room["room_id"]
    assert client.get(f"/api/rooms/{rid}").json()["room_id"] == rid
    for path in ("original", "segmentation"):
        r = client.get(f"/api/rooms/{rid}/{path}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert Image.open(io.BytesIO(r.content)).size == (room["width"], room["height"])


def test_palettes(client, room):
    r = client.post(f"/api/rooms/{room['room_id']}/palettes", json={"goal": "calm", "n": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["goal"] == "calm"
    assert 1 <= len(body["palettes"]) <= 3
    for p in body["palettes"]:
        assert {"dominant", "secondary", "accent", "ceiling"} == p["roles"].keys()
        assert p["roles"]["dominant"]["brand"]               # a real paint
        assert all(c["passed"] for c in p["checks"] if c["hard"])
        assert p["facts"]


def test_palettes_brand_filter(client, room):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"brands": ["farrow-ball"], "n": 2}).json()
    for p in body["palettes"]:
        assert p["roles"]["dominant"]["brand"] == "Farrow & Ball"


def test_preview_changes_wall_and_is_fast(client, room):
    rid = room["room_id"]
    original = np.asarray(Image.open(io.BytesIO(client.get(f"/api/rooms/{rid}/original").content)))

    r = client.post(f"/api/rooms/{rid}/preview", json={"wall": "#2F3E4E", "format": "png"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["x-paint-delta-e"].startswith("wall=")
    painted = np.asarray(Image.open(io.BytesIO(r.content)))
    assert painted.shape == original.shape
    # A dark navy wall on a light room must darken the image noticeably.
    assert painted.mean() < original.mean() - 5

    # Cached layers: repeated colour changes should be quick.
    times = []
    for hx in ("#A8B5A2", "#E8DCC8", "#C46A4A"):
        t = time.perf_counter()
        assert client.post(f"/api/rooms/{rid}/preview", json={"wall": hx}).status_code == 200
        times.append(time.perf_counter() - t)
    assert min(times) < 0.5, times


@pytest.mark.parametrize("payload", [{}, {"wall": "red"}, {"wall": "#12345"}, {"ceiling": "#GGGGGG"}])
def test_preview_rejects_bad_input(client, room, payload):
    r = client.post(f"/api/rooms/{room['room_id']}/preview", json=payload)
    assert r.status_code == 422


def test_unknown_room_is_404(client):
    assert client.get("/api/rooms/" + "0" * 32).status_code == 404
    assert client.post("/api/rooms/nope/preview", json={"wall": "#FFFFFF"}).status_code == 404


def test_upload_rejects_non_image(client):
    r = client.post("/api/rooms", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_paint_match(client):
    body = client.get("/api/paints/match", params={"hex": "2F3E4E", "k": 2}).json()
    assert body["query"] == "#2F3E4E"
    assert len(body["matches"]) == 2
    assert body["matches"][0]["match_delta_e"] < 3
    assert client.get("/api/paints/match", params={"hex": "nothex"}).status_code == 422
