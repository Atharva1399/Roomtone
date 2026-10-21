"""Refine features: colour nudges, palette locks/adjustments, click-to-fix masks."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.colorutils import hex_to_rgb, lab_to_lch, rgb_to_lab
from tests.conftest import upload

DINING = Path(__file__).resolve().parents[1] / "samples" / "dining_room.jpg"
PANELLING = (150, 520)   # a click on the wood wainscot, which the segmenter calls "wall"


def lab(hx):
    return rgb_to_lab(hex_to_rgb(hx))


# ---------- colour nudges ----------

def test_adjust_color_directions(client):
    def adj(**kw):
        r = client.post("/api/colors/adjust", json={"hex": "#C8BFB5", **kw})
        assert r.status_code == 200, r.text
        return r.json()

    base = lab("#C8BFB5")
    assert lab(adj(warmth=2)["adjusted"])[2] > base[2] + 3          # more yellow
    assert lab(adj(warmth=-2)["adjusted"])[2] < base[2] - 3         # less yellow
    assert lab(adj(lightness=2)["adjusted"])[0] > base[0] + 8
    assert lab(adj(lightness=-2)["adjusted"])[0] < base[0] - 8
    assert lab_to_lch(lab(adj(saturation=-3)["adjusted"]))[1] < lab_to_lch(base)[1]

    body = adj(lightness=1, k=2)
    assert len(body["matches"]) == 2 and body["matches"][0]["brand"]


def test_adjust_color_validation(client):
    assert client.post("/api/colors/adjust", json={"hex": "beige"}).status_code == 422
    assert client.post("/api/colors/adjust", json={"hex": "#C8BFB5", "warmth": 99}).status_code == 422


# ---------- palette locks & adjustments ----------

def test_lock_wall_rebuilds_around_it(client, room):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"n": 3, "locks": {"dominant": "#2F3E4E"}}).json()
    assert body["locks"]["dominant"]["match_delta_e"] < 3
    walls = {p["roles"]["dominant"]["hex"] for p in body["palettes"]}
    assert len(walls) == 1 and len(body["palettes"]) >= 2
    accents = [p["roles"]["accent"]["hex"] for p in body["palettes"]]
    assert len(set(accents)) == len(accents)                         # variety from accents
    for p in body["palettes"]:
        assert p["roles"]["dominant"]["source"] == "locked by you"
        assert all(c["passed"] for c in p["checks"] if c["hard"])  # wall rules only warn


def test_lock_accent(client, room):
    body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                       json={"n": 3, "locks": {"accent": "#C46A4A"}}).json()
    accents = {p["roles"]["accent"]["hex"] for p in body["palettes"]}
    assert len(accents) == 1
    assert len({p["roles"]["dominant"]["hex"] for p in body["palettes"]}) == len(body["palettes"])


def test_adjust_shifts_suggestions(client, room):
    def mean_wall_L(adjust=None):
        body = client.post(f"/api/rooms/{room['room_id']}/palettes",
                           json={"n": 4, "adjust": adjust}).json()
        return np.mean([p["roles"]["dominant"]["lab"][0] for p in body["palettes"]]), body

    base_L, _ = mean_wall_L()
    lighter_L, body = mean_wall_L({"lightness": 2})
    assert lighter_L > base_L + 4
    assert body["adjust"] == {"lightness": 2.0}
    assert any("lighter" in f for f in body["palettes"][0]["facts"])


def test_bad_lock_rejected(client, room):
    r = client.post(f"/api/rooms/{room['room_id']}/palettes", json={"locks": {"dominant": "navy"}})
    assert r.status_code == 422


# ---------- click-to-fix masks ----------

@pytest.fixture(scope="module")
def dining(client):
    return upload(client, DINING)


def _preview(client, rid, hx="#E8DCC8"):
    r = client.post(f"/api/rooms/{rid}/preview", json={"wall": hx, "format": "png"})
    return np.asarray(Image.open(io.BytesIO(r.content))).astype(int)


def test_exclude_panelling_then_undo(client, dining):
    rid = dining["room_id"]
    x, y = PANELLING
    original = np.asarray(Image.open(io.BytesIO(client.get(f"/api/rooms/{rid}/original").content))).astype(int)
    before = _preview(client, rid)
    assert np.abs(before[y, x] - original[y, x]).sum() > 30          # panelling gets painted today

    r = client.post(f"/api/rooms/{rid}/mask", json={"x": x, "y": y, "mode": "exclude"})
    assert r.status_code == 200, r.text
    edit = r.json()
    assert edit["applied"] and edit["edits"] == 1
    assert edit["changed_fraction"] > 0.03                           # the whole wainscot, not one panel
    assert edit["coverage"] < dining["paintable"]["wall"]

    after = _preview(client, rid)
    assert np.abs(after[y, x] - original[y, x]).sum() < 20           # panelling left as wood
    assert np.abs(after[200, 80] - original[200, 80]).sum() > 30     # upper wall still painted

    mask_img = client.get(f"/api/rooms/{rid}/mask")
    assert mask_img.status_code == 200 and mask_img.headers["x-mask-edits"] == "1"

    undone = client.post(f"/api/rooms/{rid}/mask/undo", json={"group": "wall"}).json()
    assert undone["edits"] == 0
    assert undone["coverage"] == pytest.approx(dining["paintable"]["wall"], abs=0.01)
    assert np.abs(_preview(client, rid)[y, x] - original[y, x]).sum() > 30


def test_part_level_is_smaller(client, dining):
    rid = dining["room_id"]
    obj = client.post(f"/api/rooms/{rid}/mask", json={"x": PANELLING[0], "y": PANELLING[1]}).json()
    client.post(f"/api/rooms/{rid}/mask/reset", json={"group": "wall"})
    part = client.post(f"/api/rooms/{rid}/mask",
                       json={"x": PANELLING[0], "y": PANELLING[1], "level": "part"}).json()
    client.post(f"/api/rooms/{rid}/mask/reset", json={"group": "wall"})
    assert part["region_fraction"] < obj["region_fraction"]


def test_include_and_reset(client, dining):
    rid = dining["room_id"]
    start = client.post(f"/api/rooms/{rid}/mask/reset", json={"group": "wall"}).json()["coverage"]
    inc = client.post(f"/api/rooms/{rid}/mask", json={"x": 500, "y": 600, "mode": "include"}).json()
    assert inc["applied"] and inc["coverage"] > start                # the rug added to the paint area
    reset = client.post(f"/api/rooms/{rid}/mask/reset", json={"group": "wall"}).json()
    assert reset["edits"] == 0 and reset["coverage"] == pytest.approx(start)


def test_mask_click_validation(client, dining):
    rid = dining["room_id"]
    assert client.post(f"/api/rooms/{rid}/mask", json={"x": 5000, "y": 10}).status_code == 422
    assert client.post(f"/api/rooms/{rid}/mask", json={"x": 10, "y": 10, "mode": "erase"}).status_code == 422
    assert client.post(f"/api/rooms/{rid}/mask", json={"x": 10, "y": 10, "group": "floor"}).status_code == 422
