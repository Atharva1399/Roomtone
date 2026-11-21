"""Roomtone-Palette (V2) in the backend: same features as training, and it re-ranks palettes."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app import config
from app.pipeline import palette_ranker
from app.pipeline.suggest import suggest_palettes

NOTEBOOK = Path(__file__).resolve().parents[2] / "training" / "roomtone_palette.py"


@pytest.fixture(scope="module")
def notebook():
    spec = importlib.util.spec_from_file_location("roomtone_palette_nb", NOTEBOOK)
    nb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb)
    return nb


def _context():
    return {"light": [72.0, 1.5, 6.0, 1.2],
            "floor": [48.0, 9.0, 22.0], "floor_area": 0.21,
            "sofa": [35.0, -4.0, -18.0], "sofa_area": 0.08,
            "bed": None, "bed_area": 0.0,
            "table": [40.0, 12.0, 25.0], "table_area": 0.03,
            "storage": None, "storage_area": 0.0,
            "door": [70.0, 2.0, 8.0], "door_area": 0.04,
            "window": [85.0, -1.0, 3.0], "window_area": 0.10,
            "decor": [55.0, 30.0, 35.0], "decor_area": 0.01}


def test_features_match_the_notebook(notebook):
    ctx = _context()
    row = {"light_L": ctx["light"][0], "light_a": ctx["light"][1], "light_b": ctx["light"][2],
           "exposure": ctx["light"][3]}
    for g in notebook.GROUPS:
        lab = ctx.get(g) or [np.nan] * 3
        row.update({f"{g}_L": lab[0], f"{g}_a": lab[1], f"{g}_b": lab[2], f"{g}_area": ctx.get(g + "_area", 0.0)})
    row.update({"wall_L": 80.0, "wall_a": 0.0, "wall_b": 5.0, "wall_area": 0.4})
    nb_ctx = notebook.room_context(pd.DataFrame([row]))
    cands = np.random.default_rng(0).uniform([20, -30, -30], [95, 30, 30], (12, 3)).astype(np.float32)
    ours = palette_ranker.features(ctx, cands)
    theirs = notebook.features(nb_ctx, np.zeros(len(cands), int), cands)
    assert ours.shape == theirs.shape
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-5)


def test_measure_context_removes_the_lights_tint():
    h, w = 200, 300
    img = np.zeros((h, w, 3), np.uint8)
    masks = {g: np.zeros((h, w), bool) for g in ("wall", "ceiling", "floor")}
    masks["ceiling"][:40] = True
    masks["wall"][40:140] = True
    masks["floor"][140:] = True
    img[masks["ceiling"]] = (255, 236, 200)   # white ceiling under warm light
    img[masks["wall"]] = (200, 190, 170)
    img[masks["floor"]] = (215, 196, 160)     # a white floor would also look warm
    ctx = palette_ranker.measure_context(img, masks)
    assert ctx is not None and ctx["light"][2] > 10            # the warm cast is measured…
    assert abs(ctx["floor"][1]) < 3 and abs(ctx["floor"][2]) < 3  # …and removed from the floor
    assert ctx["floor_area"] == pytest.approx(60 / 200, abs=0.01)
    assert ctx["sofa"] is None and ctx["sofa_area"] == 0.0


ANALYSIS = {
    "regions": {"floor": {"hex": "#8a6a4a", "lab": [48.0, 9.0, 22.0], "area": 0.21},
                "sofa": {"hex": "#3a4a6a", "lab": [35.0, -4.0, -18.0], "area": 0.08}},
    "lighting": {"brightness": "medium", "temperature": "neutral", "median_L": 60, "median_b": 4},
}


@pytest.fixture
def tiny_ranker(tmp_path, monkeypatch):
    """A ranker that simply prefers warmer (higher b*) walls, saved like the notebook's export."""
    n_in = palette_ranker.features(_context(), np.zeros((1, 3), np.float32)).shape[1]
    model = palette_ranker.Ranker(n_in)
    with torch.no_grad():
        for layer in model.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_(); layer.bias.zero_()
        model.net[0].weight[0, 2] = 1.0          # feature 2 = candidate b* / 50
        model.net[3].weight[0, 0] = 1.0
        model.net[6].weight[0, 0] = 1.0
    torch.save(model.state_dict(), tmp_path / "ranker.pt")
    monkeypatch.setattr(config, "PALETTE_RANKER_DIR", tmp_path)
    palette_ranker.load_ranker.cache_clear()
    yield
    palette_ranker.load_ranker.cache_clear()


def test_without_the_model_v1_rules_rank_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PALETTE_RANKER_DIR", tmp_path / "missing")
    palette_ranker.load_ranker.cache_clear()
    out = suggest_palettes(ANALYSIS, n=4, ranker_context=_context())
    palette_ranker.load_ranker.cache_clear()
    assert out["ranked_by"] == "V1 rules"


def test_the_model_reranks_palettes_that_pass_the_checks(tiny_ranker):
    v1 = suggest_palettes(ANALYSIS, n=8, ranker_context=None)
    v2 = suggest_palettes(ANALYSIS, n=8, ranker_context=_context())
    assert v1["ranked_by"] == "V1 rules"
    assert v2["ranked_by"] == "V1 rules + V2 roomtone-palette"
    b = lambda out: np.mean([p["roles"]["dominant"]["lab"][2] for p in out["palettes"][:3]])
    assert b(v2) > b(v1)                       # the warm-loving model pulled warmer walls up
    for p in v2["palettes"]:                   # and every palette still passes V1's hard checks
        assert all(c["passed"] for c in p["checks"] if c["hard"])


def test_a_locked_wall_is_not_reranked(tiny_ranker):
    out = suggest_palettes(ANALYSIS, n=4, locks={"dominant": "#d8cfc0"}, ranker_context=_context())
    assert out["ranked_by"] == "V1 rules"
