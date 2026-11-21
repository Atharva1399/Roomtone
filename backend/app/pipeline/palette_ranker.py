"""Roomtone-Palette (V2): a learned score for how well a wall colour suits a room.

Trained in training/roomtone_palette.py on ADE20K's indoor rooms. The room is measured the same
way as in training: each group's dominant CIELAB colour after removing the light's tint (taken
from the ceiling, or the brightest non-wall pixels) and normalising exposure, both measured
without the wall. The candidate is a paint colour, i.e. already a "true" colour.

measure_context() and features() must stay identical to the notebook's measure_room() and
features(); tests/test_palette_ranker.py checks that.
"""

from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

from app import config

GROUPS = ["wall", "ceiling", "floor", "window", "door", "sofa", "bed", "table", "storage", "decor"]
CONTEXT = ["floor", "sofa", "bed", "table", "storage", "door", "window", "decor"]
WORK_SIDE = 512


class Ranker(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)


@lru_cache(maxsize=1)
def load_ranker() -> Ranker | None:
    path = config.PALETTE_RANKER_DIR / "ranker.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = Ranker(state["mean"].numel())
    model.load_state_dict(state)
    return model.eval()


def available() -> bool:
    return load_ranker() is not None


def _dominant_lab(lab_px: np.ndarray, rng) -> np.ndarray | None:
    lab_px = lab_px[(lab_px[:, 0] > 8) & (lab_px[:, 0] < 99)]
    if len(lab_px) < 50:
        return None
    if len(lab_px) > 3000:
        lab_px = lab_px[rng.choice(len(lab_px), 3000, replace=False)]
    km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(lab_px)
    return km.cluster_centers_[np.bincount(km.labels_, minlength=3).argmax()]


def measure_context(image_rgb_u8: np.ndarray, group_masks: dict[str, np.ndarray]) -> dict | None:
    """The room as the ranker sees it: light, exposure and each group's corrected colour and area."""
    img = image_rgb_u8
    grp = np.full(img.shape[:2], 255, np.uint8)
    for gi, g in enumerate(GROUPS):
        if g in group_masks:
            grp[group_masks[g]] = gi
    s = WORK_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        grp = cv2.resize(grp, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    lab = rgb2lab(img.astype(np.float32) / 255).reshape(-1, 3)
    grp = grp.reshape(-1)
    labelled = np.ones(len(grp), bool)     # segmentation labels every pixel; 255 = in no group
    wall = grp == GROUPS.index("wall")
    if labelled.sum() == 0 or wall.sum() / labelled.sum() < 0.05:
        return None

    ceil = grp == GROUPS.index("ceiling")
    not_wall = labelled & ~wall & (grp != GROUPS.index("window"))
    if ceil.sum() / labelled.sum() >= 0.02:
        ref = np.median(lab[ceil], axis=0)
    else:
        Ls = lab[not_wall, 0]
        bright = lab[not_wall][Ls >= np.percentile(Ls, 80)] if not_wall.sum() > 100 else lab[labelled]
        ref = np.median(bright, axis=0)
    lit = not_wall | ceil
    exposure = float(np.clip(95 / max(np.percentile(lab[lit, 0], 95) if lit.sum() > 100 else ref[0], 1), 0.8, 2.0))
    norm = lab.copy()
    norm[:, 0] = np.clip(norm[:, 0] * exposure, 0, 100)
    norm[:, 1:] -= ref[1:]

    rng = np.random.default_rng(0)
    ctx = {"light": [float(ref[0]), float(ref[1]), float(ref[2]), exposure]}
    total = labelled.sum()
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        area = m.sum() / total
        c = _dominant_lab(norm[m], rng) if area >= 0.005 else None
        ctx[g] = [float(v) for v in c] if c is not None else None
        ctx[g + "_area"] = float(area) if c is not None else 0.0
    return ctx


def _lch(lab):
    a, b = lab[..., 1], lab[..., 2]
    return np.sqrt(a ** 2 + b ** 2), np.arctan2(b, a)


def features(ctx: dict, cand: np.ndarray) -> np.ndarray:
    """One room's context and (N, 3) candidate wall Lab colours -> (N, F) features."""
    cand = np.asarray(cand, np.float32)
    n = len(cand)
    cC, cH = _lch(cand)
    light = np.tile(np.asarray(ctx["light"], np.float32), (n, 1))
    parts = [cand / np.array([100, 50, 50], np.float32), (cC / 50)[:, None],
             np.cos(cH)[:, None], np.sin(cH)[:, None], light / np.array([100, 20, 20, 2], np.float32)]
    for g in CONTEXT:
        present_one = ctx.get(g) is not None and ctx.get(g + "_area", 0) > 0
        lab0 = np.tile(np.asarray(ctx[g] if present_one else (0, 0, 0), np.float32), (n, 1))
        area = np.full(n, ctx.get(g + "_area", 0.0) if present_one else 0.0, np.float32)
        present = np.full(n, present_one)
        gC, gH = _lch(lab0)
        de = np.where(present, deltaE_ciede2000(np.where(present[:, None], cand, 0), lab0), 0)
        dh = cH - gH
        w = np.minimum(gC, 20) / 20 * np.minimum(cC, 20) / 20
        parts += [present[:, None].astype(np.float32), np.sqrt(area)[:, None], lab0 / np.array([100, 50, 50]),
                  (de / 50)[:, None], ((cand[:, 0] - lab0[:, 0]) / 50 * present)[:, None],
                  ((cC - gC) / 50 * present)[:, None], (np.cos(dh) * w)[:, None], (np.sin(dh) * w)[:, None]]
    return np.concatenate(parts, 1).astype(np.float32)


def score_walls(ctx: dict, walls_lab: np.ndarray) -> np.ndarray | None:
    """Model rating per candidate wall (higher suits the room better), or None without the model."""
    model = load_ranker()
    if model is None or ctx is None or len(walls_lab) == 0:
        return None
    with torch.no_grad():
        return model(torch.from_numpy(features(ctx, walls_lab))).numpy()
