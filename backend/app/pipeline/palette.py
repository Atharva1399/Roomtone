"""Measure the room's current colours.

Clustering runs in CIELAB, where Euclidean distance roughly matches perceived
colour difference, so clusters group colours the way people see them.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from app import config
from app.pipeline.colorutils import (
    delta_e, lab_to_lch, lab_to_rgb, light_reflectance_value, rgb_to_hex, rgb_to_lab,
)
from app.pipeline.segmentation import Segmentation

# Ignore near-black and blown-out pixels: they are shadows/highlights, not surface colour.
_MIN_L, _MAX_L = 8.0, 97.0


@dataclass
class Swatch:
    hex: str
    lab: list[float]
    weight: float          # share of the sampled pixels (0-1)
    lrv: float             # light reflectance value (0-100)
    chroma: float          # colourfulness in LCh

    @classmethod
    def from_lab(cls, lab: np.ndarray, weight: float) -> "Swatch":
        rgb = lab_to_rgb(lab)
        return cls(
            hex=rgb_to_hex(rgb),
            lab=[round(float(v), 2) for v in lab],
            weight=round(float(weight), 4),
            lrv=round(light_reflectance_value(rgb), 1),
            chroma=round(float(lab_to_lch(lab)[1]), 1),
        )


def _sample_lab(image_rgb: np.ndarray, mask: np.ndarray | None, rng) -> np.ndarray:
    pixels = image_rgb.reshape(-1, 3) if mask is None else image_rgb[mask]
    if len(pixels) > config.MAX_SAMPLE_PIXELS:
        pixels = pixels[rng.choice(len(pixels), config.MAX_SAMPLE_PIXELS, replace=False)]
    lab = rgb_to_lab(pixels)
    return lab[(lab[:, 0] > _MIN_L) & (lab[:, 0] < _MAX_L)]


def extract_palette(image_rgb: np.ndarray, mask: np.ndarray | None = None,
                    k: int = 5, merge_delta_e: float = 6.0, seed: int = 0) -> list[Swatch]:
    """k-means in CIELAB, then merge clusters that look alike (CIEDE2000 < merge_delta_e).

    image_rgb: (H, W, 3) float in [0, 1]. Returns swatches sorted by weight.
    """
    rng = np.random.default_rng(seed)
    lab = _sample_lab(image_rgb, mask, rng)
    if len(lab) < k:
        return []

    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(lab)
    counts = np.bincount(km.labels_, minlength=k).astype(float)
    centers = list(km.cluster_centers_)

    # Merge near-duplicates: k-means often splits one wall into lit/shaded halves.
    merged: list[tuple[np.ndarray, float]] = []
    for i in np.argsort(-counts):
        c, n = centers[i], counts[i]
        for j, (mc, mn) in enumerate(merged):
            if delta_e(mc, c) < merge_delta_e:
                merged[j] = ((mc * mn + c * n) / (mn + n), mn + n)
                break
        else:
            merged.append((c, n))

    total = sum(n for _, n in merged)
    return sorted((Swatch.from_lab(c, n / total) for c, n in merged), key=lambda s: -s.weight)


def region_color(image_rgb: np.ndarray, mask: np.ndarray) -> Swatch | None:
    """The single most representative colour of a region (its biggest cluster)."""
    swatches = extract_palette(image_rgb, mask, k=3)
    return swatches[0] if swatches else None


def estimate_lighting(image_rgb: np.ndarray, seg: Segmentation) -> dict:
    """Rough read of the room's light: brightness and warm/cool cast.

    Uses wall + ceiling pixels, since those are usually near-neutral surfaces lit by
    the room's light. Positive b* on them means warm (yellowish) light.
    """
    masks = [seg.group_masks[g] for g in ("wall", "ceiling") if g in seg.group_masks]
    mask = np.logical_or.reduce(masks) if masks else np.ones(image_rgb.shape[:2], bool)
    if mask.sum() < 500:
        mask = np.ones(image_rgb.shape[:2], bool)

    lab = rgb_to_lab(image_rgb[mask][:: max(1, int(mask.sum()) // config.MAX_SAMPLE_PIXELS)])
    mean_L, mean_b = float(np.median(lab[:, 0])), float(np.median(lab[:, 2]))
    temperature = "warm" if mean_b > 8 else "cool" if mean_b < -2 else "neutral"
    brightness = "bright" if mean_L > 70 else "dim" if mean_L < 45 else "medium"
    return {"brightness": brightness, "temperature": temperature,
            "median_L": round(mean_L, 1), "median_b": round(mean_b, 1)}


def assign_roles(regions: dict[str, Swatch], fractions: dict[str, float],
                 overall: list[Swatch]) -> dict[str, dict]:
    """Map what is in the photo onto the 60-30-10 roles.

    dominant  = the biggest surface (normally walls)
    secondary = the biggest remaining surface that looks different from dominant
    accent    = the most colourful small colour in the room
    """
    ranked = sorted(((g, fractions[g]) for g in regions), key=lambda x: -x[1])
    if not ranked:
        return {}
    # Walls set a room's dominant colour whenever they're a substantial share of the
    # view, even if a big window or bed happens to cover more pixels in this shot.
    dom_group = "wall" if fractions.get("wall", 0) >= 0.15 else ranked[0][0]
    ranked = [(dom_group, fractions[dom_group])] + [r for r in ranked if r[0] != dom_group]
    roles = {"dominant": {"source": dom_group, **asdict(regions[dom_group])}}

    dom_lab = regions[dom_group].lab
    for g, _ in ranked[1:]:
        if delta_e(regions[g].lab, dom_lab) > 10:
            roles["secondary"] = {"source": g, **asdict(regions[g])}
            break

    used = [r["lab"] for r in roles.values()]
    candidates = [s for s in overall if all(delta_e(s.lab, u) > 12 for u in used)]
    if candidates:
        accent = max(candidates, key=lambda s: s.chroma)
        roles["accent"] = {"source": "overall", **asdict(accent)}
    return roles


def analyze_colors(image: Image.Image, seg: Segmentation) -> dict:
    """Full colour read-out of a room photo: per-region colours, overall palette,
    current 60-30-10 roles and lighting."""
    rgb = np.asarray(image, dtype=np.float64) / 255.0

    regions, fractions = {}, {}
    for group, mask in seg.group_masks.items():
        frac = float(mask.mean())
        if frac < config.MIN_REGION_FRACTION:
            continue
        sw = region_color(rgb, mask)
        if sw:
            regions[group], fractions[group] = sw, frac

    overall = extract_palette(rgb, k=8)
    return {
        "regions": {g: {"area": round(fractions[g], 4), **asdict(s)} for g, s in regions.items()},
        "overall_palette": [asdict(s) for s in overall],
        "current_roles": assign_roles(regions, fractions, overall),
        "lighting": estimate_lighting(rgb, seg),
    }
