"""Tier A preview: repaint surfaces while keeping the photo's real lighting.

Model: what the camera sees ~= paint colour x lighting (in linear RGB).
For each painted region we estimate the lighting from the original pixels:
its brightness (shading) per pixel and its colour (e.g. warm lamp pools) as a
smooth tint, then multiply both by the new paint colour. Shadows, light
falloff, corners, lamp glow and wall texture all come from the original photo,
so the only thing that changes is the paint.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from app.pipeline.colorutils import (
    delta_e, hex_to_rgb, lab_to_lch, linear_to_srgb, relative_luminance, rgb_to_lab, srgb_to_linear,
)
from app.pipeline.segmentation import PAINTABLE_GROUPS, Segmentation

# Pixels with group probability above this are treated as "certainly this surface"
# when estimating the reference lighting level and the paint's chromaticity.
_CORE_PROB = 0.9

# Chromaticity distance (rg space) at which a "wall" pixel stops being treated as
# the same paint. Brightness-independent, so shadows on one wall still match, while
# wood panelling or furniture mislabelled as wall falls outside.
_SURFACE_MATCH = (0.05, 0.10)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware smoothing (He et al. 2010): src's edges snap to guide's edges.
    Used to pull blobby segmentation masks onto real object boundaries."""
    I, p = guide.astype(np.float32), src.astype(np.float32)
    mean_I, mean_p = _box(I, r), _box(p, r)
    a = (_box(I * p, r) - mean_I * mean_p) / (_box(I * I, r) - mean_I ** 2 + eps)
    b = mean_p - a * mean_I
    return _box(a, r) * I + _box(b, r)


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def _chromaticity(lin: np.ndarray) -> np.ndarray:
    """rg chromaticity: colour without brightness, so a shadowed and a lit patch of
    the same paint land at the same point."""
    s = lin.sum(axis=-1, keepdims=True)
    return (lin / np.maximum(s, 1e-6))[..., :2]


def _surface_gate(lin: np.ndarray, core: np.ndarray) -> np.ndarray:
    """1 where a pixel looks like the main paint of this surface, 0 where it is
    clearly a different material (wood panelling, a wardrobe, a poster)."""
    chroma = _chromaticity(lin)
    samples = chroma[core]
    if len(samples) < 200:
        return np.ones(lin.shape[:2], np.float32)
    if len(samples) > 20_000:
        samples = samples[np.random.default_rng(0).choice(len(samples), 20_000, replace=False)]

    # The main paint is the largest chromaticity cluster among confident wall pixels.
    km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(samples)
    counts = np.bincount(km.labels_, minlength=2)
    main = km.cluster_centers_[np.argmax(counts)]

    dist = np.linalg.norm(chroma - main, axis=-1)
    gate = 1 - _smoothstep(dist, *_SURFACE_MATCH)
    # Near-black pixels have unreliable chromaticity; trust segmentation there.
    gate[relative_luminance(lin) < 0.01] = 1
    return gate.astype(np.float32)


def _alpha(prob: np.ndarray, gate: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Blend mask: cut the model's probability crisply at 0.5, drop off-colour
    materials, then anti-alias along the photo's real edges.

    The segmentation runs at about a quarter of the photo's resolution, so its edges
    arrive as 10-20 px ramps; a wide blend there smears paint over the neighbouring
    surface. A narrow cut plus a 1 px edge-aware filter keeps edges clean: on 300
    hand-labelled ADE20K rooms the soft edge band fell from 7.4 px to 2.6 px and
    misplaced paint by 7 %, with wall IoU unchanged. (Re-deciding the edge band from
    the photo's colours, or with SAM, placed edges no better on that test.)
    """
    alpha = _smoothstep(prob, 0.45, 0.55) * gate
    alpha = guided_filter(gray, alpha.astype(np.float32), r=1, eps=1e-4)
    return np.clip(alpha, 0, 1)


def _shading(lin: np.ndarray, core: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-pixel lighting relative to the typical lit wall pixel (1.0 = typical)."""
    Y = relative_luminance(lin).astype(np.float32)
    y_ref = float(np.median(Y[core])) if core.any() else float(np.median(Y))
    y_ref = max(y_ref, 1e-4)
    shading = Y / y_ref
    # Light edge-preserving denoise: going from a dark/saturated paint to a light one
    # otherwise amplifies sensor and JPEG noise into visible grain. Darker originals
    # are noisier, so they get stronger smoothing.
    strength = 0.25 if y_ref < 0.03 else 0.08
    shading = cv2.bilateralFilter(shading, 5, strength, 3)
    return shading, y_ref


def _light_tint(lin: np.ndarray, core: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Colour of the light falling on the surface, relative to its typical spot.

    A white wall under a warm lamp photographs yellow near the lamp; new paint
    there would look warm too. We take each pixel's colour-per-unit-brightness
    relative to the wall's typical value, and blur it heavily (masked to the
    wall) so it captures the slow-changing light colour, not texture or stains.
    Light colour can only be read off a near-neutral surface: on a saturated wall
    (deep orange) the weak channels are mostly noise, so the tint fades out as the
    original paint gets more colourful.

    Returns an (H, W, 3) multiplier with unit luminance.
    """
    Y = np.maximum(relative_luminance(lin), 1e-4)[..., None]
    chroma = lin / Y
    ref = np.median(chroma[core], axis=0)
    ratio = (chroma / np.maximum(ref, 1e-4)).astype(np.float32)

    w = core.astype(np.float32)
    blur_w = cv2.GaussianBlur(w, (0, 0), sigma) + 1e-4
    tint = np.stack([cv2.GaussianBlur(ratio[..., c] * w, (0, 0), sigma) / blur_w for c in range(3)], axis=-1)
    tint = np.clip(tint, 0.75, 1.35)

    ref_chroma = float(lab_to_lch(rgb_to_lab(linear_to_srgb(np.median(lin[core], axis=0))))[1])
    strength = float(np.clip(1 - ref_chroma / 30, 0, 1))
    tint = 1 + (tint - 1) * strength
    return tint / np.maximum(relative_luminance(tint), 1e-4)[..., None]


def is_plain_surface(image: Image.Image, seg: Segmentation, group: str) -> bool:
    """True if a surface is one flat colour (safe to repaint), False if it carries
    decoration such as a painted or stencilled ceiling."""
    mask = seg.group_probs.get(group)
    if mask is None or seg.group_masks[group].mean() < 0.01:
        return False
    core = mask > _CORE_PROB
    if core.sum() < 500:
        return False
    lab = rgb_to_lab(np.asarray(image, dtype=np.float64)[core][::7] / 255.0)
    return float(np.std(lab[:, 1]) + np.std(lab[:, 2])) < 6.0


@dataclass
class SurfaceLayers:
    """Everything about one surface that doesn't depend on the new paint colour.
    Computed once per photo (~0.5s); applying a colour on top is then ~50ms."""
    light: np.ndarray        # (H, W, 3) float32: shading x light tint, multiplies the paint
    alpha: np.ndarray        # (H, W, 1) float32 blend mask
    typical: np.ndarray      # (N,) flat indices of typically-lit pixels, for verification
    y_ref: float
    segmented_coverage: float
    painted_coverage: float


def _subsample(idx: np.ndarray, n: int) -> np.ndarray:
    return idx if idx.size <= n else idx[np.linspace(0, idx.size - 1, n).astype(np.int64)]


# linear -> sRGB through a lookup table: ~10x faster than the exact power curve on a
# full frame, and 16k steps is far finer than the 8-bit output can show.
_LUT_SIZE = 16384
_SRGB_LUT = linear_to_srgb(np.linspace(0, 1, _LUT_SIZE)).astype(np.float32)


def _linear_to_srgb_fast(lin: np.ndarray) -> np.ndarray:
    idx = (np.clip(lin, 0, 1) * (_LUT_SIZE - 1) + 0.5).astype(np.int32)
    return _SRGB_LUT[idx]


def prepare_surface(image: Image.Image, seg: Segmentation, group: str,
                    strict_surface: bool = False, prob: np.ndarray | None = None) -> SurfaceLayers | None:
    """Precompute a surface's lighting and mask. None if it isn't in the photo.

    strict_surface: also skip "wall" pixels whose colour differs from the main
    paint (catches wood panelling). Off by default: coloured light (lamp glow,
    window daylight) shifts colour as much as a material change does, so it
    also leaves holes in properly segmented walls.
    prob: the surface's probability map with the user's click edits applied;
    defaults to the segmentation's own.
    """
    if group not in PAINTABLE_GROUPS:
        raise ValueError(f"'{group}' is not paintable; choose from {PAINTABLE_GROUPS}")
    if prob is None:
        prob = seg.group_probs.get(group)
    if prob is None or (prob > 0.5).mean() < 0.002:
        return None

    lin = srgb_to_linear(np.asarray(image, dtype=np.float64) / 255.0)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    core = prob > _CORE_PROB
    gate = _surface_gate(lin, core) if strict_surface else np.ones_like(prob)
    core &= gate > 0.5
    if not core.any():
        return None
    shading, y_ref = _shading(lin, core)
    light = (shading[..., None] * _light_tint(lin, core)).astype(np.float32)
    alpha = _alpha(prob, gate, gray)[..., None].astype(np.float32)

    return SurfaceLayers(
        light=light, alpha=alpha,
        typical=_subsample(np.flatnonzero(core & (np.abs(shading - 1) < 0.1)), 5000),
        y_ref=y_ref,
        segmented_coverage=float((prob > 0.5).mean()),
        painted_coverage=float((alpha > 0.5).mean()),
    )


def apply_paints(image_rgb: np.ndarray, layers: dict[str, SurfaceLayers | None],
                 colors: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Fast path: paint precomputed surfaces. image_rgb is (H, W, 3) float32 in [0, 1].

    Returns the new image (same format) and a per-surface report with the
    measured colour of the result and its CIEDE2000 distance from the paint.
    """
    out = image_rgb.copy()
    report = {}
    for group, hex_color in colors.items():
        lay = layers.get(group)
        if lay is None:
            report[group] = {"applied": False, "reason": "region not found in photo"}
            continue
        target_rgb = hex_to_rgb(hex_color)
        painted_lin = lay.light * srgb_to_linear(target_rgb).astype(np.float32)
        # Soft-clip bright highlights instead of hard clipping to flat white.
        painted_lin = painted_lin / (1 + np.maximum(painted_lin - 1, 0))
        painted = _linear_to_srgb_fast(painted_lin)
        out = lay.alpha * painted + (1 - lay.alpha) * out

        # Verify: median colour of the repainted surface's typically-lit pixels.
        flat = out.reshape(-1, 3)
        measured = np.median(flat[lay.typical], axis=0) if lay.typical.size else target_rgb
        report[group] = {
            "applied": True,
            "target": hex_color.upper(),
            "delta_e_typical_pixel": round(delta_e(rgb_to_lab(measured), rgb_to_lab(target_rgb)), 2),
            "original_reference_luminance": round(lay.y_ref, 4),
            "segmented_coverage": round(lay.segmented_coverage, 4),
            "painted_coverage": round(lay.painted_coverage, 4),
        }
    return out, report


def to_image(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8))


def recolor(image: Image.Image, seg: Segmentation, colors: dict[str, str],
            strict_surface: bool = False) -> tuple[Image.Image, dict]:
    """One-shot repaint, e.g. colors={"wall": "#C9BBA6", "ceiling": "#F4F1EA"}.
    For repeated previews of one photo, cache prepare_surface() and call apply_paints()."""
    layers = {g: prepare_surface(image, seg, g, strict_surface) for g in colors}
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    out, report = apply_paints(rgb, layers, colors)
    return to_image(out), report


def swatch_strip(hex_colors: list[str], width: int = 600, height: int = 80) -> Image.Image:
    """Simple horizontal swatch bar for reports and debugging."""
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    n = max(1, len(hex_colors))
    for i, h in enumerate(hex_colors):
        strip[:, i * width // n:(i + 1) * width // n] = (hex_to_rgb(h) * 255).astype(np.uint8)
    return Image.fromarray(strip)
