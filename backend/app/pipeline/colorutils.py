"""Colour-space helpers. All RGB arrays are float in [0, 1] unless noted."""

import numpy as np
from skimage import color as skcolor


def hex_to_rgb(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected a 6-digit hex colour, got {hex_str!r}")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(float(np.clip(c, 0, 1)) * 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Works on a single colour (3,) or any (..., 3) array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return skcolor.rgb2lab(rgb.reshape(-1, 1, 3)).reshape(rgb.shape)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    return np.clip(skcolor.lab2rgb(lab.reshape(-1, 1, 3)).reshape(lab.shape), 0, 1)


def lab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Lightness, chroma, hue (degrees 0-360)."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return np.stack([L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360], axis=-1)


def lch_to_lab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float64)
    L, C, H = lch[..., 0], lch[..., 1], np.radians(lch[..., 2])
    return np.stack([L, C * np.cos(H), C * np.sin(H)], axis=-1)


def delta_e(lab1, lab2) -> float:
    """CIEDE2000 colour difference. ~1 is barely visible, >5 is clearly different."""
    return float(skcolor.deltaE_ciede2000(np.asarray(lab1), np.asarray(lab2)))


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0, None)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def relative_luminance(lin_rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of linear RGB."""
    return lin_rgb[..., 0] * 0.2126 + lin_rgb[..., 1] * 0.7152 + lin_rgb[..., 2] * 0.0722


def light_reflectance_value(rgb) -> float:
    """LRV (0-100) as paint brands quote it: how much light a colour reflects."""
    return float(relative_luminance(srgb_to_linear(np.asarray(rgb))) * 100)
