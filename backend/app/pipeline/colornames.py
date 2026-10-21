"""Plain colour words from measured CIELAB values.

Language models (small local ones especially) guess badly from hex codes: they
call a pale blue "warm ivory". Everything the explanation step says about how a
colour looks is therefore computed here and handed over as words.
"""

import numpy as np

from app.pipeline.colorutils import hex_to_rgb, lab_to_lch, rgb_to_lab

# Hue families by CIELAB hue angle, from measured reference colours
# (pink ~2, burgundy ~17, red/coral ~35, terracotta ~46, orange ~65, sand ~84,
#  yellow ~91, olive ~106, sage ~135, green ~144, teal/aqua ~199, blue ~250-275,
#  lavender ~305, purple ~312, plum ~341).
_FAMILIES = [
    (12, "pink"), (30, "rose"), (42, "red"), (58, "terracotta"), (75, "orange"),
    (95, "sand"), (102, "yellow"), (122, "olive"), (175, "green"), (220, "teal"),
    (290, "blue"), (325, "violet"), (352, "plum"), (361, "pink"),
]


def _family(h: float) -> str:
    for upper, name in _FAMILIES:
        if h < upper:
            return name
    return "pink"


def describe_lab(lab) -> str:
    """e.g. 'pale soft blue', 'light warm beige', 'deep green', 'off-white'."""
    L, C, H = (float(v) for v in lab_to_lch(np.asarray(lab, dtype=float)))
    light = ("very pale" if L > 88 else "pale" if L > 76 else "light" if L > 64
             else "mid-tone" if L > 48 else "deep" if L > 32 else "very dark")
    if C < 4:
        if L > 90:
            return "off-white"
        return f"{light} grey" if L > 32 else "near-black"
    fam = _family(H)
    if C < 12:
        # low-chroma colours read as tinted neutrals
        if fam in ("orange", "sand", "yellow", "terracotta"):
            return f"{light} warm beige" if L > 48 else f"{light} warm brown-grey"
        if fam in ("teal", "blue"):
            return f"{light} soft {fam}-grey"
        return f"{light} soft {fam}"
    strength = "muted" if C < 22 else "" if C < 40 else "strong"
    if fam in ("orange", "terracotta") and L < 48:
        fam = "brown"
    elif fam in ("sand", "yellow", "olive") and L < 50:
        fam = "olive-brown"
    elif fam in ("red", "rose", "pink") and L < 40:
        fam = "burgundy"
    return " ".join(w for w in (light, strength, fam) if w)


def describe_hex(hx: str) -> str:
    return describe_lab(rgb_to_lab(hex_to_rgb(hx)))
