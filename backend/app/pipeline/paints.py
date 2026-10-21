"""Match any colour to the nearest real, buyable paint.

Loads every colour book in data/paints/*.json ({name, label, hex}) once and
finds nearest neighbours by CIEDE2000, the industry colour-difference metric.
"""

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from skimage import color as skcolor

from app.pipeline.colorutils import hex_to_rgb, lab_to_lch, light_reflectance_value, rgb_to_lab

PAINT_DIR = Path(__file__).resolve().parents[2] / "data" / "paints"

BRAND_NAMES = {
    "sherwin-williams": "Sherwin-Williams",
    "benjamin-moore": "Benjamin Moore",
    "farrow-ball": "Farrow & Ball",
    "behr": "Behr",
}


@dataclass
class Paint:
    brand: str
    name: str
    code: str
    hex: str
    lab: list[float]
    lrv: float
    match_delta_e: float = 0.0   # distance from the colour that was asked for

    def to_dict(self) -> dict:
        return {"brand": self.brand, "name": self.name, "code": self.code, "hex": self.hex,
                "lab": self.lab, "lrv": self.lrv, "match_delta_e": self.match_delta_e}


@dataclass
class PaintBook:
    brands: np.ndarray
    names: np.ndarray
    codes: np.ndarray
    hexes: np.ndarray
    labs: np.ndarray      # (N, 3)
    lch: np.ndarray       # (N, 3)

    def __len__(self) -> int:
        return len(self.hexes)

    def _paint(self, i: int, de: float) -> Paint:
        return Paint(
            brand=str(self.brands[i]), name=str(self.names[i]), code=str(self.codes[i]),
            hex=str(self.hexes[i]), lab=[round(float(v), 2) for v in self.labs[i]],
            lrv=round(light_reflectance_value(hex_to_rgb(self.hexes[i])), 1),
            match_delta_e=round(float(de), 2),
        )

    def nearest(self, lab, k: int = 1, brands: list[str] | None = None) -> list[Paint]:
        """k closest paints to a Lab colour, optionally restricted to some brands."""
        idx = np.arange(len(self))
        if brands:
            idx = idx[np.isin(self.brands, [BRAND_NAMES.get(b, b) for b in brands])]
        # Cheap Euclidean prefilter, then exact CIEDE2000 on the shortlist.
        d76 = np.linalg.norm(self.labs[idx] - np.asarray(lab), axis=1)
        short = idx[np.argsort(d76)[:max(50, k * 10)]]
        de = skcolor.deltaE_ciede2000(np.broadcast_to(lab, (len(short), 3)), self.labs[short])
        order = np.argsort(de)[:k]
        return [self._paint(int(short[o]), de[o]) for o in order]


@lru_cache(maxsize=1)
def load_paints() -> PaintBook:
    brands, names, codes, hexes = [], [], [], []
    seen = set()
    for path in sorted(PAINT_DIR.glob("*.json")):
        brand = BRAND_NAMES.get(path.stem, path.stem.replace("-", " ").title())
        for entry in json.loads(path.read_text(encoding="utf-8")):
            hx = str(entry.get("hex", "")).strip().upper()
            if len(hx) == 6 and not hx.startswith("#"):
                hx = "#" + hx
            if len(hx) != 7 or (brand, hx) in seen:
                continue
            seen.add((brand, hx))
            brands.append(brand)
            names.append(str(entry.get("name", "")).strip())
            code = str(entry.get("label", "")).strip()
            if brand == "Sherwin-Williams" and code.isdigit():
                code = f"SW {code}"   # how SW codes are printed on chips and in stores
            codes.append(code)
            hexes.append(hx)
    if not hexes:
        raise RuntimeError(f"No paint colour books found in {PAINT_DIR}")

    # Some source files truncate names to their first word (55 Benjamin Moore colours
    # are all "Blue", including a brown). A name shared by 3+ colours of one brand is
    # unreliable, so drop it; the brand code is what you order by anyway.
    counts = Counter(zip(brands, names))
    names = ["" if counts[(b, n)] >= 3 else n for b, n in zip(brands, names)]

    rgb = np.array([hex_to_rgb(h) for h in hexes])
    labs = rgb_to_lab(rgb)
    return PaintBook(np.array(brands), np.array(names), np.array(codes), np.array(hexes),
                     labs, lab_to_lch(labs))
