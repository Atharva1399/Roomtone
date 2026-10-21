"""Nudge a colour the way people ask for it: warmer/cooler, lighter/darker,
muted/vivid. Works in CIELAB, so one step is roughly the same visible change
for any starting colour.

Each control is a number of steps, usually -3..+3:
    warmth      +: toward yellow-red    -: toward blue
    lightness   +: lighter               -: darker
    saturation  +: more colourful        -: greyer / more muted
"""

import numpy as np

from app.pipeline.colorutils import lab_to_lch, lch_to_lab

STEP_L = 5.0        # L* per lightness step (~ one paint-chip row)
STEP_C = 4.0        # chroma per saturation step
STEP_WARM = 3.0     # Lab units per warmth step

# Warm direction in the a*/b* plane: mostly yellow (+b*), a little red (+a*).
_WARM_DIR = np.array([0.35, 1.0]) / np.hypot(0.35, 1.0)


def adjust_lab(lab, warmth: float = 0, lightness: float = 0, saturation: float = 0) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64).copy()

    # Warmth: slide in the a*/b* plane. On a near-neutral this adds an undertone;
    # on a strong colour it shifts the hue a little toward or away from orange.
    lab[1:] += _WARM_DIR * STEP_WARM * warmth

    L, C, H = lab_to_lch(lab)
    L = float(np.clip(L + STEP_L * lightness, 3, 98))
    # Muting reduces chroma proportionally too, so very colourful paints can be
    # tamed in a few steps without first crossing into grey.
    C = C + STEP_C * saturation if saturation >= 0 else C * (0.8 ** -saturation)
    C = float(np.clip(C, 0, 90))
    return lch_to_lab(np.array([L, C, H]))
