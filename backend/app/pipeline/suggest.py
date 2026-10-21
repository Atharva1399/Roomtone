"""Generate wall-paint palettes for a room.

How a palette is derived (the answer to "how do you derive a palette?"):

1. Anchor. Walls are what we change; the floor and big furniture usually stay.
   The largest fixed surface (floor, sofa, bed...) is the anchor, and the most
   colourful fixed item sets the base hue the palette is built around.
2. Harmony. Classic colour-wheel schemes (tonal, analogous, complementary,
   split-complementary, triadic, plus warm/cool neutrals) give candidate wall
   and accent hues relative to that base hue, in CIELAB LCh.
3. Lightness & strength. The user's goal (calm / cosy / bright / bold) sets
   the wall's lightness and chroma; a dim room is pushed lighter, a cool-lit
   room gets warmer undertones.
4. 60-30-10. Dominant (~60%) = new wall paint; secondary (~30%) = the existing
   anchor the walls are designed around; accent (~10%) = a stronger colour for
   cushions, art or a feature piece. Ceiling/trim gets a white tinted to the wall.
5. Real paint. Every colour snaps to the nearest paint in real colour books
   (CIEDE2000), and all checks run again on the snapped colours.
6. Checks. Rules reject palettes that would fail in a real room (wall blends
   into the floor, accent too weak, wall too dark for a dim room...). The rest
   are scored and the best, most varied ones returned.
"""

from dataclasses import dataclass, field

import numpy as np

from app.pipeline.colorutils import (
    delta_e, hex_to_rgb, lab_to_lch, lch_to_lab, light_reflectance_value, rgb_to_lab,
)
from app.pipeline.adjust import STEP_L, adjust_lab
from app.pipeline.paints import load_paints
from app.pipeline.segmentation import PAINTABLE_GROUPS

# Wall lightness / chroma targets (CIELAB L*, C*) and accent strength per goal.
GOALS = {
    "balanced": {"wall_L": (72, 86), "wall_C": (6, 16), "accent_C": (30, 45)},
    "calm":     {"wall_L": (76, 88), "wall_C": (5, 13), "accent_C": (22, 35)},
    "cosy":     {"wall_L": (56, 72), "wall_C": (10, 22), "accent_C": (30, 45)},
    "bright":   {"wall_L": (84, 93), "wall_C": (3, 10), "accent_C": (35, 55)},
    "bold":     {"wall_L": (32, 55), "wall_C": (18, 36), "accent_C": (35, 55)},
}

# name -> (wall hue offset, accent hue offset) from the base hue, in degrees.
SCHEMES = {
    "tonal":               (0, 180),
    "analogous-warm":      (30, 210),
    "analogous-cool":      (-30, 150),
    "complementary":       (180, 0),
    "split-complementary": (150, 210),
    "triadic":             (120, 240),
}
NEUTRAL_SCHEMES = {
    "warm-neutral": 75,    # greige / cream family
    "cool-neutral": 250,   # soft grey with a blue lean
}

# Schemes that suit each interior style (from the LLM's style read). Matching
# palettes get a small score bonus; nothing is excluded, the checks still decide.
STYLE_SCHEMES = {
    "minimalist":         {"warm-neutral", "cool-neutral", "tonal"},
    "scandinavian":       {"warm-neutral", "cool-neutral", "tonal", "analogous-cool"},
    "japandi":            {"warm-neutral", "tonal", "analogous-warm"},
    "modern":             {"cool-neutral", "warm-neutral", "complementary"},
    "contemporary":       {"cool-neutral", "warm-neutral", "split-complementary", "complementary"},
    "industrial":         {"cool-neutral", "tonal", "complementary"},
    "mid-century modern": {"complementary", "triadic", "analogous-warm"},
    "traditional":        {"tonal", "analogous-warm", "warm-neutral"},
    "heritage":           {"tonal", "analogous-warm", "analogous-cool"},
    "bohemian":           {"analogous-warm", "triadic", "split-complementary"},
    "rustic":             {"analogous-warm", "tonal", "warm-neutral"},
    "coastal":            {"analogous-cool", "complementary", "cool-neutral"},
    "luxury":             {"tonal", "warm-neutral", "complementary"},
    "eclectic":           {"triadic", "split-complementary", "complementary"},
}
STYLE_BONUS = 0.06
# A wall within this many degrees of a hue the user's mood named counts as a match.
MOOD_HUE_TOLERANCE = 25
MOOD_BONUS = 0.3  # an explicit request outweighs the furnishings-harmony preference

# Hue offsets from the wall's hue that read as harmonious (colour-wheel relationships).
_HARMONIC_ANGLES = np.array([0, 30, 120, 150, 180, 210, 240, 330])


@dataclass
class Check:
    rule: str
    passed: bool
    detail: str
    hard: bool = True      # a failing hard check drops the palette

    def to_dict(self) -> dict:
        return {"rule": self.rule, "passed": bool(self.passed), "detail": self.detail, "hard": self.hard}


@dataclass
class Palette:
    scheme: str
    roles: dict                         # dominant / secondary / accent / ceiling
    checks: list[Check] = field(default_factory=list)
    score: float = 0.0
    facts: list[str] = field(default_factory=list)   # grounded reasons, fed to the LLM later
    accent_options: list = field(default_factory=list, repr=False)  # fallbacks if accent is taken

    @property
    def wall_hex(self) -> str:
        return self.roles["dominant"]["hex"]

    def to_dict(self) -> dict:
        return {"scheme": self.scheme, "score": round(self.score, 3), "roles": self.roles,
                "checks": [c.to_dict() for c in self.checks], "facts": self.facts}


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _fixed_colours(analysis: dict) -> list[dict]:
    """Regions the user isn't repainting, biggest first, with their LCh."""
    out = []
    for group, r in analysis["regions"].items():
        if group in PAINTABLE_GROUPS or group in ("window", "decor"):
            continue
        L, C, H = lab_to_lch(np.array(r["lab"]))
        out.append({"group": group, "hex": r["hex"], "lab": r["lab"], "area": r["area"],
                    "L": float(L), "C": float(C), "H": float(H)})
    return sorted(out, key=lambda f: -f["area"])


def _base_hue(fixed: list[dict], lighting: dict) -> tuple[float, str]:
    """Hue the palette is built around, and why."""
    chromatic = [f for f in fixed if f["C"] >= 12 and f["area"] >= 0.02]
    if chromatic:
        f = max(chromatic, key=lambda f: f["C"] * np.sqrt(f["area"]))
        return f["H"], f"built around your {f['group']} ({f['hex']})"
    # All-neutral room: choose a hue family from the light.
    if lighting["temperature"] == "cool":
        return 70.0, "room has neutral furnishings and cool light, so a warm base"
    return 70.0 if lighting["temperature"] == "warm" else 200.0, "room has neutral furnishings"


def _wall_target(goal: dict, lighting: dict, hue: float, neutral: bool) -> np.ndarray:
    L = float(np.mean(goal["wall_L"]))
    C = float(np.mean(goal["wall_C"]))
    if neutral:
        C = min(C, 6.0)
    # Dim rooms need lighter walls to bounce light; very bright rooms can go a touch deeper.
    if lighting["brightness"] == "dim":
        L = min(L + 5, goal["wall_L"][1] + 4)
    elif lighting["brightness"] == "bright":
        L = max(L - 2, goal["wall_L"][0])
    # Cool light greys out cool colours; lean the hue toward yellow (~80deg) a little.
    if lighting["temperature"] == "cool" and _hue_dist(hue, 80) > 20:
        hue = hue + np.sign(((80 - hue + 540) % 360) - 180) * 10
    return lch_to_lab(np.array([L, C, hue % 360]))


def _accent_target(goal: dict, wall_L: float, hue: float) -> np.ndarray:
    C = float(np.mean(goal["accent_C"]))
    # Accent must stand off the wall in lightness too: darker on light walls, lighter on dark.
    L = 45.0 if wall_L >= 60 else 72.0
    return lch_to_lab(np.array([L, C, hue % 360]))


def _ceiling_target(wall_lab: np.ndarray) -> np.ndarray:
    """Ceiling/trim white tinted toward the wall's hue, so the white doesn't look stark."""
    _, C, H = lab_to_lch(wall_lab)
    return lch_to_lab(np.array([95.0, min(3.0, C * 0.25), H]))


def _role(paint, source: str, share: float, use: str) -> dict:
    return {**paint.to_dict(), "source": source, "share": share, "use": use}


def _harmony_penalty(wall_H: float, wall_C: float, fixed: list[dict]) -> float:
    """How far chromatic fixed colours sit from harmonious angles to the wall.
    Neutral walls go with anything, so the penalty scales with the wall's chroma."""
    pen, total = 0.0, 0.0
    for f in fixed:
        if f["C"] < 12:
            continue
        rel = (f["H"] - wall_H) % 360
        off = np.min(np.minimum(np.abs(rel - _HARMONIC_ANGLES), 360 - np.abs(rel - _HARMONIC_ANGLES)))
        w = f["area"] * min(f["C"], 40) / 40
        pen += w * max(0.0, off - 15) / 45
        total += w
    return 0.0 if total == 0 else min(1.0, pen / total) * min(1.0, wall_C / 15)


def check_palette(p: Palette, fixed: list[dict], goal_name: str, lighting: dict,
                  wall_locked: bool = False, l_shift: float = 0.0) -> None:
    """Run every rule on the snapped paint colours and compute a score.
    With wall_locked, rules about the wall only warn: the user chose that wall.
    l_shift moves the goal's target lightness when the user asked for lighter/darker."""
    goal = GOALS[goal_name]
    wall = p.roles["dominant"]
    accent = p.roles["accent"]
    wall_lab, accent_lab = np.array(wall["lab"]), np.array(accent["lab"])
    wL, wC, wH = lab_to_lch(wall_lab)
    checks = []

    if fixed:
        anchor = fixed[0]
        de = delta_e(wall_lab, anchor["lab"])
        checks.append(Check("wall-distinct-from-anchor", de >= 8,
                            f"wall vs {anchor['group']}: dE {de:.1f} (need >= 8 so they don't blend)"))

    de_acc = delta_e(accent_lab, wall_lab)
    checks.append(Check("accent-stands-out", de_acc >= 20,
                        f"accent vs wall: dE {de_acc:.1f} (need >= 20 to read as an accent)"))

    lo, hi = goal["wall_L"][0] + l_shift, goal["wall_L"][1] + l_shift
    checks.append(Check("wall-lightness-fits-goal", lo - 8 <= wL <= hi + 8,
                        f"wall L* {wL:.0f}, '{goal_name}' targets {lo:.0f}-{hi:.0f}", hard=False))

    max_c = goal["wall_C"][1] + 8
    checks.append(Check("wall-not-garish", wC <= max_c,
                        f"wall chroma {wC:.0f} (max {max_c} for '{goal_name}')"))

    if lighting["brightness"] == "dim" and goal_name not in ("cosy", "bold"):
        checks.append(Check("dim-room-light-wall", wall["lrv"] >= 50,
                            f"room is dim; wall LRV {wall['lrv']:.0f} (need >= 50 to bounce light)"))

    pen = _harmony_penalty(wH, wC, fixed)
    checks.append(Check("harmony-with-furnishings", pen < 0.5,
                        f"hue clash penalty {pen:.2f} with existing furnishings", hard=False))

    match_de = max(wall["match_delta_e"], accent["match_delta_e"])
    checks.append(Check("real-paint-close", match_de <= 5,
                        f"nearest real paints within dE {match_de:.1f} of the target", hard=False))

    if wall_locked:
        for c in checks:
            if c.rule.startswith(("wall-", "dim-room")):
                c.hard = False
    p.checks = checks
    # Score: harmony matters most, then how close the paints came to the target and goal fit.
    target_L = float(np.mean(goal["wall_L"])) + l_shift
    p.score = (1.0 - pen) * 0.6 \
        + max(0.0, 1 - match_de / 6) * 0.2 \
        + max(0.0, 1 - abs(wL - target_L) / 25) * 0.2 \
        - 0.5 * sum(not c.passed for c in checks if c.hard)


def _adjust_note(adjust: dict) -> str:
    words = {"warmth": ("warmer", "cooler"), "lightness": ("lighter", "darker"),
             "saturation": ("more vivid", "more muted")}
    parts = [f"{words[k][0] if v > 0 else words[k][1]} x{abs(v):g}"
             for k, v in adjust.items() if v and k in words]
    return "adjusted by you: " + ", ".join(parts) if parts else ""


def suggest_palettes(analysis: dict, goal: str = "balanced", n: int = 5,
                     brands: list[str] | None = None, style: dict | None = None,
                     locks: dict[str, str] | None = None, adjust: dict | None = None,
                     mood: dict | None = None) -> dict:
    """Return the n best palettes for a room, plus the reasoning context.

    style:  optional LLM style read (app.llm.analyze_style); schemes that suit the
            room's style get a small ranking bonus.
    locks:  {"dominant": "#hex"} and/or {"accent": "#hex"}: keep that colour (snapped
            to the nearest real paint) and rebuild the rest of the palette around it.
            A locked wall is the user's call, so wall checks become warnings only.
    adjust: {"warmth", "lightness", "saturation"} in steps (about -3..+3), applied
            to the generated walls (all three) and accents (warmth, saturation).
    mood:   a mood reading (app.pipeline.mood): its hue families add candidates built
            on those hues and favour walls that match them; "neutral" favours the
            neutral schemes. Goal and adjust are applied by the caller.
    """
    if goal not in GOALS:
        raise ValueError(f"goal must be one of {list(GOALS)}")
    locks = {k: v for k, v in (locks or {}).items() if v}
    if set(locks) - {"dominant", "accent"}:
        raise ValueError("only 'dominant' and 'accent' can be locked")
    adjust = {k: float(v) for k, v in (adjust or {}).items() if v}
    book = load_paints()
    g, lighting = GOALS[goal], analysis["lighting"]
    fixed = _fixed_colours(analysis)
    base_H, base_why = _base_hue(fixed, lighting)
    anchor = fixed[0] if fixed else None

    locked = {role: book.nearest(rgb_to_lab(hex_to_rgb(hx)), brands=brands)[0]
              for role, hx in locks.items()}
    lock_wall = locked.get("dominant")
    if lock_wall is not None:
        # Build accents from the locked wall's own hue, not the room's base hue.
        lw_H = float(lab_to_lch(np.array(lock_wall.lab))[2])
        plans = [(name, lw_H, lw_H + (a - w), False) for name, (w, a) in SCHEMES.items()]
        plans += [(name, lw_H, base_H + 180, True) for name in NEUTRAL_SCHEMES]
        variants = {0: ""}
    else:
        plans = [(name, base_H + w, base_H + a, False) for name, (w, a) in SCHEMES.items()]
        plans += [(name, h, base_H + 180, True) for name, h in NEUTRAL_SCHEMES.items()]
        # Each scheme also tries a deeper and a lighter version of the wall, so a room
        # whose big furniture blocks the default shade still gets options in that scheme.
        variants = {0: "", -10: "deeper", 7: "lighter"}
    mood_hues = [(h["word"], float(h["hue"])) for h in (mood or {}).get("hues", [])]
    if lock_wall is None:
        plans += [("mood-match", h, h + 180, False) for _, h in mood_hues]
    acc_adjust = {k: v for k, v in adjust.items() if k in ("warmth", "saturation")}
    l_shift = STEP_L * adjust.get("lightness", 0.0)

    candidates: list[Palette] = []
    for (scheme, wall_h, acc_h, neutral), (dL, variant) in (
            (plan, v) for plan in plans for v in variants.items()):
        if lock_wall is not None:
            wall_p = lock_wall
        else:
            wall_t = _wall_target(g, lighting, wall_h, neutral)
            wall_t[0] = np.clip(wall_t[0] + dL, 25, 94)
            if adjust:
                wall_t = adjust_lab(wall_t, **adjust)
            wall_p = book.nearest(wall_t, brands=brands)[0]
        if "accent" in locked:
            acc_opts = [locked["accent"]]
        else:
            # Main accent first; fallbacks step around the colour wheel and up in lightness,
            # so a palette can swap to a distinct accent if a better palette took this one.
            acc_targets = [_accent_target(g, wall_p.lab[0], acc_h + dh) for dh in (0, 25, -25)]
            acc_targets.append(_accent_target(g, wall_p.lab[0], acc_h) + np.array([12, 0, 0]))
            if acc_adjust:
                acc_targets = [adjust_lab(t, **acc_adjust) for t in acc_targets]
            acc_opts = [book.nearest(t, brands=brands)[0] for t in acc_targets]
        acc_p = acc_opts[0]
        ceil_p = book.nearest(_ceiling_target(np.array(wall_p.lab)), brands=brands)[0]

        roles = {
            "dominant": _role(wall_p, "new paint", 0.6, "walls"),
            "secondary": ({"hex": anchor["hex"], "lab": anchor["lab"],
                           "lrv": round(light_reflectance_value(hex_to_rgb(anchor["hex"])), 1),
                           "source": f"existing {anchor['group']}", "share": 0.3,
                           "use": f"keep your {anchor['group']}; also good for rugs, curtains, throws"}
                          if anchor else
                          {**_role(book.nearest(lch_to_lab(np.array([60, 12, base_H])))[0],
                                   "new", 0.3, "large furniture, rugs, curtains")}),
            "accent": _role(acc_p, "locked by you" if "accent" in locked else "new", 0.1,
                            "cushions, art, a feature chair or door"),
            "ceiling": _role(ceil_p, "new paint", 0.0, "ceiling and trim"),
        }
        if lock_wall is not None:
            roles["dominant"]["source"] = "locked by you"
        wL, wC, wH = lab_to_lch(np.array(wall_p.lab))
        facts = [f"scheme: {scheme}" + (f" ({variant} variant)" if variant else ""),
                 f"base hue {base_H:.0f}deg, {base_why}",
                 f"wall hue {wH:.0f}deg, L* {wL:.0f}, chroma {wC:.0f}, LRV {wall_p.lrv:.0f}",
                 f"room light: {lighting['brightness']}, {lighting['temperature']}",
                 f"goal: {goal}"]
        facts += [f"{role} locked by you" for role in locked]
        if adjust and _adjust_note(adjust):
            facts.append(_adjust_note(adjust))
        p = Palette(scheme=scheme, roles=roles, facts=facts, accent_options=acc_opts)
        check_palette(p, fixed, goal, lighting, wall_locked=lock_wall is not None, l_shift=l_shift)
        # Small preference for the default shade over its variants.
        p.score -= 0.03 if variant else 0.0
        if style and scheme in STYLE_SCHEMES.get(style.get("style", ""), ()):
            p.score += STYLE_BONUS
            p.facts.append(f"suits the room's {style['style']} style")
        if mood_hues and wC >= 4:
            near = [w for w, h in mood_hues if _hue_dist(wH, h) <= MOOD_HUE_TOLERANCE]
            if near:
                p.score += MOOD_BONUS
                p.facts.append(f"matches your mood: {near[0]}")
        if (mood or {}).get("neutral") and scheme in NEUTRAL_SCHEMES:
            p.score += MOOD_BONUS
            p.facts.append("matches your mood: neutral walls")
        candidates.append(p)

    # Best first. Skip palettes whose wall is nearly the same as a better one, and
    # swap in an alternative accent if a better palette already uses that accent.
    # With a locked wall every palette shares it, so variety has to come from the accent.
    chosen: list[Palette] = []
    for p in sorted(candidates, key=lambda p: -p.score):
        if any(not c.passed for c in p.checks if c.hard):
            continue
        if lock_wall is None and any(delta_e(np.array(p.roles["dominant"]["lab"]),
                                             np.array(q.roles["dominant"]["lab"])) < 8 for q in chosen):
            continue
        if lock_wall is not None and "accent" not in locked and any(
                delta_e(np.array(p.roles["accent"]["lab"]), np.array(q.roles["accent"]["lab"])) < 8
                for q in chosen):
            continue
        used = [np.array(q.roles["accent"]["lab"]) for q in chosen]
        for opt in p.accent_options:
            if all(delta_e(np.array(opt.lab), u) >= 6 for u in used):
                if opt.hex != p.roles["accent"]["hex"]:
                    p.roles["accent"] = _role(opt, "new", 0.1, p.roles["accent"]["use"])
                    check_palette(p, fixed, goal, lighting, wall_locked=lock_wall is not None, l_shift=l_shift)
                break
        if any(not c.passed for c in p.checks if c.hard):
            continue
        chosen.append(p)
        if len(chosen) == n:
            break

    return {
        "goal": goal,
        "style": style.get("style") if style else None,
        "locks": {role: p.to_dict() for role, p in locked.items()},
        "mood": mood,
        "adjust": adjust,
        "base_hue": round(float(base_H), 1),
        "base_hue_reason": base_why,
        "anchor": anchor,
        "palettes": [p.to_dict() for p in chosen],
        "candidates_considered": len(candidates),
        "rejected": [{"scheme": p.scheme, "wall": p.wall_hex,
                      "failed": [c.rule for c in p.checks if c.hard and not c.passed]}
                     for p in candidates if any(c.hard and not c.passed for c in p.checks)],
    }
