"""Turn a free-text mood ("calm coastal morning", "moody library") into the
controls the palette pipeline already has: a goal, warmth/lightness/saturation
steps, and target hue families.

The vocabulary below runs offline and is deterministic, so the same words always
give the same palettes and the reading can be shown to the user. A language model
is consulted only when no known word matches (see app/llm.py, parse_mood).
"""

import re
from dataclasses import asdict, dataclass, field

from app.pipeline.colorutils import hex_to_rgb, lab_to_lch, rgb_to_lab

# Colour words -> a reference colour. Their hue angle (CIELAB LCh) is measured
# from the hex, not guessed, so "sage" means the hue sage actually has.
HUE_WORDS: dict[str, str] = {
    "red": "#B03A2E", "terracotta": "#C46A4A", "rust": "#B5562F", "coral": "#E07B6A",
    "orange": "#D9822B", "peach": "#E8B49A", "apricot": "#E9A86B", "sand": "#D8C3A0",
    "beige": "#D9C8AE", "ochre": "#C9A23F", "mustard": "#C9A23F", "yellow": "#E3C565",
    "butter": "#EED9A0", "olive": "#8A8A4E", "sage": "#A8B5A2", "green": "#5E8C61",
    "forest": "#2F5D46", "emerald": "#2E7D5B", "mint": "#BFE3D0", "teal": "#2F7F80",
    "aqua": "#8FCFD0", "ocean": "#4F86A8", "sea": "#4F86A8", "sky": "#9CC3E0",
    "blue": "#5B7DB1", "navy": "#2F3E4E", "indigo": "#3D4A7A", "lavender": "#B7A9CF",
    "lilac": "#C3B1D6", "purple": "#6E4F8A", "plum": "#6B3F5A", "blush": "#E3B7B5",
    "pink": "#D98BA0", "rose": "#C98E95", "burgundy": "#6D2A33", "wine": "#6D2A33",
}

# Spelling variants and plurals people type.
ALIASES = {
    "cozy": "cosy", "colour": "color", "grey": "gray", "greens": "green", "blues": "blue",
    "pinks": "pink", "neutrals": "neutral", "earth": "earthy", "pastels": "pastel",
    "beachy": "beach", "oceanic": "ocean", "sunlit": "sunny", "tranquil": "calm",
    "serene": "calm", "peaceful": "calm", "relaxing": "calm", "relaxed": "calm",
    "soothing": "calm", "zen": "calm", "restful": "calm", "quiet": "calm",
    "snug": "cosy", "hygge": "cosy", "intimate": "cosy", "inviting": "cosy",
    "airy": "bright", "open": "bright", "spacious": "bright", "clean": "bright",
    "dramatic": "moody", "dark": "moody", "cocooning": "moody", "mysterious": "moody",
    "energetic": "vibrant", "lively": "vibrant", "playful": "vibrant", "fun": "vibrant",
    "cheerful": "vibrant", "joyful": "vibrant", "happy": "vibrant", "bold": "vibrant",
    "subtle": "soft", "gentle": "soft", "dusty": "soft", "faded": "soft", "muted": "soft",
    "elegant": "luxurious", "luxury": "luxurious", "sophisticated": "luxurious",
    "classy": "luxurious", "glamorous": "luxurious", "opulent": "luxurious",
    "natural": "earthy", "organic": "earthy", "rustic": "earthy",
    "minimal": "minimalist", "simple": "minimalist", "calming": "calm",
    "cold": "cool", "chilly": "cool", "icy": "cool", "cosier": "cosy", "cozier": "cosy",
    "warmer": "warm", "cooler": "cool", "lighter": "light", "darker": "deep",
    "rainy": "rain", "cloudy": "rain", "stormy": "rain", "seaside": "coastal",
    "plants": "botanical", "greenery": "botanical", "woodland": "forest", "woods": "forest",
}


@dataclass
class Effect:
    goal: str | None = None
    warmth: float = 0
    lightness: float = 0
    saturation: float = 0
    hues: tuple[str, ...] = ()
    neutral: bool = False


# Mood and theme words -> what they do. Values are steps on the existing knobs.
MOOD_WORDS: dict[str, Effect] = {
    # feelings
    "calm": Effect(goal="calm", saturation=-1),
    "cosy": Effect(goal="cosy", warmth=1),
    "warm": Effect(warmth=2),
    "sunny": Effect(goal="bright", warmth=1, hues=("butter",)),
    "bright": Effect(goal="bright", lightness=1),
    "fresh": Effect(goal="bright", warmth=-1),
    "cool": Effect(warmth=-2),
    "crisp": Effect(warmth=-1, lightness=1),
    "moody": Effect(goal="bold", lightness=-2, saturation=-1),
    "vibrant": Effect(goal="bold", saturation=2),
    "soft": Effect(saturation=-2),
    "pastel": Effect(goal="bright", lightness=1, saturation=-1),
    "light": Effect(lightness=1),
    "deep": Effect(lightness=-2),
    "rich": Effect(lightness=-1, saturation=1),
    "luxurious": Effect(lightness=-1, saturation=-1),
    "romantic": Effect(goal="calm", warmth=1, hues=("blush",)),
    "focused": Effect(goal="calm", warmth=-1, saturation=-1),
    "earthy": Effect(warmth=1, saturation=-1, hues=("terracotta", "olive")),
    "neutral": Effect(neutral=True, saturation=-1),
    "minimalist": Effect(goal="bright", neutral=True, saturation=-1),
    # places and themes
    "coastal": Effect(goal="bright", warmth=-1, hues=("aqua", "sand")),
    "beach": Effect(goal="bright", warmth=-1, hues=("aqua", "sand")),
    "nordic": Effect(goal="bright", neutral=True, warmth=-1),
    "scandinavian": Effect(goal="bright", neutral=True, warmth=-1),
    "japandi": Effect(goal="calm", neutral=True, warmth=1, saturation=-1),
    "forest": Effect(goal="calm", hues=("forest", "sage")),
    "botanical": Effect(goal="calm", hues=("sage", "green")),
    "garden": Effect(goal="calm", hues=("sage", "green")),
    "jungle": Effect(goal="bold", hues=("emerald",)),
    "desert": Effect(warmth=2, saturation=-1, hues=("terracotta", "sand")),
    "tuscan": Effect(goal="cosy", warmth=2, hues=("terracotta", "ochre")),
    "mediterranean": Effect(goal="bright", hues=("terracotta", "ocean")),
    "sunset": Effect(goal="cosy", warmth=2, hues=("coral", "apricot")),
    "library": Effect(goal="bold", lightness=-2, hues=("forest", "burgundy")),
    "cabin": Effect(goal="cosy", warmth=2, hues=("rust", "olive")),
    "spa": Effect(goal="calm", warmth=-1, hues=("aqua", "sage")),
    "morning": Effect(goal="bright", lightness=1),
    "evening": Effect(goal="cosy", lightness=-1, warmth=1),
    "nursery": Effect(goal="calm", lightness=1, saturation=-1),
    "office": Effect(goal="calm", warmth=-1),
    "study": Effect(goal="calm", warmth=-1),
    "kids": Effect(goal="bold", saturation=1),
    "retro": Effect(saturation=1, hues=("mustard", "teal")),
    "industrial": Effect(neutral=True, warmth=-1, lightness=-1),
    "rain": Effect(goal="calm", warmth=-1, saturation=-1, hues=("sky",)),
}

# Words that name a feeling outweigh place and theme words when they disagree on the
# goal: "calm coastal morning" is calm first, coastal second.
FEELING_WORDS = {"calm", "cosy", "warm", "sunny", "bright", "fresh", "cool", "crisp", "moody",
                 "vibrant", "soft", "pastel", "light", "deep", "rich", "luxurious", "romantic",
                 "focused", "earthy", "neutral", "minimalist"}

NEGATORS = {"not", "no", "without", "less", "never"}
# "not too dark", "not so cold": the negation carries over these.
INTENSIFIERS = {"too", "so", "very", "overly", "that", "really", "super", "quite"}
MAX_STEP = 3


@dataclass
class MoodReading:
    text: str
    goal: str | None = None
    adjust: dict[str, float] = field(default_factory=dict)
    hues: list[dict] = field(default_factory=list)   # {word, hex, hue}
    neutral: bool = False
    matched: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    source: str = "keywords"

    @property
    def empty(self) -> bool:
        return not (self.goal or self.adjust or self.hues or self.neutral)

    def summary(self) -> str:
        """Plain-language read-back, e.g. 'calm · cooler · softer · sage, aqua'."""
        parts = []
        if self.goal:
            parts.append(self.goal)
        words = {"warmth": ("warmer", "cooler"), "lightness": ("lighter", "darker"),
                 "saturation": ("more vivid", "softer")}
        for k, (up, down) in words.items():
            v = self.adjust.get(k, 0)
            if v:
                parts.append(up if v > 0 else down)
        if self.neutral:
            parts.append("neutral walls")
        if self.hues:
            parts.append(", ".join(h["word"] for h in self.hues))
        return " · ".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = self.summary()
        return d


def hue_of(word: str) -> dict:
    hx = HUE_WORDS[word]
    _, _, h = lab_to_lch(rgb_to_lab(hex_to_rgb(hx)))
    return {"word": word, "hex": hx, "hue": round(float(h), 1)}


def _tokens(text: str) -> list[str]:
    return [ALIASES.get(t, t) for t in re.findall(r"[a-z]+", text.lower())]


def parse_mood(text: str) -> MoodReading:
    """Deterministic reading of a mood phrase. Unknown words are listed, not guessed."""
    text = text.strip()[:200]
    reading = MoodReading(text=text)
    goals: dict[str, int] = {}
    adj = {"warmth": 0.0, "lightness": 0.0, "saturation": 0.0}
    named: list[str] = []      # colours the user typed, in their order
    themed: list[str] = []     # colours implied by moods and themes

    toks = _tokens(text)
    negate = False
    for t in toks:
        if t in NEGATORS:
            negate = True
            continue
        if t in INTENSIFIERS:
            continue
        sign = -1 if negate else 1
        negate = False
        if t in MOOD_WORDS:
            e = MOOD_WORDS[t]
            reading.matched.append(t)
            if e.goal and sign > 0:
                goals[e.goal] = goals.get(e.goal, 0) + (2 if t in FEELING_WORDS else 1)
            adj["warmth"] += sign * e.warmth
            adj["lightness"] += sign * e.lightness
            adj["saturation"] += sign * e.saturation
            if sign > 0:
                themed += list(e.hues)
                reading.neutral = reading.neutral or e.neutral
        elif t in HUE_WORDS:
            reading.matched.append(t)
            if sign > 0:
                named.append(t)   # a colour the user named outranks a theme's
        elif len(t) > 2 and t not in {"and", "the", "with", "for", "room", "feel", "feeling",
                                      "vibe", "vibes", "like", "very", "bit", "but", "that",
                                      "color", "colors", "walls", "wall", "my", "our", "want"}:
            reading.ignored.append(t)

    if goals:
        # most-mentioned goal; ties go to the first one mentioned
        reading.goal = max(goals, key=lambda g: (goals[g], -list(goals).index(g)))
    reading.adjust = {k: max(-MAX_STEP, min(MAX_STEP, v)) for k, v in adj.items() if v}
    reading.matched = list(dict.fromkeys(reading.matched))
    seen = []
    for w in named + themed:
        if w not in seen:
            seen.append(w)
    reading.hues = [hue_of(w) for w in seen[:2]]
    if reading.hues:
        reading.neutral = False   # a named colour beats "neutral"
    return reading


def reading_from_fields(text: str, goal: str | None, warmth: float, lightness: float,
                        saturation: float, hue_words: list[str], neutral: bool,
                        source: str) -> MoodReading:
    """Build a reading from structured fields (used for the LLM fallback)."""
    clamp = lambda v: max(-MAX_STEP, min(MAX_STEP, round(float(v))))
    adjust = {k: clamp(v) for k, v in (("warmth", warmth), ("lightness", lightness),
                                        ("saturation", saturation)) if clamp(v)}
    hues = [hue_of(w) for w in hue_words if w in HUE_WORDS][:2]
    return MoodReading(text=text, goal=goal, adjust=adjust, hues=hues,
                       neutral=neutral and not hues, matched=[], source=source)
