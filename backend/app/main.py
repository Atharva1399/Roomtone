"""HTTP API for the room-to-palette suggester.

Run (from the backend folder, with `conda activate palette-env`):
    uvicorn app.main:app --reload --port 8000
Interactive docs: http://localhost:8000/docs

Flow:
    POST /api/rooms                      upload a photo -> room_id + colour analysis
    POST /api/rooms/{id}/style           the LLM's read of the room's style (optional, cached)
    POST /api/rooms/{id}/palettes        suggested palettes (real paints, 60-30-10, checks,
                                         optional explanations)
    POST /api/rooms/{id}/preview         repainted photo for any wall/ceiling colour (fast)

Refine:
    palettes {locks, adjust}             keep a colour / nudge suggestions warmer, lighter...
    POST /api/colors/adjust              nudge one colour, get the nearest real paints
    POST /api/rooms/{id}/mask            click to exclude/include an object from the paint area
    POST /api/rooms/{id}/mask/undo|reset
    GET  /api/rooms/{id}/mask            photo with the paint area highlighted

LLM features need HF_TOKEN (Hugging Face) or ANTHROPIC_API_KEY; without either, everything
else works and explanations come from templates.
"""

import io
import re
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

from app import config
from app.llm import (LLMUnavailable, analyze_style, explain_palettes, llm_configured,
                     parse_mood_llm, provider_label, template_explanation)
from app.pipeline import palette_ranker
from app.pipeline.mood import parse_mood
from app.pipeline.adjust import adjust_lab
from app.pipeline.colorutils import hex_to_rgb, lab_to_rgb, rgb_to_hex, rgb_to_lab
from app.pipeline.paints import BRAND_NAMES, load_paints
from app.pipeline.recolor import apply_paints, to_image
from app.pipeline.segmentation import PAINTABLE_GROUPS, _load_model, load_image, overlay
from app.pipeline.suggest import GOALS, suggest_palettes
from app.rooms import Room, store

# Refuse absurd pixel counts before decoding (decompression-bomb guard).
Image.MAX_IMAGE_PIXELS = 50_000_000

_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Load the model and paint books at startup, not on the first user's request."""
    _load_model()
    load_paints()
    yield


app = FastAPI(title="AI Interior Colour Palette Suggester", version="0.1.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["*"],
                   expose_headers=["X-Paint-Delta-E"])


# ---------- request / response models ----------

def _normalise_hex(v: str | None) -> str | None:
    if v is None:
        return None
    if not _HEX.match(v):
        raise ValueError("colour must be a 6-digit hex like #A8B5A2")
    return "#" + v.lstrip("#").upper()


Goal = Literal["balanced", "calm", "cosy", "bright", "bold"]
assert set(Goal.__args__) == set(GOALS), "keep Goal in sync with suggest.GOALS"


class Adjust(BaseModel):
    """Steps, about -3..+3. Positive = warmer / lighter / more vivid."""
    warmth: float = Field(0, ge=-5, le=5)
    lightness: float = Field(0, ge=-5, le=5)
    saturation: float = Field(0, ge=-5, le=5)


class Locks(BaseModel):
    dominant: str | None = Field(None, description="keep this wall colour", examples=["#2F3E4E"])
    accent: str | None = Field(None, description="keep this accent colour")

    @field_validator("dominant", "accent")
    @classmethod
    def _check_hex(cls, v: str | None) -> str | None:
        return _normalise_hex(v)


class PaletteRequest(BaseModel):
    goal: Goal | None = Field(None, description="omit to use the style analysis' suggestion (else balanced)")
    n: int = Field(4, ge=1, le=8)
    brands: list[str] | None = Field(None, description=f"limit to: {list(BRAND_NAMES)}")
    use_style: bool = Field(True, description="let the room's style (if analysed) steer the ranking")
    explain: bool = Field(False, description="add a plain-language explanation per palette")
    explain_with: Literal["auto", "template"] = Field(
        "auto", description="template: instant explanations from the facts; auto: the LLM when one is available")
    locks: Locks | None = Field(None, description="keep a wall and/or accent colour, rebuild the rest")
    adjust: Adjust | None = Field(None, description="nudge all suggestions warmer/lighter/more vivid...")
    mood: str | None = Field(None, max_length=200, description="free text, e.g. 'calm coastal morning'",
                             examples=["moody library"])


class AdjustRequest(Adjust):
    hex: str = Field(..., examples=["#C8BFB5"])
    k: int = Field(3, ge=1, le=10, description="how many nearest real paints to return")
    brand: str | None = None

    @field_validator("hex")
    @classmethod
    def _check_hex(cls, v: str) -> str:
        return _normalise_hex(v)


class MaskEditRequest(BaseModel):
    group: Literal["wall", "ceiling"] = "wall"
    x: int = Field(..., ge=0, description="pixel column in the room image (see width)")
    y: int = Field(..., ge=0, description="pixel row in the room image (see height)")
    mode: Literal["exclude", "include"] = "exclude"
    level: Literal["object", "part"] = Field(
        "object", description="object: whole thing under the click (all the panelling); part: just one piece")


class MaskGroupRequest(BaseModel):
    group: Literal["wall", "ceiling"] = "wall"


class PreviewRequest(BaseModel):
    wall: str | None = Field(None, examples=["#A8B5A2"])
    ceiling: str | None = None
    format: Literal["jpeg", "png"] = "jpeg"
    quality: int = Field(88, ge=40, le=100)

    @field_validator("wall", "ceiling")
    @classmethod
    def _check_hex(cls, v: str | None) -> str | None:
        return _normalise_hex(v)


# ---------- helpers ----------

def _room_or_404(room_id: str) -> Room:
    room = store.get(room_id)
    if room is None:
        raise HTTPException(404, "room not found (it may have expired; upload the photo again)")
    return room


def _image_response(img: Image.Image, fmt: str = "jpeg", quality: int = 88, headers=None) -> Response:
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=quality)
    return Response(buf.getvalue(), media_type=f"image/{fmt}", headers=headers or {})


def _room_summary(room: Room) -> dict:
    return {
        "room_id": room.id,
        "width": room.image.width,
        "height": room.image.height,
        "paintable": {g: round(room.seg.fraction(g), 4) for g in PAINTABLE_GROUPS},
        "paint_ceiling_by_default": room.paint_ceiling,
        "analysis": room.analysis,
    }


# ---------- endpoints ----------

@app.get("/api/health")
def health() -> dict:
    segmentation = ("V2 roomtone-surface-b4" if config.SEGMENTATION_MODEL == str(config.V2_SEGMENTATION_MODEL)
                    else "V1 " + config.SEGMENTATION_MODEL.split("/")[-1])
    ranking = "V1 rules + V2 roomtone-palette" if palette_ranker.available() else "V1 rules"
    return {"status": "ok", "gpu": torch.cuda.is_available(), "llm": provider_label(),
            "segmentation": segmentation, "palette_ranking": ranking,
            "paints": len(load_paints()), "rooms_cached": len(store)}


@app.post("/api/rooms", status_code=201)
def create_room(file: UploadFile = File(...)) -> dict:
    """Upload a room photo. Segments it and measures its current colours."""
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"photo larger than {config.MAX_UPLOAD_BYTES // 2**20} MB")
    try:
        image = load_image(io.BytesIO(data))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise HTTPException(400, "could not read that file as a photo (use JPEG or PNG)")
    room = store.create(image)
    return _room_summary(room)


@app.get("/api/rooms/{room_id}")
def get_room(room_id: str) -> dict:
    return _room_summary(_room_or_404(room_id))


@app.get("/api/rooms/{room_id}/original")
def get_original(room_id: str) -> Response:
    return _image_response(_room_or_404(room_id).image)


@app.get("/api/rooms/{room_id}/segmentation")
def get_segmentation(room_id: str) -> Response:
    room = _room_or_404(room_id)
    return _image_response(overlay(room.image, room.seg))


@app.post("/api/rooms/{room_id}/style")
def get_style(room_id: str, refresh: bool = False) -> dict:
    """The vision model reads the photo: room type, style, mood, what stays, a suggested goal.
    Cached per room; ~5-15s the first time."""
    room = _room_or_404(room_id)
    if room.style is None or refresh:
        try:
            room.style = analyze_style(room.image, room.analysis,
                                       still_wanted=lambda: store.is_latest(room.id))
        except LLMUnavailable as e:
            raise HTTPException(503, f"style analysis unavailable: {e}")
    return room.style


@app.post("/api/rooms/{room_id}/palettes")
def get_palettes(room_id: str, req: PaletteRequest) -> dict:
    """Suggested palettes, each with real paints for the 60-30-10 roles, the checks
    it passed, and the facts behind it. With explain=true each palette also gets a
    plain-language explanation (from the LLM, or a template if no LLM is available)."""
    room = _room_or_404(room_id)
    brands = [b for b in (req.brands or []) if b in BRAND_NAMES] or None
    style = room.style if req.use_style else None

    # A mood is read by the vocabulary first; the LLM is asked only when no word is known.
    reading = None
    if req.mood and req.mood.strip():
        reading = parse_mood(req.mood)
        if reading.empty and llm_configured():
            try:
                reading = parse_mood_llm(req.mood)
            except LLMUnavailable:
                pass

    mood_goal = reading.goal if reading else None
    goal = req.goal or mood_goal or (style or {}).get("suggested_goal") or "balanced"
    adjust = req.adjust.model_dump() if req.adjust else {}
    for k, v in (reading.adjust if reading else {}).items():
        adjust[k] = max(-3.0, min(3.0, adjust.get(k, 0) + v))

    result = suggest_palettes(
        room.analysis, goal=goal, n=req.n, brands=brands, style=style,
        locks=req.locks.model_dump(exclude_none=True) if req.locks else None,
        adjust=adjust or None, mood=reading.to_dict() if reading else None,
        ranker_context=room.ranker_context)
    result["paint_ceiling_by_default"] = room.paint_ceiling
    result["goal_source"] = ("request" if req.goal else "mood" if mood_goal
                             else "style" if style else "default")

    if req.explain and result["palettes"]:
        try:
            if req.explain_with == "template":
                raise LLMUnavailable("templates requested")
            # The style read informs the words even when it isn't steering the ranking.
            room.explain_seq += 1
            seq = room.explain_seq
            explanations = explain_palettes(
                result["palettes"], room.analysis, room.style, goal,
                # skip if the user has moved on: newer palettes or another room
                still_wanted=lambda: room.explain_seq == seq and store.is_latest(room.id))
            result["explanation_source"] = provider_label()
        except LLMUnavailable as e:
            explanations = [template_explanation(p, room.analysis, goal) for p in result["palettes"]]
            result["explanation_source"] = "template"
            result["explanation_note"] = str(e)
        for p, e in zip(result["palettes"], explanations):
            p["explanation"] = e
    return result


@app.post("/api/rooms/{room_id}/preview")
def preview(room_id: str, req: PreviewRequest) -> Response:
    """Repainted photo. The first call per surface prepares its lighting (~0.5s);
    after that each colour change only recombines cached layers."""
    room = _room_or_404(room_id)
    colors = {g: c for g, c in (("wall", req.wall), ("ceiling", req.ceiling)) if c}
    if not colors:
        raise HTTPException(422, "give at least one of: wall, ceiling")
    out, report = apply_paints(room.rgb, room.layers(colors), colors)
    # Per-surface verification travels in a header so the body stays a plain image.
    summary = ";".join(
        f"{g}={r['delta_e_typical_pixel']}" if r["applied"] else f"{g}=not-found"
        for g, r in report.items())
    return _image_response(to_image(out), req.format, req.quality,
                           headers={"X-Paint-Delta-E": summary})


@app.post("/api/rooms/{room_id}/mask")
def edit_mask(room_id: str, req: MaskEditRequest) -> dict:
    """Click to fix what gets painted: 'exclude' removes the object under the click
    (e.g. wood panelling labelled as wall), 'include' adds a missed patch.
    The first click per photo prepares the click model (~0.5s); later clicks ~50ms."""
    room = _room_or_404(room_id)
    if req.x >= room.image.width or req.y >= room.image.height:
        raise HTTPException(422, f"click outside the image ({room.image.width}x{room.image.height})")
    return room.edit_mask(req.group, req.x, req.y, req.mode, req.level)


@app.post("/api/rooms/{room_id}/mask/undo")
def undo_mask(room_id: str, req: MaskGroupRequest) -> dict:
    return _room_or_404(room_id).undo_mask(req.group)


@app.post("/api/rooms/{room_id}/mask/reset")
def reset_mask(room_id: str, req: MaskGroupRequest) -> dict:
    return _room_or_404(room_id).reset_mask(req.group)


@app.get("/api/rooms/{room_id}/mask")
def get_mask(room_id: str, group: Literal["wall", "ceiling"] = "wall") -> Response:
    """The photo with the area that will be painted tinted magenta, for checking edits."""
    room = _room_or_404(room_id)
    lay = room.layers([group])[group]
    out = room.rgb.copy()
    if lay is not None:
        out = out * (1 - 0.55 * lay.alpha) + np.array([1.0, 0.0, 0.8], np.float32) * 0.55 * lay.alpha
    return _image_response(to_image(out), headers={"X-Mask-Edits": str(room.mask_state(group)["edits"])})


@app.post("/api/colors/adjust")
def adjust_color(req: AdjustRequest) -> dict:
    """Nudge one colour (warmer/cooler, lighter/darker, muted/vivid) and return the
    nearest real paints to the result. Use with preview for live refinement."""
    lab = adjust_lab(rgb_to_lab(hex_to_rgb(req.hex)), req.warmth, req.lightness, req.saturation)
    brands = [req.brand] if req.brand in BRAND_NAMES else None
    paints = load_paints().nearest(lab, k=req.k, brands=brands)
    return {"input": req.hex, "adjusted": rgb_to_hex(lab_to_rgb(lab)),
            "matches": [p.to_dict() for p in paints]}


@app.get("/api/paints/match")
def match_paint(hex: str, k: int = 3, brand: str | None = None) -> dict:
    """Nearest real paints to any colour (e.g. from a colour picker)."""
    try:
        h = _normalise_hex(hex)
    except ValueError as e:
        raise HTTPException(422, str(e))
    k = max(1, min(k, 10))
    brands = [brand] if brand in BRAND_NAMES else None
    paints = load_paints().nearest(rgb_to_lab(hex_to_rgb(h)), k=k, brands=brands)
    return {"query": h, "matches": [p.to_dict() for p in paints]}
