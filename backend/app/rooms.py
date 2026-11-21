"""In-memory cache of uploaded rooms.

Segmentation (~0.3s on GPU) and the per-surface lighting layers (~0.5s each)
are computed once per photo; every later preview only multiplies a new paint
colour into the cached layers. Click edits to a surface's mask invalidate just
that surface's layers.
"""

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app import config
from app.pipeline import click_mask, palette_ranker
from app.pipeline.palette import analyze_colors
from app.pipeline.recolor import SurfaceLayers, is_plain_surface, prepare_surface
from app.pipeline.segmentation import Segmentation, segment

# One photo through the GPU at a time: a 4 GB card can't hold several at once.
_gpu_lock = threading.Lock()


@dataclass
class Room:
    id: str
    image: Image.Image
    rgb: np.ndarray                      # (H, W, 3) float32 in [0, 1]
    seg: Segmentation
    analysis: dict
    paint_ceiling: bool                  # ceiling is plain enough to repaint
    style: dict | None = None            # the LLM's style read, cached once fetched
    ranker_context: dict | None = None   # the room as Roomtone-Palette (V2) sees it, if installed
    # Per surface, the user's click edits in order: (region mask, "exclude" | "include").
    mask_edits: dict[str, list[tuple[np.ndarray, str]]] = field(default_factory=dict)
    _sam: click_mask.SamEmbedding | None = None
    explain_seq: int = 0                 # bumps on every explanation request
    _layers: dict[str, SurfaceLayers | None] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def surface_prob(self, group: str) -> np.ndarray | None:
        """The segmentation's probability map for a surface with click edits applied."""
        base = self.seg.group_probs.get(group)
        edits = self.mask_edits.get(group)
        if not edits:
            return base
        prob = base.copy() if base is not None else np.zeros(self.rgb.shape[:2], np.float32)
        for region, mode in edits:
            prob[region] = 0.0 if mode == "exclude" else 1.0
        return prob

    def layers(self, groups) -> dict[str, SurfaceLayers | None]:
        with self._lock:
            for g in groups:
                if g not in self._layers:
                    self._layers[g] = prepare_surface(self.image, self.seg, g,
                                                      prob=self.surface_prob(g))
            return {g: self._layers[g] for g in groups}

    def edit_mask(self, group: str, x: int, y: int, mode: str, level: str = "object") -> dict:
        """Remove (exclude) or add (include) the object under a click from a surface."""
        with self._lock:
            if self._sam is None:
                with _gpu_lock:
                    self._sam = click_mask.embed(self.image)
            with _gpu_lock:
                options = click_mask.mask_options(self._sam, x, y)

            current = self.surface_prob(group)
            before = current > 0.5 if current is not None else np.zeros(self.rgb.shape[:2], bool)
            region, score = click_mask.pick(options, level)
            # Safeguard: one exclude click must not wipe out most of the surface
            # (e.g. clicking the wall itself); fall back to a smaller reading.
            if mode == "exclude" and before.any():
                for cand, s in [(region, score)] + [o for o in options if o[0].sum() < region.sum()]:
                    if (cand & before).sum() <= 0.6 * before.sum():
                        region, score = cand, s
                        break
                else:
                    return {"applied": False, "reason": "that click covers most of the surface; "
                                                        "click on the object to remove instead"}

            self.mask_edits.setdefault(group, []).append((region, mode))
            self._layers.pop(group, None)
            after = self.surface_prob(group) > 0.5
            return {"applied": True, "sam_score": round(score, 3),
                    "region_fraction": round(float(region.mean()), 4),
                    "changed_fraction": round(float((before ^ after).mean()), 4),
                    **self.mask_state(group)}

    def undo_mask(self, group: str) -> dict:
        with self._lock:
            if self.mask_edits.get(group):
                self.mask_edits[group].pop()
                self._layers.pop(group, None)
            return self.mask_state(group)

    def reset_mask(self, group: str) -> dict:
        with self._lock:
            self.mask_edits.pop(group, None)
            self._layers.pop(group, None)
            return self.mask_state(group)

    def mask_state(self, group: str) -> dict:
        prob = self.surface_prob(group)
        return {"group": group, "edits": len(self.mask_edits.get(group, [])),
                "coverage": round(float((prob > 0.5).mean()), 4) if prob is not None else 0.0}


class RoomStore:
    def __init__(self, max_rooms: int = config.MAX_CACHED_ROOMS):
        self._rooms: OrderedDict[str, Room] = OrderedDict()
        self._max = max_rooms
        self._lock = threading.Lock()

    def create(self, image: Image.Image) -> Room:
        with _gpu_lock:
            seg = segment(image)
        room = Room(
            id=uuid.uuid4().hex,
            image=image,
            rgb=np.asarray(image, dtype=np.float32) / 255.0,
            seg=seg,
            analysis=analyze_colors(image, seg),
            paint_ceiling=is_plain_surface(image, seg, "ceiling"),
            ranker_context=(palette_ranker.measure_context(np.asarray(image), seg.group_masks)
                            if palette_ranker.available() else None),
        )
        # Warm the wall layers now so the first preview is already fast.
        room.layers(["wall"])
        with self._lock:
            self._rooms[room.id] = room
            self.latest_id = room.id
            while len(self._rooms) > self._max:
                self._rooms.popitem(last=False)
        return room

    def get(self, room_id: str) -> Room | None:
        with self._lock:
            room = self._rooms.get(room_id)
            if room:
                self._rooms.move_to_end(room_id)
            return room

    def is_latest(self, room_id: str) -> bool:
        """The most recently uploaded room: model work for older rooms is skipped."""
        return getattr(self, "latest_id", None) == room_id

    def __len__(self) -> int:
        return len(self._rooms)


store = RoomStore()
