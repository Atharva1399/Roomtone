"""Room segmentation with SegFormer (ADE20K).

ADE20K has 150 fine-grained classes. We fold the ones that matter for colour
decisions into a handful of room "groups" (wall, floor, sofa, ...), and keep a
soft per-pixel probability for each group so recolouring gets smooth edges.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from app import config

# Room group -> ADE20K class names that belong to it.
# Names are matched against the model's id2label after stripping whitespace.
ROOM_GROUPS: dict[str, list[str]] = {
    "wall": ["wall"],
    "ceiling": ["ceiling"],
    "floor": ["floor", "rug", "carpet"],
    "window": ["windowpane", "window", "curtain", "blind", "screen door"],
    "door": ["door", "double door"],
    "sofa": ["sofa", "armchair", "swivel chair", "chair", "seat", "ottoman", "bench", "stool"],
    "bed": ["bed", "blanket", "pillow", "cushion"],
    "table": ["table", "desk", "coffee table", "pool table", "counter", "countertop", "kitchen island"],
    "storage": ["cabinet", "wardrobe", "chest of drawers", "shelf", "bookcase", "buffet", "case"],
    "decor": ["painting", "picture", "mirror", "vase", "lamp", "sconce", "chandelier",
              "light", "plant", "flower", "clock", "book", "poster"],
}

# Groups a user can actually paint.
PAINTABLE_GROUPS = ("wall", "ceiling")


@dataclass
class Segmentation:
    labels: np.ndarray            # (H, W) int, ADE20K class index per pixel
    group_probs: dict[str, np.ndarray]  # group -> (H, W) float32 soft mask in [0, 1]
    group_masks: dict[str, np.ndarray]  # group -> (H, W) bool hard mask (argmax)
    id2label: dict[int, str]

    def fraction(self, group: str) -> float:
        return float(self.group_masks[group].mean()) if group in self.group_masks else 0.0


def load_image(path_or_file, max_side: int = config.MAX_IMAGE_SIDE) -> Image.Image:
    img = Image.open(path_or_file).convert("RGB")
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


@lru_cache(maxsize=1)
def _load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = SegformerImageProcessor.from_pretrained(config.SEGMENTATION_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(
        config.SEGMENTATION_MODEL, dtype=dtype
    ).to(device).eval()
    return processor, model, device, dtype


def _group_class_ids(id2label: dict[int, str]) -> dict[str, list[int]]:
    name_to_id = {name.strip().lower(): i for i, name in id2label.items()}
    return {
        group: [name_to_id[n] for n in names if n in name_to_id]
        for group, names in ROOM_GROUPS.items()
    }


@torch.inference_mode()
def segment(image: Image.Image) -> Segmentation:
    processor, model, device, dtype = _load_model()
    scale = config.SEGMENTATION_SIDE / max(image.size)
    size = {"height": max(32, round(image.height * scale / 32) * 32),
            "width": max(32, round(image.width * scale / 32) * 32)}
    inputs = processor(images=image, size=size, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype)

    logits = model(pixel_values=pixel_values).logits          # (1, 150, h/4, w/4)
    logits = F.interpolate(
        logits.float(), size=(image.height, image.width), mode="bilinear", align_corners=False
    )
    probs = logits.softmax(dim=1)[0]                           # (150, H, W)
    labels = probs.argmax(dim=0)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    group_probs, group_masks = {}, {}
    for group, ids in _group_class_ids(id2label).items():
        if not ids:
            continue
        idx = torch.tensor(ids, device=probs.device)
        group_probs[group] = probs[idx].sum(dim=0).cpu().numpy().astype(np.float32)
        group_masks[group] = torch.isin(labels, idx).cpu().numpy()

    return Segmentation(
        labels=labels.cpu().numpy().astype(np.int32),
        group_probs=group_probs,
        group_masks=group_masks,
        id2label=id2label,
    )


def overlay(image: Image.Image, seg: Segmentation, alpha: float = 0.5) -> Image.Image:
    """Colour-coded group overlay for debugging and demos."""
    palette = {
        "wall": (230, 80, 80), "ceiling": (240, 200, 60), "floor": (90, 160, 90),
        "window": (80, 170, 230), "door": (150, 100, 60), "sofa": (180, 90, 200),
        "bed": (240, 140, 180), "table": (120, 120, 220), "storage": (60, 200, 190),
        "decor": (250, 150, 50),
    }
    base = np.asarray(image, dtype=np.float32)
    out = base.copy()
    for group, mask in seg.group_masks.items():
        out[mask] = (1 - alpha) * base[mask] + alpha * np.array(palette.get(group, (200, 200, 200)))
    return Image.fromarray(out.astype(np.uint8))
