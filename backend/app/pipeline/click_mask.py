"""Click-to-fix for paint masks, using Segment Anything (SAM).

The room segmenter sometimes labels wood panelling, a wardrobe or a window
blind as "wall". The user clicks on the mistake; SAM returns the object under
the click; we remove it from (or add it to) the surface's paint mask.

SAM's heavy step (the image embedding, ~0.3s on GPU) runs once per photo; each
click after that is ~30ms.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

SAM_MODEL = "facebook/sam-vit-base"


@dataclass
class SamEmbedding:
    embeddings: torch.Tensor          # kept on CPU between clicks to spare VRAM
    original_size: tuple[int, int]    # (H, W) of the room image
    reshaped_size: tuple[int, int]    # size SAM resized it to (longest side 1024)


@lru_cache(maxsize=1)
def _load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained(SAM_MODEL)
    model = SamModel.from_pretrained(SAM_MODEL).to(device).eval()
    return processor, model, device


@torch.inference_mode()
def embed(image: Image.Image) -> SamEmbedding:
    processor, model, device = _load()
    inputs = processor(images=image, return_tensors="pt")
    emb = model.get_image_embeddings(inputs["pixel_values"].to(device))
    return SamEmbedding(
        embeddings=emb.cpu(),
        original_size=tuple(int(v) for v in inputs["original_sizes"][0]),
        reshaped_size=tuple(int(v) for v in inputs["reshaped_input_sizes"][0]),
    )


@torch.inference_mode()
def mask_options(emb: SamEmbedding, x: int, y: int) -> list[tuple[np.ndarray, float]]:
    """SAM's three readings of a click (sub-part / part / whole object) as
    (mask (H, W) bool, confidence), largest first."""
    processor, model, device = _load()
    H, W = emb.original_size
    scale = emb.reshaped_size[0] / H          # same factor on both axes
    points = torch.tensor([[[[x * scale, y * scale]]]], dtype=torch.float32, device=device)
    labels = torch.ones((1, 1, 1), dtype=torch.long, device=device)

    out = model(image_embeddings=emb.embeddings.to(device), input_points=points,
                input_labels=labels, multimask_output=True)
    masks = processor.image_processor.post_process_masks(
        out.pred_masks.cpu(), torch.tensor([emb.original_size]), torch.tensor([emb.reshaped_size]))[0][0]
    scores = out.iou_scores[0, 0].float().cpu().numpy()
    opts = [(masks[i].numpy().astype(bool), float(scores[i])) for i in range(len(scores))]
    return sorted(opts, key=lambda o: -o[0].sum())


def pick(options: list[tuple[np.ndarray, float]], level: str = "object",
         min_score: float = 0.7) -> tuple[np.ndarray, float]:
    """object: the largest confident mask, e.g. all the wood panelling from one click.
    part:   SAM's single most confident mask, often one panel or one door."""
    if level == "part":
        return max(options, key=lambda o: o[1])
    confident = [o for o in options if o[1] >= min_score]
    return (confident or [max(options, key=lambda o: o[1])])[0]
