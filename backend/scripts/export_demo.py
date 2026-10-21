"""Export real pipeline output for the frontend landing page.

Everything the landing page shows about a room (before/after, the lighting
layer, palettes, counts) is produced here by the same code the API runs, so
the page demonstrates the mechanism instead of describing it.

Usage (from the backend folder, with `conda activate palette-env`):
    python -m scripts.export_demo ../frontend/public/demo
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipeline import click_mask
from app.pipeline.palette import analyze_colors
from app.pipeline.paints import load_paints
from app.pipeline.recolor import apply_paints, prepare_surface, to_image
from app.pipeline.segmentation import load_image, segment
from app.pipeline.suggest import SCHEMES, NEUTRAL_SCHEMES, suggest_palettes

SAMPLES = Path(__file__).resolve().parents[1] / "samples"

# Rooms offered on the landing page. The Kolkata photo carries a third-party
# watermark, so it is never used here.
ROOMS = {
    # excludes: click-to-fix clicks (in 1024-px coordinates), as a user would make in the studio
    "dining_room": {"title": "Dining room", "excludes": [(150, 520), (530, 385), (40, 600)]},  # wainscot, tilt-top table, floor
    "bedroom_historic": {"title": "Canopy bedroom", "excludes": []},
    "living_luxury": {"title": "Open-plan living", "excludes": []},
    "bedroom_modern": {"title": "City bedroom", "excludes": []},
}
HERO = "dining_room"
LIGHT_DEMO = "bedroom_modern"
HERO_SIDE = 1920
CARD_SIDE = 1024


def save_jpg(img: Image.Image, path: Path, quality: int = 86) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality, optimize=True, progressive=True)


def wall_prob(image, seg, excludes, base_side):
    """Wall mask with the same click-to-fix edits the studio would make."""
    prob = seg.group_probs["wall"].copy()
    if excludes:
        emb = click_mask.embed(image)
        scale = image.width / base_side
        for x, y in excludes:
            opts = click_mask.mask_options(emb, int(x * scale), int(y * scale))
            region, _ = click_mask.pick(opts, "object")
            prob[region] = 0.0
    return prob


def palette_summary(p: dict) -> dict:
    keep = ("hex", "brand", "name", "code", "lrv", "source", "use")
    return {"scheme": p["scheme"], "facts": p["facts"],
            "roles": {r: {k: v for k, v in p["roles"][r].items() if k in keep}
                      for r in ("dominant", "secondary", "accent", "ceiling")},
            "checks": [{"rule": c["rule"], "passed": c["passed"], "detail": c["detail"]}
                       for c in p["checks"]]}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = {"rooms": {}, "counts": {
        "paints": len(load_paints()),
        "room_classes": 150,
        "harmony_schemes": len(SCHEMES) + len(NEUTRAL_SCHEMES),
    }}

    for name, meta in ROOMS.items():
        side = HERO_SIDE if name in (HERO, LIGHT_DEMO) else CARD_SIDE
        image = load_image(SAMPLES / f"{name}.jpg", max_side=side)
        seg = segment(image)
        analysis = analyze_colors(image, seg)
        result = suggest_palettes(analysis, n=4)
        pals = result["palettes"]

        prob = wall_prob(image, seg, meta["excludes"], base_side=1024)
        layers = {"wall": prepare_surface(image, seg, "wall", prob=prob)}
        rgb = np.asarray(image, dtype=np.float32) / 255.0

        room_dir = out / "rooms" / name
        save_jpg(image, room_dir / "original.jpg")
        for i, p in enumerate(pals):
            painted, _ = apply_paints(rgb, layers, {"wall": p["roles"]["dominant"]["hex"]})
            save_jpg(to_image(painted), room_dir / f"palette_{i}.jpg")

        if name == HERO:
            # Segmentation in the site's palette: walls cobalt, floor rust, furniture walnut,
            # windows and decor stone. Same masks the pipeline paints with.
            tints = {"wall": (38, 67, 196), "floor": (194, 69, 42), "ceiling": (90, 110, 210),
                     "sofa": (74, 47, 33), "table": (74, 47, 33), "storage": (74, 47, 33),
                     "bed": (74, 47, 33), "window": (217, 207, 192), "door": (74, 47, 33),
                     "decor": (217, 207, 192)}
            seg_img = np.asarray(image, dtype=np.float32) * 0.45 + 140
            for group, mask in seg.group_masks.items():
                if group in tints:
                    seg_img[mask] = seg_img[mask] * 0.35 + np.array(tints[group]) * 0.65
            save_jpg(Image.fromarray(np.clip(seg_img, 0, 255).astype(np.uint8)), room_dir / "segmentation.jpg")

        if name == LIGHT_DEMO:
            # The lighting layer: the wall's own light and shadow on a neutral mid-grey
            # paint, with the rest of the room faded back so the light reads.
            lay = layers["wall"]
            grey, _ = apply_paints(rgb, layers, {"wall": "#8E8C88"})
            faded = rgb * 0.22 + np.array([0.93, 0.91, 0.88], np.float32) * 0.78
            shown = lay.alpha * grey + (1 - lay.alpha) * faded
            save_jpg(to_image(shown), room_dir / "light_layer.jpg")

        data["rooms"][name] = {
            "title": meta["title"],
            "size": [image.width, image.height],
            "lighting": analysis["lighting"],
            "current_roles": {k: {"hex": v["hex"], "source": v["source"]}
                              for k, v in analysis["current_roles"].items()},
            "base_hue_reason": result["base_hue_reason"],
            "candidates_considered": result["candidates_considered"],
            "excluded_clicks": len(meta["excludes"]),
            "palettes": [palette_summary(p) for p in pals],
        }
        print(f"{name}: {image.size}, {len(pals)} palettes")

    data["hero"], data["light_demo"] = HERO, LIGHT_DEMO
    (out / "demo.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print("wrote", out / "demo.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
