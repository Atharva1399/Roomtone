"""Run the full backend pipeline on one or more room photos.

Usage (from the backend folder, with `conda activate palette-env`):
    python -m scripts.run_pipeline samples/room1.jpg
    python -m scripts.run_pipeline "samples/*.jpg" --goal calm
    python -m scripts.run_pipeline samples/room1.jpg --wall "#A8B5A2"   # just test a colour

Writes, per photo, to outputs/<name>/:
    segmentation.png   colour-coded room regions
    analysis.json      current palette, 60-30-10 roles, lighting
    palettes.json      suggested palettes with paints, checks and facts
    palette_N.png      before | after preview with the palette's swatches
"""

import argparse
import glob
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.pipeline.palette import analyze_colors
from app.pipeline.recolor import is_plain_surface, recolor, swatch_strip
from app.pipeline.segmentation import load_image, overlay, segment
from app.pipeline.suggest import GOALS, suggest_palettes

ROLE_ORDER = ("dominant", "secondary", "accent", "ceiling")


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def palette_sheet(before: Image.Image, after: Image.Image, palette: dict, title: str) -> Image.Image:
    """Before | after, then one labelled swatch per 60-30-10 role underneath."""
    w, h = before.size
    band = 110
    sheet = Image.new("RGB", (w * 2, h + band + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill="black", font=_font(20))
    sheet.paste(before, (0, 34))
    sheet.paste(after, (w, 34))

    sw = (w * 2) // len(ROLE_ORDER)
    for i, role in enumerate(ROLE_ORDER):
        r = palette["roles"][role]
        x0, y0 = i * sw, h + 34
        sheet.paste(swatch_strip([r["hex"]], width=sw, height=60), (x0, y0))
        name = f"{r.get('brand', '')} {r.get('name', '')} {r.get('code', '')}".strip() or r["source"]
        draw.text((x0 + 8, y0 + 64), f"{role.upper()}  {r['hex']}  LRV {r['lrv']:.0f}", fill="black", font=_font(14))
        draw.text((x0 + 8, y0 + 84), name[:48], fill=(60, 60, 60), font=_font(13))
    return sheet


def run(path: Path, goal: str, test_walls: list[str] | None, n: int, out_root: Path) -> None:
    out = out_root / path.stem
    out.mkdir(parents=True, exist_ok=True)
    timing = {}

    t = time.perf_counter()
    image = load_image(path)
    seg = segment(image)
    timing["segmentation"] = time.perf_counter() - t

    t = time.perf_counter()
    analysis = analyze_colors(image, seg)
    timing["analysis"] = time.perf_counter() - t
    overlay(image, seg).save(out / "segmentation.png")
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))

    print(f"\n{path.name}  ({image.width}x{image.height})  lighting: "
          f"{analysis['lighting']['brightness']}, {analysis['lighting']['temperature']}")

    if test_walls:
        for hex_color in test_walls:
            result, rep = recolor(image, seg, {"wall": hex_color})
            palette = {"roles": {r: {"hex": hex_color, "lrv": 0, "source": "test"} for r in ROLE_ORDER}}
            palette_sheet(image, result, palette, f"test {hex_color}").save(out / f"test_{hex_color.lstrip('#')}.png")
            print(f"  test {hex_color}: {rep['wall']}")
        return

    t = time.perf_counter()
    result = suggest_palettes(analysis, goal=goal, n=n)
    timing["suggest"] = time.perf_counter() - t
    (out / "palettes.json").write_text(json.dumps(result, indent=2))

    paint_ceiling = is_plain_surface(image, seg, "ceiling")
    print(f"  base hue {result['base_hue']:.0f}deg: {result['base_hue_reason']}"
          f"  | ceiling {'repainted' if paint_ceiling else 'left as is (not plain or not visible)'}")
    t = time.perf_counter()
    for i, p in enumerate(result["palettes"], 1):
        colors = {"wall": p["roles"]["dominant"]["hex"]}
        if paint_ceiling:
            colors["ceiling"] = p["roles"]["ceiling"]["hex"]
        after, _ = recolor(image, seg, colors)
        d = p["roles"]["dominant"]
        title = f"#{i} {p['scheme']} (score {p['score']:.2f}) - walls: {d['brand']} {d['name']} {d['code']}"
        palette_sheet(image, after, p, title).save(out / f"palette_{i}.png")
        a = p["roles"]["accent"]
        soft = [c["rule"] for c in p["checks"] if not c["passed"]]
        label = p["facts"][0].removeprefix("scheme: ")
        print(f"  {i}. {label:32s} wall {d['hex']} {d['brand']} {d['code']} '{d['name']}'  "
              f"accent {a['hex']} '{a['name'] or a['code']}'  score {p['score']:.2f}"
              + (f"  warn: {soft}" if soft else ""))
    timing["previews"] = time.perf_counter() - t
    print(f"  {len(result['palettes'])} shown of {result['candidates_considered']} candidates, "
          f"{len(result['rejected'])} failed a hard check")
    print("  timing: " + ", ".join(f"{k} {v:.2f}s" for k, v in timing.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="room photo paths (globs ok)")
    parser.add_argument("--goal", default="balanced", choices=list(GOALS))
    parser.add_argument("-n", type=int, default=4, help="palettes per photo")
    parser.add_argument("--wall", action="append", help="skip suggestions; just preview this wall colour")
    parser.add_argument("--out", default="outputs", help="output folder")
    args = parser.parse_args()

    paths = [Path(p) for pattern in args.images for p in (glob.glob(pattern) or [pattern])]
    for p in paths:
        run(p, args.goal, args.wall, args.n, Path(args.out))


if __name__ == "__main__":
    main()
