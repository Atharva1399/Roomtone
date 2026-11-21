# %% [markdown]
# # Roomtone-Palette (V2): a learned wall-colour ranker
#
# Roomtone **V1** ranks candidate palettes with hand-written rules (colour-wheel harmony with the
# furniture, lightness for the goal, five checks). **V2** adds **Roomtone-Palette**, a small neural
# network trained by us to score **how well a wall colour suits a room**: its floor, furniture,
# decor and light. In the app, V1's checks still filter out unsafe palettes; Roomtone-Palette
# re-ranks the ones that pass.
#
# **Where the data comes from (built automatically, no manual labelling):** the indoor photos of
# ADE20K (the same ones Roomtone-Surface trained on, cached in your Drive). For every room we use
# ADE20K's own labels to measure the real wall colour and the colours it lives with, after
# correcting the photo's colour cast and exposure.
#
# **What the model learns:** pick the room's real wall colour out of a line-up of real wall colours
# from *other* rooms. Every candidate is a colour people really chose for some room, so the only way
# to win is to understand which colours go with *this* room's floor, furniture and light.
#
# **Honest limits:** ADE20K walls are everyday rooms, not designer picks, so the model learns
# "what real rooms pair together", not taste. A blind A/B test with people is the next step.
#
# Runs on Colab in about 20–40 min (mostly measuring the photos); a GPU is optional.
# Credits: ADE20K, Zhou et al., MIT CSAIL (research/non-commercial terms).

# %% [markdown]
# ## 1. Setup

# %%
import os, sys, json, math, time, shutil, zipfile, urllib.request
from pathlib import Path
import numpy as np

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/roomtone_palette")
    SURFACE_DIR = Path("/content/drive/MyDrive/roomtone_surface")   # ADE20K zip + indoor list from V2 surface
    DATA_DIR = Path("/content/data")
else:   # local runs: set these to your own folders
    DRIVE_DIR = Path(os.environ.get("PALETTE_OUT", "roomtone_palette"))
    SURFACE_DIR = Path(os.environ.get("SURFACE_DIR", "roomtone_surface"))
    DATA_DIR = Path(os.environ.get("ADE_DATA", "data"))
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
ADE_ROOT = DATA_DIR / "ADEChallengeData2016"
ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

WORK_SIDE = 512             # photos are measured at this longest side (plenty for colours)
NEGATIVES = 15              # wrong walls per room in training (10 from other rooms, 5 shifted)
EPOCHS = 40
BATCH_ROOMS = 256
LR = 1e-3
SEED = 0
TEST_LINEUP = 20            # test: real wall + 19 walls from other test rooms

# %%
def fetch_ade20k():
    if (ADE_ROOT / "annotations").exists():
        return
    cached = SURFACE_DIR / "ADE20K.zip"
    local = DATA_DIR / "ADE20K.zip"
    if cached.exists():
        print("copying ADE20K from the Roomtone-Surface folder…")
        shutil.copy(cached, local)
    else:
        print("downloading ADE20K (about 1 GB)…")
        urllib.request.urlretrieve(ADE_URL, local)
        if IN_COLAB:
            shutil.copy(local, cached if SURFACE_DIR.exists() else DRIVE_DIR / "ADE20K.zip")
    with zipfile.ZipFile(local) as z:
        z.extractall(DATA_DIR)
    local.unlink()

if __name__ == "__main__" or IN_COLAB:
    fetch_ade20k()

# %% [markdown]
# ## 2. Room groups
# The same groups as the Roomtone backend (`backend/app/pipeline/segmentation.py`), applied to
# ADE20K's ground-truth labels. The 150 names are ADE20K's classes in order (annotation value − 1).

# %%
ADE150 = ("wall building sky floor tree ceiling road bed windowpane grass cabinet sidewalk person earth "
          "door table mountain plant curtain chair car water painting sofa shelf house sea mirror rug field "
          "armchair seat fence desk rock wardrobe lamp bathtub railing cushion base box column signboard "
          "chest_of_drawers counter sand sink skyscraper fireplace refrigerator grandstand path stairs runway "
          "case pool_table pillow screen_door stairway river bridge bookcase blind coffee_table toilet flower "
          "book hill bench countertop stove palm kitchen_island computer swivel_chair boat bar arcade_machine "
          "hovel bus towel light truck tower chandelier awning streetlight booth television_receiver airplane "
          "dirt_track apparel pole land bannister escalator ottoman bottle buffet poster stage van ship "
          "fountain conveyer_belt canopy washer plaything swimming_pool stool barrel basket waterfall tent bag "
          "minibike cradle oven ball food step tank trade_name microwave pot animal bicycle lake dishwasher "
          "screen blanket sculpture hood sconce vase traffic_light tray ashcan fan pier crt_screen plate "
          "monitor bulletin_board shower radiator glass clock flag").replace("_", "~").split()
ADE150 = [n.replace("~", " ") for n in ADE150]
assert len(ADE150) == 150

ROOM_GROUPS = {
    "wall": ["wall"], "ceiling": ["ceiling"], "floor": ["floor", "rug", "carpet"],
    "window": ["windowpane", "window", "curtain", "blind", "screen door"],
    "door": ["door", "double door"],
    "sofa": ["sofa", "armchair", "swivel chair", "chair", "seat", "ottoman", "bench", "stool"],
    "bed": ["bed", "blanket", "pillow", "cushion"],
    "table": ["table", "desk", "coffee table", "pool table", "counter", "countertop", "kitchen island"],
    "storage": ["cabinet", "wardrobe", "chest of drawers", "shelf", "bookcase", "buffet", "case"],
    "decor": ["painting", "picture", "mirror", "vase", "lamp", "sconce", "chandelier",
              "light", "plant", "flower", "clock", "book", "poster"],
}
GROUPS = list(ROOM_GROUPS)
CONTEXT = ["floor", "sofa", "bed", "table", "storage", "door", "window", "decor"]  # what the wall must suit
NONE = 255
LUT = np.full(256, NONE, np.uint8)            # annotation value -> group index
for gi, (g, names) in enumerate(ROOM_GROUPS.items()):
    for n in names:
        if n in ADE150:
            LUT[ADE150.index(n) + 1] = gi
SKY = ADE150.index("sky") + 1

# %% [markdown]
# ## 3. Build the dataset: measure every indoor room
# For each room: the colour of each group (the biggest k-means cluster in CIELAB, as V1 measures
# colours) and its area. Two corrections make photos comparable:
# * **Colour cast:** the ceiling is almost always white, so its tint is the light's tint; it is
#   subtracted from every colour (the brightest non-wall pixels stand in when there is no ceiling).
# * **Exposure:** colours are rescaled so the room's bright neutral surfaces sit near L* 95.
#
# Both are measured **without the wall pixels**, so the light features never leak the answer.
# The result is saved to Drive as `palette_rooms.csv`: the dataset.

# %%
import cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from sklearn.cluster import KMeans
from multiprocessing import Pool

def dominant_lab(lab_px: np.ndarray, rng) -> np.ndarray | None:
    lab_px = lab_px[(lab_px[:, 0] > 8) & (lab_px[:, 0] < 99)]
    if len(lab_px) < 50:
        return None
    if len(lab_px) > 3000:
        lab_px = lab_px[rng.choice(len(lab_px), 3000, replace=False)]
    km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(lab_px)
    return km.cluster_centers_[np.bincount(km.labels_, minlength=3).argmax()]

def measure_room(args):
    split, stem = args
    try:
        img = cv2.cvtColor(cv2.imread(str(ADE_ROOT / "images" / split / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        ann = cv2.imread(str(ADE_ROOT / "annotations" / split / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    s = WORK_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ann = cv2.resize(ann, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    grp = LUT[ann]
    labelled = ann > 0
    if labelled.sum() == 0 or (ann == SKY).mean() > 0.01:
        return None
    lab = rgb2lab(img.astype(np.float32) / 255).reshape(-1, 3)
    grp, labelled = grp.reshape(-1), labelled.reshape(-1)
    wall = grp == GROUPS.index("wall")
    if wall.sum() / labelled.sum() < 0.05:
        return None

    # Light, measured without wall pixels (and without windows, which show the outdoors).
    ceil = grp == GROUPS.index("ceiling")
    not_wall = labelled & ~wall & (grp != GROUPS.index("window"))
    if ceil.sum() / labelled.sum() >= 0.02:
        ref = np.median(lab[ceil], axis=0)
        light_src = "ceiling"
    else:
        Ls = lab[not_wall, 0]
        bright = lab[not_wall][Ls >= np.percentile(Ls, 80)] if not_wall.sum() > 100 else lab[labelled]
        ref = np.median(bright, axis=0)
        light_src = "bright"
    exposure = float(np.clip(95 / max(np.percentile(lab[not_wall | ceil, 0], 95) if (not_wall | ceil).sum() > 100 else ref[0], 1), 0.8, 2.0))
    norm = lab.copy()
    norm[:, 0] = np.clip(norm[:, 0] * exposure, 0, 100)
    norm[:, 1:] -= ref[1:]

    rng = np.random.default_rng(0)
    row = {"split": split, "id": stem, "light_L": float(ref[0]), "light_a": float(ref[1]), "light_b": float(ref[2]),
           "exposure": exposure, "light_from_ceiling": light_src == "ceiling"}
    total = labelled.sum()
    for gi, g in enumerate(GROUPS):
        m = grp == gi
        area = m.sum() / total
        c = dominant_lab(norm[m], rng) if area >= 0.005 else None
        row[f"{g}_area"] = float(area) if c is not None else 0.0
        for k, v in zip("Lab", c if c is not None else (np.nan,) * 3):
            row[f"{g}_{k}"] = float(v)
    if np.isnan(row["wall_L"]):
        return None
    return row

def load_index() -> dict:
    for f in (SURFACE_DIR / "indoor_index.json", DRIVE_DIR / "indoor_index.json"):
        if f.exists():
            return json.loads(f.read_text())
    # Same rule as Roomtone-Surface: wall+ceiling+floor >= 25 %, wall >= 5 %, sky < 1 %.
    index = {}
    for split in ("training", "validation"):
        keep = []
        for p in sorted((ADE_ROOT / "annotations" / split).glob("*.png")):
            ann = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)[::4, ::4]
            g, v = LUT[ann], ann > 0
            n = v.sum()
            if not n:
                continue
            fr = np.bincount(g[v & (g != NONE)], minlength=len(GROUPS)) / n
            if fr[0] + fr[1] + fr[2] >= 0.25 and fr[0] >= 0.05 and (ann == SKY).sum() / n < 0.01:
                keep.append(p.stem)
        index[split] = keep
    (DRIVE_DIR / "indoor_index.json").write_text(json.dumps(index))
    return index

def build_dataset():
    import pandas as pd
    out = DRIVE_DIR / "palette_rooms.csv"
    if out.exists():
        return pd.read_csv(out)
    index = load_index()
    jobs = [(s, i) for s in ("training", "validation") for i in index[s]]
    t = time.time()
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        rows = [r for r in pool.imap(measure_room, jobs, chunksize=16) if r]
    print(f"measured {len(rows)} of {len(jobs)} rooms in {time.time() - t:.0f} s")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df

if __name__ == "__main__" or IN_COLAB:
    ROOMS = build_dataset()
    print(ROOMS.split.value_counts().to_dict(), "rooms")
    print(ROOMS[[f"{g}_area" for g in GROUPS]].gt(0).mean().round(2).to_dict(), "share of rooms with each group")

# %% [markdown]
# ## 4. Features
# For a room and one candidate wall colour: the candidate itself (L, a, b, chroma, hue), the room's
# light, and for every context group (floor, sofa, bed, table, storage, door, window, decor): its
# colour, its area, and how the candidate relates to it (CIEDE2000 distance, lightness, chroma and
# hue differences). Missing groups are zeros with a "present" flag.

# %%
def lch(lab):
    a, b = lab[..., 1], lab[..., 2]
    return np.sqrt(a ** 2 + b ** 2), np.arctan2(b, a)

def room_context(df) -> dict:
    """Per-room arrays the feature function needs, from the dataset table."""
    ctx = {"light": df[["light_L", "light_a", "light_b", "exposure"]].to_numpy(np.float32)}
    for g in CONTEXT:
        ctx[g] = df[[f"{g}_L", f"{g}_a", f"{g}_b"]].to_numpy(np.float32)
        ctx[g + "_area"] = df[f"{g}_area"].to_numpy(np.float32)
    ctx["wall"] = df[["wall_L", "wall_a", "wall_b"]].to_numpy(np.float32)
    return ctx

def features(ctx: dict, rooms: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """rooms: (N,) room indices; cand: (N, 3) candidate wall Lab -> (N, F) features."""
    cC, cH = lch(cand)
    parts = [cand / np.array([100, 50, 50], np.float32), (cC / 50)[:, None],
             np.cos(cH)[:, None], np.sin(cH)[:, None],
             ctx["light"][rooms] / np.array([100, 20, 20, 2], np.float32)]
    for g in CONTEXT:
        lab = ctx[g][rooms]
        area = ctx[g + "_area"][rooms]
        present = (area > 0) & ~np.isnan(lab[:, 0])
        lab0 = np.where(present[:, None], lab, 0)
        gC, gH = lch(lab0)
        de = np.where(present, deltaE_ciede2000(np.where(present[:, None], cand, 0), lab0), 0)
        dh = cH - gH
        w = np.minimum(gC, 20) / 20 * np.minimum(cC, 20) / 20   # hue only matters when both have colour
        parts += [present[:, None].astype(np.float32), np.sqrt(area)[:, None], lab0 / np.array([100, 50, 50]),
                  (de / 50)[:, None], ((cand[:, 0] - lab0[:, 0]) / 50 * present)[:, None],
                  ((cC - gC) / 50 * present)[:, None], (np.cos(dh) * w)[:, None], (np.sin(dh) * w)[:, None]]
    return np.concatenate(parts, 1).astype(np.float32)

# %% [markdown]
# ## 5. Line-ups: the real wall vs wrong walls
# Training: each room's real wall + 10 walls taken from other rooms + 5 shifted versions of its own
# wall (hue turned 60–180°, or lightness moved 15–35). Fresh line-ups every epoch.

# %%
def shifted(wall: np.ndarray, rng, n: int) -> np.ndarray:
    out = np.repeat(wall[:, None], n, 1).copy()
    C, H = lch(out)
    turn = rng.random(out.shape[:2]) < 0.6
    H = H + turn * rng.choice([-1, 1], out.shape[:2]) * np.deg2rad(rng.uniform(60, 180, out.shape[:2]))
    C = np.where(turn, np.maximum(C, rng.uniform(4, 25, out.shape[:2])), C)
    out[..., 1], out[..., 2] = C * np.cos(H), C * np.sin(H)
    out[..., 0] = np.where(turn, out[..., 0], np.clip(out[..., 0] + rng.choice([-1, 1], out.shape[:2]) * rng.uniform(15, 35, out.shape[:2]), 15, 97))
    return out

def lineups(ctx, rooms: np.ndarray, rng, n_swap: int, n_shift: int, pool: np.ndarray | None = None):
    """(R, 1 + n_swap + n_shift, 3) candidate walls; index 0 is always the real wall."""
    pool = rooms if pool is None else pool
    real = ctx["wall"][rooms][:, None]
    swap_ids = rng.choice(pool, (len(rooms), n_swap))
    clash = swap_ids == rooms[:, None]
    swap_ids[clash] = rng.choice(pool, clash.sum())
    parts = [real, ctx["wall"][swap_ids]]
    if n_shift:
        parts.append(shifted(ctx["wall"][rooms], rng, n_shift))
    return np.concatenate(parts, 1)

def featurise(ctx, rooms, cands):
    R, K, _ = cands.shape
    X = features(ctx, np.repeat(rooms, K), cands.reshape(-1, 3))
    return X.reshape(R, K, -1)

# %% [markdown]
# ## 6. The model and training
# A small neural network (about 50k weights) scores one (room, wall) pair; training pushes the real
# wall to the top of its line-up (softmax over the line-up). 10 % of training rooms are held back to
# pick the best epoch; ADE20K's validation rooms are kept for the final test only.

# %%
import torch
import torch.nn as nn

class Ranker(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1))
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)

def mrr_top1(scores: np.ndarray) -> dict:
    """scores: (R, K) with the real wall at index 0. Ties are broken at random, so a method
    that gives many walls the same score gets no credit for the tie."""
    scores = scores + np.random.default_rng(0).uniform(0, 1e-6, scores.shape) * (np.abs(scores).max() + 1)
    rank = (scores[:, 1:] > scores[:, :1]).sum(1) + 1
    return {"top1": float((rank == 1).mean()), "top3": float((rank <= 3).mean()),
            "MRR": float((1 / rank).mean()),
            "beats_share": float((scores[:, 1:] < scores[:, :1]).mean())}

def score_model(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        dev = next(model.parameters()).device
        out = np.concatenate([model(torch.from_numpy(X[i:i + 512]).to(dev)).cpu().numpy()
                              for i in range(0, len(X), 512)])
    model.train()
    return out

def train_ranker(df):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    ctx = room_context(df)
    train_all = np.flatnonzero(df.split.to_numpy() == "training")
    rng.shuffle(train_all)
    n_hold = len(train_all) // 10
    hold, train = train_all[:n_hold], train_all[n_hold:]
    X_hold = featurise(ctx, hold, lineups(ctx, hold, np.random.default_rng(1), TEST_LINEUP - 1, 0))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sample = featurise(ctx, train[:2000], lineups(ctx, train[:2000], rng, 10, 5))
    model = Ranker(sample.shape[-1]).to(dev)
    flat = torch.from_numpy(sample.reshape(-1, sample.shape[-1]))
    model.mean.copy_(flat.mean(0)); model.std.copy_(flat.std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best, best_state, history = -1, None, []

    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train)
        X = featurise(ctx, order, lineups(ctx, order, rng, 10, NEGATIVES - 10))
        losses = []
        for i in range(0, len(order), BATCH_ROOMS):
            xb = torch.from_numpy(X[i:i + BATCH_ROOMS]).to(dev)
            logits = model(xb)                                   # (B, K)
            loss = nn.functional.cross_entropy(logits, torch.zeros(len(xb), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
        R, K, F = X_hold.shape
        m = mrr_top1(score_model(model, X_hold.reshape(-1, F)).reshape(R, K))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **m})
        if m["MRR"] > best:
            best, best_state = m["MRR"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:2d}  loss {np.mean(losses):.3f}  held-out top1 {m['top1']:.3f}  MRR {m['MRR']:.3f}")
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "n_in": sample.shape[-1], "history": history}, DRIVE_DIR / "ranker.pt")
    return model.cpu(), ctx

if __name__ == "__main__" or IN_COLAB:
    MODEL, CTX = train_ranker(ROOMS)

# %% [markdown]
# ## 7. Test on rooms the model never saw: V2 vs V1's rules vs baselines
# Each test room (ADE20K validation) gets a line-up of its real wall + 19 real walls from other test
# rooms. Whoever ranks the real wall higher understands the room better.
# * **Random:** chance is 1 in 20 (top-1 5 %).
# * **Popularity:** "common wall colours first" (how many training walls look alike). It cannot
#   use the room at all, so it shows how much the room context matters.
# * **V1 rules:** Roomtone V1's own ranking: harmony with the furniture's hues, the wall-vs-anchor
#   and not-garish checks, and lightness for the "balanced" goal.
# * **V2 Roomtone-Palette:** the trained model.
#
# A second test, **"Hue shift"**, lines the real wall up against copies of itself turned round the
# colour wheel (same lightness): it asks only "is this the right hue for these furnishings?".

# %%
from sklearn.neighbors import NearestNeighbors

HARMONIC = np.deg2rad(np.array([0, 30, 120, 150, 180, 210, 240, 330]))

def v1_score(ctx, rooms, cands) -> np.ndarray:
    """V1's ranking (backend/app/pipeline/suggest.py), goal 'balanced', without the paint-match term."""
    R, K, _ = cands.shape
    out = np.zeros((R, K))
    for r_i, r in enumerate(rooms):
        fixed = [(g, ctx[g][r], ctx[g + "_area"][r]) for g in ("floor", "sofa", "bed", "table", "storage", "door")
                 if ctx[g + "_area"][r] >= 0.005 and not np.isnan(ctx[g][r][0])]
        fixed.sort(key=lambda f: -f[2])
        for k in range(K):
            w = cands[r_i, k]
            wC, wH = lch(w)
            pen, tot = 0.0, 0.0
            for _, lab, area in fixed:
                gC, gH = lch(lab)
                if gC < 12:
                    continue
                rel = (gH - wH) % (2 * np.pi)
                off = np.rad2deg(np.min(np.minimum(np.abs(rel - HARMONIC), 2 * np.pi - np.abs(rel - HARMONIC))))
                wt = area * min(gC, 40) / 40
                pen += wt * max(0.0, off - 15) / 45
                tot += wt
            pen = 0.0 if tot == 0 else min(1.0, pen / tot) * min(1.0, wC / 15)
            fails = 0
            if fixed and deltaE_ciede2000(w, fixed[0][1]) < 8:
                fails += 1
            if wC > 24:
                fails += 1
            out[r_i, k] = (1 - pen) * 0.6 + max(0.0, 1 - abs(w[0] - 79) / 25) * 0.2 - 0.5 * fails
    return out

def evaluate_all(model, df, ctx) -> dict:
    test = np.flatnonzero(df.split.to_numpy() == "validation")
    train = np.flatnonzero(df.split.to_numpy() == "training")
    rng = np.random.default_rng(123)
    nn_pop = NearestNeighbors(n_neighbors=20).fit(ctx["wall"][train])
    results = {}
    tests = {"Line-up (19 other rooms' walls)": lineups(ctx, test, rng, TEST_LINEUP - 1, 0, pool=test),
             "Hue shift (same wall, other hues)": hue_lineup(ctx["wall"][test], 9)}
    for name, cands in tests.items():
        R, K, _ = cands.shape
        pop = -nn_pop.kneighbors(cands.reshape(-1, 3))[0].mean(1).reshape(R, K)
        X = featurise(ctx, test, cands)
        results[name] = {
            "Random": {"top1": 1 / K, "top3": min(3, K) / K, "MRR": float(np.mean(1 / np.arange(1, K + 1))), "beats_share": 0.5},
            "Popularity": mrr_top1(pop),
            "V1 rules": mrr_top1(v1_score(ctx, test, cands)),
            "V2 Roomtone-Palette": mrr_top1(score_model(model, X.reshape(-1, X.shape[-1])).reshape(R, K)),
        }
    results["_meta"] = {"test_rooms": int(len(test)), "train_rooms": int(len(train))}
    (DRIVE_DIR / "palette_results.json").write_text(json.dumps(results, indent=2))
    return results

def hue_lineup(walls: np.ndarray, n: int) -> np.ndarray:
    """Real wall + n copies turned 40..320 degrees round the wheel (chroma at least 8 so hue shows)."""
    C, H = lch(walls)
    C = np.maximum(C, 8)
    turns = np.deg2rad(np.linspace(40, 320, n))
    out = np.zeros((len(walls), n + 1, 3), np.float32)
    out[:, :, 0] = walls[:, None, 0]
    hs = np.concatenate([H[:, None], H[:, None] + turns[None]], 1)
    out[:, :, 1], out[:, :, 2] = C[:, None] * np.cos(hs), C[:, None] * np.sin(hs)
    return out

if __name__ == "__main__" or IN_COLAB:
    RESULTS = evaluate_all(MODEL, ROOMS, CTX)
    for test_name, rows in RESULTS.items():
        if test_name.startswith("_"):
            continue
        print(f"\n{test_name}")
        for k, v in rows.items():
            print(f"  {k:22s} top-1 {v['top1']:6.1%}  top-3 {v['top3']:6.1%}  MRR {v['MRR']:.3f}  "
                  f"beats {v['beats_share']:6.1%} of wrong walls")

# %% [markdown]
# ## 8. See it: real wall vs the walls each method would pick
# For a few test rooms: the room's photo, its real wall, and the top-3 walls from V1's rules and
# from Roomtone-Palette, out of the same 20-wall line-up.

# %%
def show_examples(model, df, ctx, k: int = 6):
    import matplotlib.pyplot as plt
    from skimage.color import lab2rgb
    test = np.flatnonzero(df.split.to_numpy() == "validation")
    rng = np.random.default_rng(7)
    rooms = rng.choice(test, k, replace=False)
    cands = lineups(ctx, rooms, rng, TEST_LINEUP - 1, 0, pool=test)
    X = featurise(ctx, rooms, cands)
    s2 = score_model(model, X.reshape(-1, X.shape[-1])).reshape(len(rooms), -1)
    s1 = v1_score(ctx, rooms, cands)
    fig, axes = plt.subplots(k, 3, figsize=(12, 2.6 * k), gridspec_kw={"width_ratios": [2, 1, 2]})
    sw = lambda lab: np.clip(lab2rgb(np.asarray(lab, np.float64).reshape(1, -1, 3)), 0, 1)
    for row, r in enumerate(rooms):
        img = cv2.cvtColor(cv2.imread(str(ADE_ROOT / "images" / "validation" / f"{df.id.iloc[r]}.jpg")), cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(img); axes[row, 0].axis("off")
        axes[row, 1].imshow(sw(cands[row, :1]), aspect="auto"); axes[row, 1].set_title("real wall"); axes[row, 1].axis("off")
        gap = np.array([[100.0, 0, 0]])   # a white swatch between the two methods
        picks = np.concatenate([cands[row, np.argsort(-s1[row])[:3]], gap, cands[row, np.argsort(-s2[row])[:3]]])
        axes[row, 2].imshow(sw(picks), aspect="auto")
        axes[row, 2].set_title("V1 rules top-3   |   V2 Roomtone-Palette top-3"); axes[row, 2].axis("off")
    fig.tight_layout(); fig.savefig(DRIVE_DIR / "palette_examples.png", dpi=80); plt.show()

if IN_COLAB:
    show_examples(MODEL, ROOMS, CTX)

# %% [markdown]
# ## 9. Export Roomtone-Palette for the backend
# `roomtone-palette/` (and a zip) in your Drive: the weights, the feature list and the test results.
# Unzip into `backend/models/roomtone-palette/`.

# %%
def export(model, results):
    out = DRIVE_DIR / "roomtone-palette"
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "ranker.pt")
    (out / "config.json").write_text(json.dumps({
        "name": "roomtone-palette", "version": 1, "n_in": int(model.net[0].in_features),
        "context_groups": CONTEXT, "colour_space": "CIELAB, ceiling-tint and exposure corrected",
        "features": "see training/roomtone_palette.py: features()", "results": results}, indent=2))
    (out / "README.md").write_text(
        "# Roomtone-Palette (V2)\n\nA small neural network trained by the Roomtone team to score how well a "
        "wall colour suits a room (its floor, furniture, decor and light). Trained on the indoor rooms of "
        "ADE20K by ranking each room's real wall above real walls from other rooms.\n\n"
        "Credits: ADE20K (Zhou et al., MIT CSAIL), research/non-commercial terms.\n")
    shutil.make_archive(str(DRIVE_DIR / "roomtone-palette"), "zip", out)
    print("saved", out, "and", DRIVE_DIR / "roomtone-palette.zip")

if __name__ == "__main__" or IN_COLAB:
    export(MODEL, RESULTS)
