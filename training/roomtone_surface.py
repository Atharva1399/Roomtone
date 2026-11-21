# %% [markdown]
# # Roomtone-Surface (V2): fine-tuning SegFormer-B4 for paintable room surfaces
#
# Roomtone **V1** segments rooms with the off-the-shelf `nvidia/segformer-b4-finetuned-ade-512-512`
# (150 ADE20K classes, folded into room groups afterwards). **V2** replaces it with
# **Roomtone-Surface**: the same SegFormer-B4, fine-tuned by us on indoor rooms only and optimised
# for Roomtone's own 11 room groups.
#
# What we change compared with the V1 model:
# 1. **Indoor rooms only**: ADE20K images filtered to rooms (wall + ceiling + floor cover ≥ 25 %, no sky).
# 2. **A Roomtone loss**: the model is trained on Roomtone's groups
#    (`other, wall, ceiling, floor, window, door, sofa, bed, table, storage, decor`), scored by summing each
#    group's class probabilities, exactly as the backend reads it. A smaller loss on the 150 fine classes
#    keeps the details. Training starts from V1 itself, so V2 begins at V1's accuracy and only has to improve,
#    and it drops into the backend with no code change.
# 3. **Paint augmentation**: half of the training walls are repainted a random colour under the photo's own
#    light (the V1 preview formula), so the model learns "wall" from shape and context, not from today's paint.
#
# **Run on Google Colab with a T4 GPU** (Runtime › Change runtime type › T4 GPU), top to bottom.
# Training saves to Google Drive every 500 steps; after a disconnect, run all cells again and it resumes.
#
# Credits: SegFormer, Xie et al., NeurIPS 2021 (NVIDIA, research/non-commercial licence).
# ADE20K, Zhou et al., MIT CSAIL (research/non-commercial terms).

# %% [markdown]
# ## 1. Setup

# %%
import sys, subprocess
import torch

IN_COLAB = "google.colab" in sys.modules
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
if IN_COLAB:
    assert torch.cuda.is_available(), "No GPU: Runtime › Change runtime type › T4 GPU"
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers>=4.40",
                    "opencv-python-headless", "safetensors"], check=True)

# %%
from pathlib import Path

DRIVE_DIR = Path("/content/drive/MyDrive/roomtone_surface")   # checkpoints, index, results
DATA_DIR = Path("/content/data")                              # fast local disk for the images
BASE_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"      # the V1 model
MODEL_NAME = "roomtone-surface-b4"

CROP = 512                  # training crop, as SegFormer-B4 was trained on ADE20K
BATCH, ACCUM = 4, 2         # 4 images per step on a 16 GB T4, x2 accumulation = effective batch 8
STEPS = 12_000              # optimiser steps; ~1.2–1.5 s each on a T4 (estimate) ≈ 4–5 h
WARMUP = 1_000
LR = 6e-5                   # encoder; the decode head gets 10x (SegFormer's recipe)
FINE_LOSS = 0.5             # weight of the 150-class loss next to the Roomtone-group loss
WEIGHT_DECAY = 0.01
PAINT_AUG_P = 0.5           # share of training images whose walls get repainted
EVAL_EVERY, EVAL_IMAGES = 1_000, 300
SAVE_EVERY = 500
EVAL_SIDE = 768             # longest side at test time, same as the V1 backend
NUM_WORKERS = 2             # free Colab has 2 CPU cores
SEED = 0

# %%
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 2. Download ADE20K (about 1 GB)
# The zip is kept in your Drive after the first download, so later sessions copy it instead (faster).

# %%
import shutil, urllib.request, zipfile

ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ADE_ROOT = DATA_DIR / "ADEChallengeData2016"

def fetch_ade20k():
    if (ADE_ROOT / "annotations").exists():
        return
    local, cached = DATA_DIR / "ADE20K.zip", DRIVE_DIR / "ADE20K.zip"
    if cached.exists():
        print("copying the cached zip from Drive…")
        shutil.copy(cached, local)
    else:
        print("downloading ADE20K…")
        urllib.request.urlretrieve(ADE_URL, local)
        shutil.copy(local, cached)
    with zipfile.ZipFile(local) as z:
        z.extractall(DATA_DIR)
    local.unlink()

if IN_COLAB:
    fetch_ade20k()
    for split in ("training", "validation"):
        print(split, len(list((ADE_ROOT / "images" / split).glob("*.jpg"))), "images")

# %% [markdown]
# ## 3. Roomtone's classes
# The groups are copied from the V1 backend (`backend/app/pipeline/segmentation.py`); keep the two in sync.
# Every ADE20K class not listed below belongs to `other`; unlabelled pixels are ignored.
# The model keeps ADE20K's 150 outputs; a group's probability is the sum of its classes' probabilities.

# %%
import numpy as np
from transformers import AutoConfig

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
CLASSES = ["other", *ROOM_GROUPS]          # 11 classes; index = Roomtone-Surface label
WALL, CEILING, FLOOR = (CLASSES.index(c) for c in ("wall", "ceiling", "floor"))
NOT_PAINT = [CLASSES.index(c) for c in ("window", "door", "sofa", "bed", "table", "storage", "decor")]
IGNORE = 255

def class_members(ade_id2label: dict) -> list[list[int]]:
    """For each Roomtone class, the ADE model ids (0..149) it merges; `other` gets the rest."""
    name_to_id = {n.strip().lower(): int(i) for i, n in ade_id2label.items()}
    members = [[name_to_id[n] for n in names if n in name_to_id] for names in ROOM_GROUPS.values()]
    used = {i for m in members for i in m}
    return [[i for i in sorted(name_to_id.values()) if i not in used], *members]

def build_lut(members: list[list[int]]) -> np.ndarray:
    """ADE annotation value (0 = unlabelled, 1..150 = class id + 1) -> Roomtone class."""
    lut = np.zeros(256, np.uint8)
    lut[0] = IGNORE
    for c, ids in enumerate(members):
        lut[np.array(ids, int) + 1] = c
    return lut

ADE_ID2LABEL = {int(k): v for k, v in AutoConfig.from_pretrained(BASE_MODEL).id2label.items()}
MEMBERS = class_members(ADE_ID2LABEL)
LUT = build_lut(MEMBERS)                        # ADE annotation value -> group
TO_MODEL_ID = np.full(256, IGNORE, np.uint8)    # ADE annotation value -> model class id (0..149)
TO_MODEL_ID[1:151] = np.arange(150)
GROUP_OF = torch.from_numpy(LUT[1:151].astype(np.int64))   # model class id -> group
ID_TO_GROUP = np.full(256, IGNORE, np.uint8)    # model class id (or IGNORE) -> group (or IGNORE)
ID_TO_GROUP[:150] = LUT[1:151]
_NAME_TO_VALUE = {n.strip().lower(): i + 1 for i, n in ADE_ID2LABEL.items()}
SKY_VALUE = _NAME_TO_VALUE["sky"]
WALL_ID = _NAME_TO_VALUE["wall"] - 1
for c, ids in zip(CLASSES, MEMBERS):
    names = [ADE_ID2LABEL[i].strip() for i in ids]
    print(f"{c:8s} <- {len(ids):3d} ADE classes: {', '.join(names[:8])}{' …' if len(names) > 8 else ''}")

# %% [markdown]
# ## 4. Keep only indoor rooms
# A photo counts as a room when wall + ceiling + floor cover at least 25 % of its labelled pixels,
# walls alone at least 5 %, and sky under 1 %. The list is saved to Drive, so this runs once.

# %%
import json
from multiprocessing import Pool
from PIL import Image

def room_stats(path: str):
    ann = np.asarray(Image.open(path))[::4, ::4]   # every 4th pixel is plenty for coverage stats
    lab = LUT[ann]
    valid = lab != IGNORE
    n = int(valid.sum())
    if n == 0:
        return None
    frac = np.bincount(lab[valid], minlength=len(CLASSES)) / n
    sky = float((ann == SKY_VALUE).sum()) / n
    is_room = frac[WALL] + frac[CEILING] + frac[FLOOR] >= 0.25 and frac[WALL] >= 0.05 and sky < 0.01
    return Path(path).stem if is_room else None

def indoor_ids(split: str) -> list[str]:
    paths = sorted(str(p) for p in (ADE_ROOT / "annotations" / split).glob("*.png"))
    with Pool(4) as pool:
        return [s for s in pool.map(room_stats, paths, chunksize=64) if s]

def load_index() -> dict[str, list[str]]:
    index_file = DRIVE_DIR / "indoor_index.json"
    if index_file.exists():
        return json.loads(index_file.read_text())
    index = {split: indoor_ids(split) for split in ("training", "validation")}
    index_file.write_text(json.dumps(index))
    return index

if IN_COLAB:
    INDEX = load_index()
    print({k: len(v) for k, v in INDEX.items()}, "indoor images")

# %% [markdown]
# ## 5. Data pipeline and paint augmentation

# %%
import colorsys
import cv2
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)

def random_paint(rng) -> np.ndarray:
    """A plausible wall paint: mostly soft colours, sometimes deep or saturated ones."""
    h = rng.random()
    s = rng.uniform(0, 1) ** 2 * 0.75
    v = rng.uniform(0.3, 0.97)
    return np.array(colorsys.hsv_to_rgb(h, s, v), np.float32)

def paint_walls(img: np.ndarray, lab: np.ndarray, rng) -> np.ndarray:
    """Repaint the walls under the photo's own light: paint × shading, in linear RGB."""
    wall = lab == WALL_ID
    if wall.sum() < 1_000:
        return img
    lin = srgb_to_linear(img)
    lum = lin @ LUMA
    shade = np.clip(lum / (np.median(lum[wall]) + 1e-4), 0, 3)
    painted = linear_to_srgb(np.clip(srgb_to_linear(random_paint(rng)) * shade[..., None], 0, 1))
    alpha = cv2.GaussianBlur(wall.astype(np.float32), (0, 0), 1.5)[..., None]
    return (alpha * painted + (1 - alpha) * img).astype(np.float32)

def photometric(img: np.ndarray, rng) -> np.ndarray:
    """Brightness, contrast, saturation, hue and warm/cool light changes."""
    img = img * rng.uniform(0.7, 1.3)
    img = (img - img.mean()) * rng.uniform(0.75, 1.25) + img.mean()
    img = img * np.array([rng.uniform(0.9, 1.1), 1.0, rng.uniform(0.9, 1.1)], np.float32)  # light colour
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8)) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def train_sample(img: np.ndarray, lab: np.ndarray, rng):
    """Random scale (0.5–2x), crop with no class over 75 %, flip, paint, photometric."""
    h, w = lab.shape
    s = CROP * rng.uniform(0.5, 2.0) / min(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ph, pw = max(CROP - nh, 0), max(CROP - nw, 0)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        lab = cv2.copyMakeBorder(lab, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE)
    for _ in range(10):
        y = rng.integers(0, lab.shape[0] - CROP + 1)
        x = rng.integers(0, lab.shape[1] - CROP + 1)
        crop = lab[y:y + CROP, x:x + CROP]
        v = crop[crop != IGNORE]
        if v.size and np.bincount(v).max() / v.size < 0.75:
            break
    img, lab = img[y:y + CROP, x:x + CROP], lab[y:y + CROP, x:x + CROP]
    if rng.random() < 0.5:
        img, lab = img[:, ::-1], lab[:, ::-1]
    img = img.astype(np.float32) / 255
    if rng.random() < PAINT_AUG_P:
        img = paint_walls(img, lab, rng)
    return photometric(img, rng), lab

def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((np.clip(img, 0, 1) - MEAN) / STD).transpose(2, 0, 1).copy())

def eval_size(h: int, w: int) -> tuple[int, int]:
    """Longest side EVAL_SIDE, both sides a multiple of 32, as in the V1 backend."""
    s = EVAL_SIDE / max(h, w)
    return max(32, round(h * s / 32) * 32), max(32, round(w * s / 32) * 32)

class ADEIndoor(Dataset):
    def __init__(self, ids: list[str], split: str, train: bool):
        self.ids, self.split, self.train = ids, split, train

    def __len__(self):
        return len(self.ids)

    def load(self, i: int):
        stem = self.ids[i]
        img = cv2.cvtColor(cv2.imread(str(ADE_ROOT / "images" / self.split / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        ann = cv2.imread(str(ADE_ROOT / "annotations" / self.split / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        return img, TO_MODEL_ID[ann]        # model class ids, IGNORE where unlabelled

    def __getitem__(self, i: int):
        img, lab = self.load(i)
        if self.train:
            img, lab = train_sample(img, lab, np.random.default_rng())
            return to_tensor(img), torch.from_numpy(lab.astype(np.int64))
        h, w = eval_size(*lab.shape)
        small = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        return to_tensor(small), torch.from_numpy(lab.astype(np.int64))

# %% [markdown]
# ## 6. Model and the Roomtone loss
# The model starts as V1 (all 150 outputs kept). The loss scores Roomtone's groups: each group's
# probability is the sum of its member classes' probabilities, the same way the backend reads the model.

# %%
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

def build_model() -> SegformerForSemanticSegmentation:
    return SegformerForSemanticSegmentation.from_pretrained(BASE_MODEL)

def group_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """(N, 150, H, W) logits -> (N, 11, H, W) log-probabilities of Roomtone's groups."""
    probs = logits.float().softmax(1)
    groups = torch.zeros(probs.shape[0], len(CLASSES), *probs.shape[2:], device=probs.device)
    groups.index_add_(1, GROUP_OF.to(probs.device), probs)
    return groups.clamp_min(1e-8).log()

def roomtone_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Group cross-entropy + FINE_LOSS x the usual 150-class cross-entropy."""
    logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    valid = labels != IGNORE
    group_labels = torch.where(valid, GROUP_OF.to(labels.device)[labels.clamp(max=149)],
                               torch.full_like(labels, IGNORE))
    group_loss = F.nll_loss(group_log_probs(logits), group_labels, ignore_index=IGNORE)
    fine_loss = F.cross_entropy(logits, labels, ignore_index=IGNORE)
    return group_loss + FINE_LOSS * fine_loss

def predict_groups(model, x: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Roomtone group per pixel at the label's size, for any 150-class ADE20K SegFormer."""
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(pixel_values=x).logits
    logits = F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)
    return group_log_probs(logits).argmax(1)[0].cpu().numpy()

# %% [markdown]
# ## 7. Metrics
# Per-class IoU (overlap ÷ union), mIoU, pixel accuracy, and **wall leak**: the share of pixels
# predicted as wall that are really furniture, doors, windows or decor, i.e. paint that would land
# on the wrong thing.

# %%
class Confusion:
    def __init__(self, n: int = len(CLASSES)):
        self.n, self.m = n, np.zeros((n, n), np.int64)   # rows = truth, columns = prediction

    def add(self, pred: np.ndarray, truth: np.ndarray):
        v = truth != IGNORE
        self.m += np.bincount(truth[v].astype(np.int64) * self.n + pred[v],
                              minlength=self.n * self.n).reshape(self.n, self.n)

    def report(self) -> dict:
        tp = np.diag(self.m).astype(float)
        union = self.m.sum(0) + self.m.sum(1) - tp
        iou = np.divide(tp, union, out=np.full(self.n, np.nan), where=union > 0)
        pred_wall = self.m[:, WALL].sum()
        return {
            "mIoU": float(np.nanmean(iou)),
            "pixel_acc": float(tp.sum() / max(self.m.sum(), 1)),
            "wall_leak": float(self.m[NOT_PAINT, WALL].sum() / max(pred_wall, 1)),
            **{f"IoU_{c}": float(x) for c, x in zip(CLASSES, iou)},
        }

@torch.inference_mode()
def evaluate(model, ds: ADEIndoor, limit: int | None = None) -> dict:
    model.eval()
    conf = Confusion()
    for i in range(min(len(ds), limit or len(ds))):
        x, y = ds[i]
        conf.add(predict_groups(model, x[None].cuda(), tuple(y.shape)), ID_TO_GROUP[y.numpy()])
    model.train()
    return conf.report()

# %% [markdown]
# ## 8. Train (resumes from Drive automatically)

# %%
import math, time

CKPT = DRIVE_DIR / "last.pt"
BEST = DRIVE_DIR / "best.pt"

def lr_factor(step: int) -> float:
    if step < WARMUP:
        return (step + 1) / WARMUP
    return max(0.0, 1 - (step - WARMUP) / max(1, STEPS - WARMUP))   # poly, power 1

def make_optimizer(model):
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    no_decay = [p for n, p in model.named_parameters()
                if not n.startswith("decode_head") and (p.ndim == 1 or "norm" in n)]
    decay = [p for n, p in model.named_parameters()
             if not n.startswith("decode_head") and not (p.ndim == 1 or "norm" in n)]
    return torch.optim.AdamW([
        {"params": decay, "lr": LR, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "lr": LR, "weight_decay": 0.0},
        {"params": head, "lr": LR * 10, "weight_decay": WEIGHT_DECAY},
    ])

def save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)

def score(r: dict) -> float:
    """Model selection: equal weight on overall mIoU and on the wall, the surface we repaint."""
    return (r["mIoU"] + r["IoU_wall"]) / 2

def train():
    torch.manual_seed(SEED)
    train_ds = ADEIndoor(INDEX["training"], "training", train=True)
    val_ds = ADEIndoor(INDEX["validation"][:EVAL_IMAGES], "validation", train=False)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    model = build_model().cuda().train()
    opt = make_optimizer(model)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    step, best, history = 0, -1.0, []

    if CKPT.exists():
        ck = torch.load(CKPT, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        step, best, history = ck["step"], ck["best"], ck["history"]
        print(f"resumed at step {step} (best score {best:.4f})")
    else:
        r = evaluate(model, val_ds)
        print(f"step 0 (= V1): mIoU {r['mIoU']:.4f}  wall {r['IoU_wall']:.4f}  wall leak {r['wall_leak']:.4f}")

    batches, t0, running = iter(loader), time.time(), 0.0
    while step < STEPS:
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            try:
                x, y = next(batches)
            except StopIteration:
                batches = iter(loader)
                x, y = next(batches)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(pixel_values=x.cuda(non_blocking=True)).logits
            loss = roomtone_loss(logits, y.cuda(non_blocking=True)) / ACCUM
            scaler.scale(loss).backward()
            running += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1

        if step % 50 == 0:
            rate = (time.time() - t0) / 50
            eta = (STEPS - step) * rate / 3600
            print(f"step {step}/{STEPS}  loss {running / 50:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{rate:.2f} s/step  ETA {eta:.1f} h")
            t0, running = time.time(), 0.0
        if step % EVAL_EVERY == 0 or step == STEPS:
            r = evaluate(model, val_ds)
            history.append({"step": step, **r})
            print(f"  eval: mIoU {r['mIoU']:.4f}  wall {r['IoU_wall']:.4f}  ceiling {r['IoU_ceiling']:.4f}  "
                  f"wall leak {r['wall_leak']:.4f}")
            if score(r) > best:
                best = score(r)
                save_atomic({"model": model.state_dict(), "step": step, "eval": r}, BEST)
                print("  new best, saved")
        if step % SAVE_EVERY == 0 or step == STEPS:
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "scaler": scaler.state_dict(), "step": step, "best": best, "history": history}, CKPT)
    print("training finished")

# %%
if IN_COLAB:
    train()

# %% [markdown]
# ## 9. Test: V1 SegFormer-B4 vs Mask2Former vs Roomtone-Surface (V2)
# All three on every indoor ADE20K validation image, scored on Roomtone's 11 groups.
# SegFormer V1 and V2 are read the way the backend reads them (summed group probabilities);
# Mask2Former's per-pixel class is mapped to its group.

# %%
import statistics
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

MASK2FORMER_CANDIDATES = ["facebook/mask2former-swin-base-IN21k-ade-semantic",
                          "facebook/mask2former-swin-base-ade-semantic"]

def load_v2() -> SegformerForSemanticSegmentation:
    model = build_model()
    model.load_state_dict(torch.load(BEST, map_location="cpu", weights_only=False)["model"])
    return model.cuda().eval()

@torch.inference_mode()
def compare(n: int | None = None) -> dict:
    val_ds = ADEIndoor(INDEX["validation"][:n] if n else INDEX["validation"], "validation", train=False)
    v1 = SegformerForSemanticSegmentation.from_pretrained(BASE_MODEL).cuda().eval()
    v2 = load_v2()
    m2f_name = next(m for m in MASK2FORMER_CANDIDATES if _exists(m))
    m2f_proc = AutoImageProcessor.from_pretrained(m2f_name)
    m2f = Mask2FormerForUniversalSegmentation.from_pretrained(m2f_name).cuda().eval()
    m2f_members = class_members({int(k): v for k, v in m2f.config.id2label.items()})

    conf = {k: Confusion() for k in ("V1 SegFormer-B4", "Mask2Former Swin-B", "V2 Roomtone-Surface")}
    times = {k: [] for k in conf}
    for i in range(len(val_ds)):
        x, y = val_ds[i]
        stem = val_ds.ids[i]
        x = x[None].cuda()
        size = tuple(y.shape)
        truth = ID_TO_GROUP[y.numpy()]

        def run(key, fn):
            torch.cuda.synchronize(); t = time.perf_counter()
            pred = fn()
            torch.cuda.synchronize(); times[key].append(time.perf_counter() - t)
            conf[key].add(pred, truth)

        def seg_v1():
            return predict_groups(v1, x, size)

        def seg_v2():
            return predict_groups(v2, x, size)

        def seg_m2f():
            img = cv2.cvtColor(cv2.imread(str(ADE_ROOT / "images" / "validation" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
            inputs = m2f_proc(images=img, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = m2f(**inputs)
            ade = m2f_proc.post_process_semantic_segmentation(out, target_sizes=[size])[0].cpu().numpy()
            return build_lut(m2f_members)[ade + 1]

        run("V1 SegFormer-B4", seg_v1)
        run("Mask2Former Swin-B", seg_m2f)
        run("V2 Roomtone-Surface", seg_v2)
        if i % 200 == 0:
            print(f"{i}/{len(val_ds)}")

    results = {k: {**c.report(), "median_s": statistics.median(times[k][5:] or times[k])}
               for k, c in conf.items()}
    results["_meta"] = {"images": len(val_ds), "mask2former": m2f_name, "eval_side": EVAL_SIDE}
    (DRIVE_DIR / "results.json").write_text(json.dumps(results, indent=2))
    return results

def _exists(repo: str) -> bool:
    try:
        AutoConfig.from_pretrained(repo)
        return True
    except Exception:
        return False

# %%
if IN_COLAB:
    import pandas as pd
    RESULTS = compare()
    cols = ["mIoU", "IoU_wall", "IoU_ceiling", "IoU_floor", "wall_leak", "pixel_acc", "median_s"]
    table = pd.DataFrame({k: v for k, v in RESULTS.items() if not k.startswith("_")}).T[cols]
    display(table.style.format({c: "{:.2%}" for c in cols[:-1]} | {"median_s": "{:.3f} s"}))

# %% [markdown]
# ## 10. See it: the same photos through V1 and V2

# %%
COLOURS = np.array([(200, 200, 200), (230, 80, 80), (240, 200, 60), (90, 160, 90), (80, 170, 230),
                    (150, 100, 60), (180, 90, 200), (240, 140, 180), (120, 120, 220), (60, 200, 190),
                    (250, 150, 50)], np.uint8)

def overlay(img: np.ndarray, lab: np.ndarray) -> np.ndarray:
    col = COLOURS[np.minimum(lab, len(COLOURS) - 1)]
    out = (0.5 * img + 0.5 * col).astype(np.uint8)
    out[lab == IGNORE] = img[lab == IGNORE]
    return out

@torch.inference_mode()
def show_examples(k: int = 4):
    import matplotlib.pyplot as plt
    ds = ADEIndoor(INDEX["validation"], "validation", train=False)
    v1 = SegformerForSemanticSegmentation.from_pretrained(BASE_MODEL).cuda().eval()
    v2 = load_v2()
    fig, axes = plt.subplots(k, 3, figsize=(15, 4 * k))
    for row, i in enumerate(np.linspace(0, len(ds) - 1, k).astype(int)):
        x, y = ds[i]
        img, _ = ds.load(i)
        size = tuple(y.shape)
        p1 = predict_groups(v1, x[None].cuda(), size)
        p2 = predict_groups(v2, x[None].cuda(), size)
        truth = ID_TO_GROUP[y.numpy()]
        for ax, lab, title in zip(axes[row], (truth, p1, p2), ("Ground truth", "V1 SegFormer-B4", "V2 Roomtone-Surface")):
            ax.imshow(overlay(img, lab)); ax.set_title(title); ax.axis("off")
    fig.tight_layout()
    fig.savefig(DRIVE_DIR / "examples.png", dpi=80)
    plt.show()

if IN_COLAB:
    show_examples()

# %% [markdown]
# ## 11. Export Roomtone-Surface for the Roomtone backend
# Saves the best checkpoint as a Hugging Face model folder (fp16, about 130 MB) with a model card,
# zipped in your Drive as `roomtone-surface-b4.zip`. Download it and unzip it into
# `backend/models/roomtone-surface-b4/`. It has the same 150 outputs as V1, so the backend only needs
# its model path changed.

# %%
from transformers import SegformerImageProcessor

def export():
    out = DRIVE_DIR / MODEL_NAME
    model = load_v2().half().cpu()
    model.save_pretrained(out, safe_serialization=True)
    SegformerImageProcessor.from_pretrained(BASE_MODEL).save_pretrained(out)
    best = torch.load(BEST, map_location="cpu", weights_only=False)
    results = json.loads((DRIVE_DIR / "results.json").read_text()) if (DRIVE_DIR / "results.json").exists() else {}
    (out / "README.md").write_text(
        "# Roomtone-Surface (SegFormer-B4, fine-tuned)\n\n"
        "Roomtone V2 segmentation model for paintable room surfaces. It keeps ADE20K's 150 outputs and is "
        f"optimised for Roomtone's 11 groups ({', '.join(CLASSES)}): a group's probability is the sum of "
        "its classes' probabilities, as in `backend/app/pipeline/segmentation.py`.\n\n"
        f"Fine-tuned by the Roomtone team from `{BASE_MODEL}` on the indoor images of ADE20K "
        f"({len(INDEX['training'])} training photos), with a Roomtone group loss and paint augmentation. "
        f"Best checkpoint: step {best['step']}.\n\n"
        f"## Results (indoor ADE20K validation, {len(INDEX['validation'])} photos)\n\n"
        f"```json\n{json.dumps(results, indent=2)}\n```\n\n"
        "## Credits and licences\n\n"
        "- SegFormer: Xie et al., NeurIPS 2021, NVIDIA (research/non-commercial licence).\n"
        "- ADE20K: Zhou et al., MIT CSAIL (research/non-commercial terms).\n"
    )
    shutil.make_archive(str(DRIVE_DIR / MODEL_NAME), "zip", out)
    print("saved", out, "and", DRIVE_DIR / f"{MODEL_NAME}.zip")

if IN_COLAB:
    export()
