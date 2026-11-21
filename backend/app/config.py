"""Central settings for the backend pipeline."""

import os
from pathlib import Path

# V1: SegFormer-B4 fine-tuned on ADE20K (150 indoor/outdoor classes). Fits in 4 GB VRAM.
# V2: Roomtone-Surface, our fine-tune of it for rooms (training/), used when its folder exists.
# SEGMENTATION_MODEL in the environment overrides both.
V1_SEGMENTATION_MODEL = "nvidia/segformer-b4-finetuned-ade-512-512"
V2_SEGMENTATION_MODEL = Path(__file__).resolve().parents[1] / "models" / "roomtone-surface-b4"
SEGMENTATION_MODEL = os.environ.get("SEGMENTATION_MODEL") or (
    str(V2_SEGMENTATION_MODEL) if (V2_SEGMENTATION_MODEL / "config.json").exists() else V1_SEGMENTATION_MODEL
)

# Longest image side we work at. Larger photos are downscaled on load.
MAX_IMAGE_SIDE = 1024

# Longest side fed to the segmentation model (aspect ratio kept, rounded to /32).
# The model's default 512x512 squashes the photo and gives blobby edges.
SEGMENTATION_SIDE = 768

# Pixels sampled per region for k-means (keeps clustering fast on big images).
MAX_SAMPLE_PIXELS = 20_000

# API: uploads larger than this are rejected; rooms kept in memory (least recently used dropped).
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_CACHED_ROOMS = 20

# Origins allowed to call the API from a browser (the future frontend's dev servers).
CORS_ORIGINS = ["http://localhost:3000", "http://localhost:5173",
                "http://127.0.0.1:3000", "http://127.0.0.1:5173"]

# A segmentation region must cover at least this share of the image to be reported.
MIN_REGION_FRACTION = 0.005
