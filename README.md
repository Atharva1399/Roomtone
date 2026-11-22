# Roomtone

**One room photo in. Real, buyable paint palettes out, previewed on your own walls in your own light.**

Roomtone is an AI interior colour palette suggester (problem statement PS 39). Upload a photo of a room
and it finds the walls, floor and furniture, measures their colours and the room's light, builds wall-paint
palettes around what you're keeping, and shows each palette on your own walls. Only the paint changes:
shadows, lamp glow and furniture stay exactly as photographed. Every colour is a real paint (9,707 colours
from Sherwin-Williams, Benjamin Moore, Behr and Farrow & Ball) with its code and light reflectance value.

**Author:** Atharva Gaikwad ([@Atharva1399](https://github.com/Atharva1399))
**Team:** Atharva, Aryan, Divya and Pranav

---

## Features

- **Room reading:** our segmentation model finds walls, ceiling, floor and furniture; colours and light are measured in CIELAB.
- **Palettes:** 24 colour-harmony candidates built around what stays, filtered by five rules, ranked by our own model, shown at their 60-30-10 proportions.
- **Real previews:** the new paint is multiplied into the photo's measured light; about 0.15 s per colour, within ΔE00 0–3 of the paint on the lit wall.
- **Explanations:** why these walls, why this accent, where to use it; worded by a local language model, with templates when it's off.
- **Refine:** free-text moods ("calm coastal morning"), goals, warmer/lighter/vivid nudges, colour locks.
- **Click-to-fix:** click anything painted by mistake and SAM removes it in about 50 ms, with undo.
- **Fully local:** everything, including the language model, runs offline on one laptop with a 4 GB GPU.

## How a palette is derived

1. **Segment:** Roomtone-Surface labels every pixel with a room group.
2. **Measure:** k-means in CIELAB per surface, plus the room's brightness and warm/cool light.
3. **Harmonise:** 8 colour-wheel schemes around the furnishings that stay, × 3 lightness levels = 24 candidates.
4. **Assign roles:** 60% walls, 30% what you keep, 10% accent; a tinted white for the ceiling.
5. **Check & rank:** 5 rules reject palettes that would fail in the room; Roomtone-Palette ranks the rest.
6. **Snap to paint:** nearest of 9,707 real paints by CIEDE2000.
7. **Preview:** paint × measured shading × light tint, inside a crisp wall mask.
8. **Explain:** the measured facts, in plain words.

## Models

| Model | Job | Size | Source |
|---|---|---|---|
| **Roomtone-Surface** | Wall, floor, ceiling and furniture masks | 64M | **Trained by us:** SegFormer-B4 fine-tuned on 9,263 indoor ADE20K rooms ([notebook](training/roomtone_surface.ipynb)) |
| **Roomtone-Palette** | Ranks the wall colours that suit a room | ~56k | **Trained by us** from scratch on 10,193 measured rooms ([notebook](training/roomtone_palette.ipynb)) |
| SAM ViT-B | Click-to-fix object masks | 94M | `facebook/sam-vit-base`, pretrained |
| gemma3:4b | Style read, explanations, free-text moods | 4B | Google Gemma 3 via Ollama, pretrained; never picks colours |

### Results (unseen ADE20K indoor test rooms)

**Roomtone-Surface vs the original SegFormer-B4** (933 rooms, 11 room groups)

| Metric | Original | Roomtone-Surface |
|---|---|---|
| mIoU | 74.01% | **74.47%** |
| Pixel accuracy | 87.85% | **87.96%** |
| Wall IoU | 81.62% | **81.77%** |
| Wall precision | 88.80% | **89.28%** |
| Wall recall | 90.99% | 90.62% |
| Wall F1 | 89.88% | **89.95%** |
| Macro F1 | 84.46% | **84.84%** |
| Wall leak (paint on furniture) | 6.68% | **5.94%** (11% less) |
| Time per photo | 0.088 s | 0.088 s |

**Roomtone-Palette** (932 rooms): with the real wall hidden among 20 real walls from other rooms, it ranks the
real one first in **22.6%** of rooms, in the top 3 in **45.1%**, with MRR **0.401**, and above **76.1%** of the wrong walls.

**Speed** (RTX 3050 Ti laptop): colour preview ~0.15 s · new palettes 0.29 s · click-to-fix ~50 ms · new photo ~2 s once ·
language model 5–37 s in the background. **Tests:** 62 / 62 passing.

## Tech stack

- **Backend:** Python 3.11, FastAPI, PyTorch, Hugging Face Transformers, scikit-learn, scikit-image, OpenCV
- **Frontend:** React 19, Vite, TypeScript
- **Language model:** Ollama (`gemma3:4b`), with Hugging Face and Claude API as optional fallbacks, then templates
- **Training:** Google Colab notebooks (free T4 GPU)

## Project structure

```
backend/     FastAPI app, colour pipeline, models, tests
  app/pipeline/   segmentation, palette measurement, suggestion, recolour, ranker, click-to-fix
  data/paints/    the four paint colour books (JSON)
  models/         trained weights (not in git; see below)
frontend/    React + Vite studio
training/    Colab notebooks for Roomtone-Surface and Roomtone-Palette
```

## Run it

**Backend** (Python 3.11 with a CUDA GPU; CPU works but is slower):

```bash
cd backend
pip install "torch==2.14.1" torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server forwards `/api` to the backend on port 8000.

**Our trained models:** the weights are too large for git. Download them from the training notebooks' output
(Google Drive) and unzip into:

```
backend/models/roomtone-surface-b4/   config.json, model.safetensors, preprocessor_config.json
backend/models/roomtone-palette/      ranker.pt, config.json
```

When a folder is present the backend uses our model; otherwise it falls back to the original SegFormer-B4 or
the rule-based ranking. `GET /api/health` shows which is active.

**Language model (optional):** install [Ollama](https://ollama.com) and run `ollama pull gemma3:4b`. Without it,
explanations come from templates and everything else works. Settings: `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`.

**Tests:**

```bash
cd backend
pytest -q
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | GPU, active models, paint count |
| `POST /api/rooms` | Upload a photo: segmentation and colour analysis |
| `GET /api/rooms/{id}` · `/original` · `/segmentation` | Room data, photo, mask overlay |
| `POST /api/rooms/{id}/style` | The language model's style read |
| `POST /api/rooms/{id}/palettes` | Palettes (goal, mood, locks, nudges, brands, explanations) |
| `POST /api/rooms/{id}/preview` | Repainted photo for any wall/ceiling colour |
| `POST /api/rooms/{id}/mask` · `/mask/undo` · `/mask/reset` · `GET /mask` | Click-to-fix |
| `POST /api/colors/adjust` · `GET /api/paints/match` | Colour nudges, nearest real paints |

## Versions

- **v1.0** (tag): off-the-shelf models (SegFormer-B4, SAM, gemma3:4b) and rule-based palette ranking.
- **main (V2):** our own Roomtone-Surface and Roomtone-Palette, plus crisper paint edges.

## Credits

- **SegFormer:** Xie et al., NeurIPS 2021 (NVIDIA); Roomtone-Surface is a fine-tune of SegFormer-B4.
- **ADE20K:** Zhou et al., CVPR 2017 (MIT CSAIL); training and test data for both our models.
- **Segment Anything:** Kirillov et al., ICCV 2023 (Meta).
- **Gemma 3:** Google.
- **Paint colour books:** [jpederson/colornerd](https://github.com/jpederson/colornerd).

SegFormer and ADE20K are released under research / non-commercial terms, so Roomtone is a research and
student project. Check those licences, and the paint brands, before any commercial use.
