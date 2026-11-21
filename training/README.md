# Training: Roomtone V2 models

Roomtone **V1** uses off-the-shelf models (SegFormer-B4, SAM ViT-B, gemma3:4b; git tag `v1.0`).
**V2** replaces them with models we fine-tune ourselves. The first is **Roomtone-Surface**.

## Roomtone-Surface

SegFormer-B4 fine-tuned on the indoor rooms of ADE20K, optimised for Roomtone's 11 room groups
(`other, wall, ceiling, floor, window, door, sofa, bed, table, storage, decor`) with a group loss,
plus paint augmentation (walls repainted a random colour under the photo's own light). It starts
from the V1 model and keeps its 150 outputs, so the backend only needs its model path changed.

1. Open `roomtone_surface.ipynb` in Google Colab (File › Upload notebook) and pick a T4 GPU
   (Runtime › Change runtime type).
2. Run all cells. The notebook downloads ADE20K (about 1 GB), keeps the indoor rooms, trains
   (about 4–5 h on a T4; it saves to Google Drive and resumes after a disconnect), tests V1 vs
   Mask2Former vs V2, and exports `roomtone-surface-b4.zip` to your Drive.
3. Unzip it into `backend/models/roomtone-surface-b4/`.

## Roomtone-Palette

A small neural network that scores how well a wall colour suits a room (its floor, furniture, decor
and light). In the app, V1's rule checks still filter palettes; Roomtone-Palette re-ranks the ones
that pass.

The dataset is built automatically: the notebook measures every indoor ADE20K room with ADE20K's own
labels (colour cast and exposure corrected) and saves it as `palette_rooms.csv`. The model learns to
pick each room's real wall colour out of real wall colours from other rooms, and is tested against
random, popularity and V1's rules on rooms it never saw.

1. Open `roomtone_palette.ipynb` in Colab and run all cells (a GPU is optional). It reuses the
   ADE20K zip and indoor list in `MyDrive/roomtone_surface`, so run it with the same Google account.
2. It saves `palette_rooms.csv`, `palette_results.json`, `palette_examples.png` and
   `roomtone-palette.zip` to `MyDrive/roomtone_palette`.
3. Unzip `roomtone-palette.zip` into `backend/models/roomtone-palette/`.

## Editing the notebooks

Edit the `.py` sources, then regenerate the notebooks:

```
python training/make_notebook.py
```

Credits: SegFormer (Xie et al., NeurIPS 2021, NVIDIA) and ADE20K (Zhou et al., MIT CSAIL), both
under research/non-commercial terms. Roomtone-Surface is a fine-tune of their work, not a model
built from scratch.
