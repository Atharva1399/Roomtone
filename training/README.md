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

Edit `roomtone_surface.py` (the source), then regenerate the notebook:

```
python training/make_notebook.py
```

Credits: SegFormer (Xie et al., NeurIPS 2021, NVIDIA) and ADE20K (Zhou et al., MIT CSAIL), both
under research/non-commercial terms. Roomtone-Surface is a fine-tune of their work, not a model
built from scratch.
