"""Turn the training scripts (# %% cells) into Colab notebooks:
roomtone_surface.py -> roomtone_surface.ipynb, roomtone_palette.py -> roomtone_palette.ipynb.

    python training/make_notebook.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def cells(source: str) -> list[dict]:
    out = []
    for block in re.split(r"^# %%", source, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            out.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": body.splitlines(keepends=True)})
    return out


def main():
    for name in ("roomtone_surface", "roomtone_palette"):
        nb = {
            "nbformat": 4, "nbformat_minor": 5,
            "metadata": {
                "accelerator": "GPU",
                "colab": {"provenance": [], "gpuType": "T4"},
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "language_info": {"name": "python"},
            },
            "cells": cells((HERE / f"{name}.py").read_text(encoding="utf-8")),
        }
        (HERE / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", HERE / f"{name}.ipynb", f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
