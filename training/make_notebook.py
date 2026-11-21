"""Turn roomtone_surface.py (# %% cells) into roomtone_surface.ipynb for Google Colab.

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
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells((HERE / "roomtone_surface.py").read_text(encoding="utf-8")),
    }
    (HERE / "roomtone_surface.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", HERE / "roomtone_surface.ipynb", f"({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
