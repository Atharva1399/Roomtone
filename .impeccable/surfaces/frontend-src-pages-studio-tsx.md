---
version: 1
slug: "frontend-src-pages-studio-tsx"
primary_target: "frontend/src/pages/Studio.tsx"
related_targets: []
---

# Surface brief: Studio

Scope: `frontend/src/pages/Studio.tsx` and its components. Mode: Operate.
Task: upload or open a room, read its analysis, compare palettes on the photo, read why, refine (goal, locks, nudges, click-to-fix mask), settle on paints. States: uploading/analysing, ready, previewing, LLM vs template explanations, mask editing, errors (bad file, room expired, backend down).
Constraints: every swatch shows name, code, LRV in text; keyboard-operable slider; legible on a projector.

## Direction contract

THESIS: The photo is the workspace and the palette is a material strip, not a grid of colour dots. Refuses the category default of a generic dashboard with round swatches and a chat box.

OWN-WORLD: Same material board as the landing: limestone ground, plaster panels, walnut rail for the active palette, near-black ink, rust for edits/exclusions, cobalt for selection and locks. Archivo, tabular numerals for LRV and codes. Hairline dividers, outline pills, square-cornered swatch chips.

STORY: The user sees their room analysed (current 60-30-10, light), flips through palettes on the photo instantly, reads a plain reason for each, and refines until the strip feels right.

FIRST VIEWPORT: Slim top bar (wordmark, room name, new photo). Left ~65%: photo canvas with before/after handle and a mode switch (Preview / Fix what's painted). Right ~35%: goal chips, then stacked palette cards, the active one expanded with its 60/30/10 strip, paint names, explanation and refine steppers.

FORM: Pinned by user references, translated to Operate (type, palette, density, one signature move: the 60/30/10 proportional paint strip). Position 1 of 1. seed: none (pinned by user references; roll not run per new-work.md: "a user- or brief-pinned direction beats the roll, always").

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
