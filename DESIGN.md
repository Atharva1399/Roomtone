---
name: Roomtone
description: An architect's material board for choosing wall paint from your own room photo.
colors:
  stone-0: "#f9f7f3"
  stone-50: "#f3f0ea"
  stone-100: "#e9e4da"
  plaster: "#d9cfc0"
  plaster-deep: "#b9ab97"
  walnut: "#3a2419"
  walnut-2: "#4a2f21"
  walnut-ink: "#f3e9dd"
  walnut-ink-2: "#cdb9a5"
  ink: "#16130f"
  ink-2: "#4a443c"
  ink-3: "#6b645a"
  line: "rgba(22, 19, 15, 0.14)"
  line-strong: "rgba(22, 19, 15, 0.32)"
  line-on-walnut: "rgba(243, 233, 221, 0.18)"
  rust: "#c2452a"
  rust-ink: "#a8361d"
  cobalt: "#2643c4"
  cobalt-ink: "#2340b8"
  ok: "#2f6b3f"
typography:
  display:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "clamp(3.25rem, 1.2rem + 7.2vw, 8.5rem)"
    fontWeight: 640
    lineHeight: 0.92
    letterSpacing: "-0.04em"
    fontVariation: "'wdth' 92"
  statement:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "clamp(2.25rem, 1.4rem + 3vw, 4rem)"
    fontWeight: 300
    lineHeight: 1.02
    letterSpacing: "-0.03em"
    fontVariation: "'wdth' 125"
  title:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "clamp(1.5rem, 1.1rem + 1.2vw, 2.125rem)"
    fontWeight: 640
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  lede:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 400
    lineHeight: 1.45
  body:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "'ss01'"
    fontVariation: "'wdth' 100"
  label:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 560
    letterSpacing: "0.01em"
  caption:
    fontFamily: "'Archivo Variable', 'Archivo', system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 560
rounded:
  chip: "2px"
  pill: "999px"
spacing:
  s-1: "4px"
  s-2: "8px"
  s-3: "12px"
  s-4: "16px"
  s-5: "24px"
  s-6: "32px"
  s-7: "48px"
  s-8: "72px"
  s-9: "112px"
  s-10: "160px"
  gutter: "clamp(16px, 4vw, 64px)"
components:
  button-pill:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "0 24px"
    height: "40px"
  button-pill-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.stone-0}"
  button-pill-solid:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.stone-0}"
    rounded: "{rounded.pill}"
    padding: "0 24px"
    height: "40px"
  button-pill-solid-hover:
    backgroundColor: "{colors.walnut}"
    textColor: "{colors.stone-0}"
  button-pill-light:
    backgroundColor: "transparent"
    textColor: "{colors.walnut-ink}"
    rounded: "{rounded.pill}"
  chip-goal:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "0 16px"
    height: "36px"
  chip-goal-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.stone-0}"
  segmented-option-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.stone-0}"
    rounded: "{rounded.pill}"
    height: "36px"
  segmented-option-edit:
    backgroundColor: "{colors.rust}"
    textColor: "{colors.stone-0}"
  palette-tab:
    backgroundColor: "{colors.stone-0}"
    textColor: "{colors.ink}"
    rounded: "{rounded.chip}"
    padding: "8px"
  palette-tab-active:
    backgroundColor: "{colors.walnut}"
    textColor: "{colors.walnut-ink}"
  palette-rail:
    backgroundColor: "{colors.walnut}"
    textColor: "{colors.walnut-ink}"
    rounded: "{rounded.chip}"
    padding: "16px"
  paint-chip:
    rounded: "{rounded.chip}"
    size: "36px"
  lock-locked:
    backgroundColor: "{colors.cobalt}"
    textColor: "{colors.stone-0}"
    size: "36px"
  dropzone:
    backgroundColor: "{colors.stone-0}"
    textColor: "{colors.ink}"
    rounded: "{rounded.chip}"
    padding: "24px"
  dropzone-walnut:
    backgroundColor: "{colors.walnut-2}"
    textColor: "{colors.walnut-ink}"
  site-header:
    backgroundColor: "{colors.stone-0}"
    textColor: "{colors.ink}"
    height: "72px"
    padding: "0 clamp(16px, 4vw, 64px)"
---

# Design System: Roomtone

## Overview

**Creative North Star: "The Architect's Material Board"**

Roomtone is laid out like a sample board pinned in an architect's studio: a limestone sheet, a warm plaster panel, a block of walnut, near-black ink annotations, and the paint itself cut into strips at its real proportions. The world was pinned by the user through two reference sheets (architectural-editorial interior studios) and the build carries it on both surfaces: the Landing persuades with the mechanism working, the Studio operates on the user's own photo. Colour is the product, so the system's own colour stays quiet; walnut is the one heavy material, and rust and cobalt are pencil marks, never fields.

Density moves with the surface. The Landing is open and architectural: hairline guide lines run the full page height at the gutters and centre, sections breathe on 160px rhythm, and type swings between a tight heavy display cut and a wide light statement cut of the same family. The Studio is a working bench: a photo canvas on the left, a plaster panel on the right with the active palette sitting on a walnut rail. Depth is a hairline and a soft lift, never a stack of cards.

Truth governs ornament. Every number, colour and count on screen comes from the pipeline; exact counts carry no "+", check results read as plain sentences with the number that decided them, and every swatch names its paint, code and LRV in text.

**Key Characteristics:**
- Limestone ground, plaster mid-tone, walnut block, near-black ink.
- Rust and cobalt are marks and states only, never backgrounds of sections.
- One family (Archivo variable, width axis): tight heavy display (wdth 88-92) against wide light statements (wdth 125).
- Hairline dividers and full-height guide lines instead of boxes.
- Outline pill buttons; square-cornered (2px) swatch chips.
- The signature 60/30/10 paint strip, drawn at true proportion.

## Colors

A warm mineral neutral board with one deep wood block and two small pigment marks.

### Primary
- **Walnut Block** (walnut): the one heavy material. Whole-section field for the "paint x light" band on the Landing, the rail behind the open palette in the Studio, the active palette tab, and the solid pill's hover. Text on it is Walnut Paper (walnut-ink) and Walnut Dust (walnut-ink-2, 7:1); hairlines on it are line-on-walnut.
- **Walnut Shadow** (walnut-2): a step lighter, for the drop zone placed on walnut, furniture legends, and the room readout icons.

### Secondary
- **Brick Rust** (rust): the "+" mark before labels and headings, the active "Fix what's painted" edit mode, disclosure markers, the alert border, and the drop zone's upload icon. As text it becomes **Rust Ink** (rust-ink, 5.6:1 on limestone) for warnings and errors.

### Tertiary
- **Survey Cobalt** (cobalt): selection, locks and focus only. The focus ring, `::selection`, a locked paint, the active level in a refine stepper, the selected sample-room chip on the Landing, and the analysing scan line. As link-hover text it is **Cobalt Ink** (cobalt-ink).
- **Check Green** (ok): the tick on a passing rule check, and nothing else.

### Neutral
- **Sheet** (stone-0): the lightest paper: site header, studio bar, Landing page body, drop zone, cards and the open palette.
- **Limestone** (stone-50): the body ground, the Studio canvas ground, the sample-rooms section, the inside of the chip card.
- **Plaster Light** (stone-100): the Studio panel, the room readout, image wells behind photos, hover fill in segmented controls.
- **Plaster** (plaster): the mark's dominant band, inactive stepper ticks.
- **Deep Plaster** (plaster-deep): the scrollbar thumb.
- **Ink** (ink): text, outline pill borders, active goal chip and segmented option fill, the Landing footer field.
- **Ink 2** (ink-2, 8.7:1): secondary copy, ledes, explanations.
- **Ink 3** (ink-3, 5.2:1): labels, indexes, hex codes, counts, hints.
- **Hairline** (line) and **Strong Hairline** (line-strong): dividers, guide lines, chip and segmented outlines.

### Named Rules
**The Marks-Not-Fields Rule.** Rust and cobalt never fill a section, card or panel. Rust marks edits, exclusions, the "+" and warnings; cobalt marks selection, locks and focus. The only filled rust or cobalt shapes are state controls (an active edit mode, a locked paint) and tiny legend squares.

**The Walnut-Is-Active Rule.** Walnut is the material of the thing currently chosen: the active palette tab, the rail of the open palette. In the Studio, the walnut block means "this is the one you are looking at".

**The Colour-Is-Content Rule.** The interface never competes with the paint. Palette colours appear only as swatches, strips and repainted photos, always with name, code and LRV in text beside them.

## Typography

**Display Font:** Archivo Variable (with Archivo, system-ui, sans-serif), loaded with its width axis.
**Body Font:** the same family at normal width (wdth 100) with stylistic set ss01.

**Character:** One grotesk doing two voices, as on the reference sheets: condensed and heavy for names and numbers, extended and light for statements. Weights sit between the named stops (540, 560, 620, 640, 660) because the variable font is used as a material, not a menu.

### Hierarchy
- **Display** (640, wdth 88-92, -0.04em, line-height 0.9): section titles that act as posters ("Try it on a real room") and the Landing hero, which runs a slightly smaller cut (clamp(3rem, 0.9rem + 5.6vw, 7rem), wdth 90). The derivation step figures (150, 24, 9,707) use the same cut at clamp(2.75rem, 1.6rem + 2.6vw, 4.75rem), wdth 88.
- **Statement** (300, wdth 125, -0.03em): the wide light voice for section heads ("How a palette is derived", "Paint x the light"); at display size and wdth 125 for the closing call. The multiplication sign inside a statement is set upright at wdth 100, weight 260.
- **Title** (640, -0.02em): step titles, palette names on the walnut rail, light facts.
- **Lede** (400, 1.25rem, 1.45): offers and section copy, max 54ch, in ink-2.
- **Body** (400, 1rem, 1.5): explanations and step copy; long text capped at 36-54ch.
- **Label** (540-600, 0.875rem): buttons, chips, nav, paint names, panel headings at 1rem/640.
- **Caption** (500-560, 0.75rem): paint roles, hex and LRV meta, legend text, counts, brand names.

### Named Rules
**The Two-Voices Rule.** Heavy-and-narrow names things and states numbers; light-and-wide makes statements. Never set a statement heavy or a number light.

**The Tabular Truth Rule.** Every number that comes from the pipeline (LRV, codes, percentages, counts, indexes) is set with tabular numerals.

**The Plus Rule.** The rust "+" prefixes labels and headings (header nav, studio panel headings, explanation terms), as on the reference sheets. It never decorates a number: exact counts are written exactly ("4 of 24 candidates", "9,707"), never "24+".

## Layout

A gutter of clamp(16px, 4vw, 64px) frames everything. Spacing runs on a 4px base (4, 8, 12, 16, 24, 32, 48, 72, 112, 160px), with 160px between Landing sections and 72px for inner section breaks.

The Landing is a two-column editorial grid: headline or statement on the left, offer or copy aligned to the right column's end; hairline guide lines (1px, line) run the full page height at both gutters and the centre, and disappear below 860px. The derivation is a five-column strip of steps divided by hairlines, collapsing to two columns at 1100px and one at 860px. Sample rooms use a sticky intro beside a two-column card grid whose even cards drop 72px for a staggered board.

The Studio is a canvas plus panel: the photo takes the flexible column, the panel is 360-440px, sticky under a 64px bar and scrolling on its own. Below 1000px the panel drops beneath the canvas; below 560px the room name leaves the bar and panel blocks take the page gutter.

Breakpoints observed: 1100px, 1000px, 860px, 560px, 520px.

Indexes ride the title line. Step numbers (01-05) and sample-room numbers sit right-aligned on the same baseline as their title, in ink-3 at label size, never stacked above it.

## Elevation & Depth

Mostly flat, separated by hairlines and tone: limestone to sheet to plaster. Lift exists only as a soft, warm, diffuse shadow, used at rest for things that are physically placed on the board (the before/after knob, the chip card, the scan label) and on hover for things you can pick up (drop zone, sample-room cards). Cards at rest sit on a single 1px hairline under their bottom edge.

### Shadow Vocabulary
- **Lift** (`box-shadow: 0 10px 30px -12px rgba(22, 19, 15, 0.28)`): hover on drop zone and sample-room cards; at rest on the slider knob, chip card and scan label.
- **Rest line** (`box-shadow: 0 1px 0 var(--line)`): the resting edge of room cards, the open palette, and under the chip card.
- **Canvas** (`box-shadow: 0 1px 0 var(--line), 0 24px 60px -36px rgba(22, 19, 15, 0.45)`): the Studio photo frame only.
- **Swatch edge** (`box-shadow: inset 0 0 0 1px rgba(22, 19, 15, 0.10-0.16)`): every swatch, chip and strip, so pale paints keep an edge on pale ground.

### Named Rules
**The Hairline-First Rule.** Separate with a 1px hairline or a tone step before reaching for a shadow. A shadow means "this is lifted off the board", not "this is a section".

## Shapes

Two radii only. Things you press are pills (999px): buttons, goal chips, segmented controls, tags over photos. Things that are material are near-square (2px): swatches, paint strips, photos inside cards, drop zone, palette tabs, the walnut rail, alerts. Circles are reserved for single-glyph controls (lock, refine steppers, slider knob, 36-44px). The mark is three rectangles in 60/30/10 proportion: plaster, walnut, rust.

## Components

### Buttons
Outline pills, quiet at rest and inked on hover.
- **Shape:** full pill (999px), 40px tall, 24px side padding, 1px ink border.
- **Outline (default):** transparent with ink text and border; hover fills ink with sheet text.
- **Solid:** ink fill, sheet text; hover turns walnut.
- **Light (on walnut or ink):** walnut-ink border and text; hover fills walnut-ink with walnut text.
- **Hover / Focus:** 220ms colour transitions on the ease-out curve; focus is the global 2px cobalt ring at 3px offset.
- **Disabled:** 45% opacity, no hover fill.
- **Text link button:** ink, weight 600, 1px underline at 0.25em offset; hover to cobalt ink.
- **Icon button:** borderless pill with a 16px line icon and label; hover shows a strong hairline border.

### Chips
- **Goal chip:** 36px outline pill in strong hairline; hover inks the border; active fills ink with sheet text.
- **Sample-room chip (Landing):** 36px pill with a 20px square swatch at its start; active is a cobalt ring with an 8% cobalt tint (selection).
- **Segmented control:** a 3px-padded pill track on sheet with 36px pill options; the active option fills ink, and the edit mode fills rust. A 30px small variant exists.

### Cards / Containers
- **Corner Style:** 2px.
- **Background:** sheet on limestone, or sheet on plaster in the Studio panel.
- **Shadow Strategy:** rest line at rest, lift on hover (see Elevation).
- **Border:** none, or a single hairline.
- **Internal Padding:** 12-16px; room cards 16px body under a 4:5 photo whose "as photographed" version wipes in from the left on hover and focus (700ms).

### Inputs / Fields
- **Drop zone:** the one field. Sheet with a strong hairline and 2px corners, 24px padding, a rust line icon, a 1.25rem/620 main line and an ink-3 sub line. Hover and drag-over ink the border and lift; drag-over also tones to plaster light. On walnut it becomes walnut-2 with walnut-ink text and a walnut-ink border on hover. Errors show beneath in rust ink.

### Navigation
- **Site header:** 72px sheet bar on a hairline, three columns (nav, wordmark, outline pill CTA); nav links are label size, weight 540, prefixed with the rust "+", underline on hover; nav hides below 860px.
- **Studio bar:** 64px sticky sheet bar: back arrow, wordmark, a "/" in strong hairline, then the room name in ink-2; the new-photo pill sits at the end.
- **Wordmark:** the 60/30/10 mark beside "Roomtone" at 1.125rem/700, wdth 112, -0.02em.

### Paint Strip (signature)
A palette drawn as a strip cut from a material board: three bands at true 60 / 30 / 10 width, 2px corners, an inset swatch edge. Each band carries its percentage (caption, top), its role and paint name (label, bottom), with text ink chosen by WCAG luminance (ink or sheet). The 10% band keeps only its percentage, and drops even that when the strip is under 340px wide. Sizes: 132px (open palette), 28px (palette tabs), 14px (room cards). Band colours ease over 360ms when a palette changes.

### Palette Tab and Walnut Rail
Palette tabs sit in a two-column grid: sheet, hairline border, 2px corners, a 28px strip, the palette name and its index in ink-3. The active tab turns walnut with walnut-ink text. The open palette sits on a walnut rail (name at title size, scheme in walnut-ink-2, the 132px strip), followed by paint rows on sheet.

### Paint Row
A 36px square chip, role (caption, ink-3) above name (label, 600), hex and LRV right-aligned in tabular caption, and a 36px circular lock. Unlocked is a hairline circle; hover turns it cobalt; locked fills cobalt with a sheet glyph. Long names truncate with an ellipsis and carry the full name in a title tooltip.

### Rule Checks
Plain sentences with the deciding number ("Walls stand apart from your floor: colour difference 13.9, where 8 is the minimum."), a 16px check glyph in check green for a pass, rust ink for a warning.

### Before / After Slider
A full-bleed comparison with a 2px sheet divider, a 44px sheet knob with lift and a two-way chevron, and sheet tags ("As photographed", the paint name) in the bottom corners. Operable by keyboard; focus rings the knob in cobalt and scales it 1.06.

### Motion
One curve everywhere: ease-out `cubic-bezier(0.16, 1, 0.3, 1)`, 200-360ms for state changes. The Studio's one authored motion is the analysing scan: a cobalt measuring line passing down a desaturated photo (1.8s loop). The repainted half settles in from slightly desaturated (420ms). Reduced motion collapses all of it to 1ms.

## Do's and Don'ts

### Do:
- **Do** keep the ground in the stone and plaster family and use walnut as the single heavy block per view.
- **Do** use rust only for the "+" mark, edits and exclusions, warnings and errors; cobalt only for selection, locks and focus.
- **Do** mark the active, chosen material in walnut (active palette tab, open-palette rail).
- **Do** write exact counts exactly, with tabular numerals and no "+".
- **Do** say rule checks as plain sentences that include the number that decided them.
- **Do** put step and card indexes (01, 02...) on the title line, right-aligned, in ink-3.
- **Do** show every paint's name, code and LRV in text next to its swatch.
- **Do** use 2px corners for anything material and full pills for anything pressed.
- **Do** separate with hairlines (1px, line) and tone steps before shadows.

### Don't:
- **Don't** fill a section, card or panel with rust or cobalt.
- **Don't** round swatches into dots; swatches are square-cornered chips and strips.
- **Don't** set statements heavy or numbers light; the two voices stay in their roles.
- **Don't** let the interface's own colour compete with the paint on screen.
- **Don't** add a "+" to numbers, or round a count up to sound bigger.
- **Don't** put eyebrow or kicker lines above section headings; the rust "+" prefix and the on-line index are the only heading devices.

## Provenance

Every raster under `frontend/public/demo` carries an embedded origin: the Wikimedia Commons source and licence for photographs, or a note of how the pipeline derived it (repainted preview, light layer, segmentation). Rerunning `backend/scripts/export_demo.py` overwrites these files and strips that metadata; after any rerun, re-embed each origin with `impeccable embed-prompt`. Photo credits also appear in the Landing footer and in `backend/samples/CREDITS.md`.
