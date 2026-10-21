# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React + Vite (TypeScript) frontend in `frontend/`, calling the FastAPI backend in `backend/` (`http://127.0.0.1:8000`, interactive docs at `/docs`). Chosen by the user.

## Users

Primary audience right now: a hackathon jury watching a live demo on a laptop or projector. They judge problem statement PS 39 ("AI Interior Colour Palette Suggester") and will ask two questions: how is a palette derived, and how achievable are the previews. Real end users (people choosing wall paint for their own room, without design training) come later and must still be able to use the studio unaided.

## Product Purpose

Turn one room photo into wall-paint palettes the user can believe: analyse the room, generate palette options with realistic previews in that same photo, explain each choice, and let the user refine. Success in the demo: a judge sees a real room repainted convincingly within seconds, understands why each palette was chosen, and can watch it being refined live.

## Positioning

The palette is derived, not imagined: the photo is segmented into surfaces (walls, floor, furniture), the room's existing colours are measured in CIELAB, colour-harmony schemes are built around what stays (floor, sofa), every colour is snapped to a real, buyable paint, and rule checks reject palettes that would fail in that room. Previews repaint only the wall and keep the photo's own light (paint colour x the photo's shading and light tint), so the preview shows the exact paint, not an AI re-imagining of the room.

## Operating Context

- Flow: upload photo -> room analysis (surfaces, current 60-30-10 colours, light) -> 3-5 palettes -> live preview on the photo -> explanations -> refine -> pick.
- Refine controls that exist in the API: goal (balanced, calm, cosy, bright, bold), lock a wall or accent colour and rebuild the rest, nudge warmer/cooler, lighter/darker, muted/vivid, nudge a single colour to the nearest real paints, click on the photo to exclude or include an object from the paint area (undo, reset), brand filter.
- Explanations come from an LLM when one is configured (Hugging Face or Claude); otherwise from templates built from the same facts. The UI must work and read well in both cases.
- Speeds measured on the dev machine (RTX 3050 Ti): upload and analysis about 2 s, palettes about 0.3 s, a colour change about 0.1 s, first click-to-fix about 2 s then about 50 ms.

## Capabilities and Constraints

- 9,707 real paint colours from four colour books: Sherwin-Williams, Benjamin Moore, Behr, Farrow & Ball (hex values are the brands' on-screen approximations). Some Benjamin Moore entries have only a code, no name.
- Segmentation model: SegFormer trained on ADE20K (150 classes), grouped into wall, ceiling, floor, window, door, sofa, bed, table, storage, decor.
- Paintable surfaces: wall and ceiling. Ornate ceilings are left unpainted by default.
- Known limitation: the segmenter sometimes labels wood panelling, flat white cabinets or blinds as wall; the click-to-fix tool is the remedy.
- No Indian paint brands yet (Asian Paints, Berger, Nerolac); not to be implied.
- No user accounts, saving, or sharing yet. Rooms live in server memory (last 20).

## Brand Commitments

- Name "Roomtone" is a placeholder proposed by the assistant; the user may replace it.
- Visual references supplied by the user (two interior-studio landing pages: an architectural-editorial layout with large grotesk display type, warm stone and off-white grounds, rust and cobalt accents, a deep brown block, hairline grids, large stat numerals, full-bleed interior photography). These are binding inspiration for the frontend.

## Evidence on Hand

- Ten sample room photos in `backend/samples/` from Wikimedia Commons with licences and credits in `backend/samples/CREDITS.md` (CC0, CC BY, CC BY-SA; attribution required for BY and BY-SA). `living_kolkata.png` carries a third-party watermark and should not lead any marketing surface.
- Real pipeline output for each sample (palettes, previews, checks) can be generated from the running backend.
- No testimonials, customers, users, press, benchmarks against competitors, or pricing exist. None may be invented.

## Product Principles

1. Show the mechanism, not a claim: every number and colour on screen comes from the pipeline.
2. The photo is the canvas: previews happen in the user's own room, with its own light.
3. Real paint or nothing: every suggested colour names a buyable paint and its code.
4. Explain like a consultant: plain reasons tied to the room (light, floor, style), never jargon without a gloss.
5. Works without the LLM: templates are a first-class path, not an error state.

## Accessibility & Inclusion

Colour is the product, so information must never rely on colour alone: every swatch carries its name, code and LRV in text, and before/after comparison works with the keyboard. Projector demo: text must stay legible at a distance and in washed-out projection.
