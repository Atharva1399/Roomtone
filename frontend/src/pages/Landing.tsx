import { useState } from 'react'
import { ArrowRight, Check } from 'lucide-react'
import { BeforeAfter } from '../components/BeforeAfter'
import { Dropzone } from '../components/Dropzone'
import { Wordmark } from '../components/Mark'
import { PaintStrip } from '../components/PaintStrip'
import { SiteHeader } from '../components/SiteHeader'
import { SCHEME_LABEL } from '../lib/api'
import type { DemoPalette, DemoRole, DemoRoom } from '../lib/demo'
import { PHOTO_CREDITS, roomImage, useDemo } from '../lib/demo'
import { checkSentence } from '../lib/checks'
import { go, openSample } from '../lib/router'
import './Landing.css'

const HERO_DEFAULT = 1 // the warm neutral reads best against the dining room's wood
const SAMPLE_ORDER = ['dining_room', 'bedroom_historic', 'bedroom_modern', 'living_luxury']

const paintName = (r: DemoRole) => [r.brand, r.name, r.code].filter(Boolean).join(' ') || r.hex
const fmt = (n: number) => n.toLocaleString('en-US')

export function Landing() {
  const demo = useDemo()

  return (
    <div className="landing">
      <SiteHeader />
      <main>
        <Hero demo={demo ? { id: demo.hero, room: demo.rooms[demo.hero] } : null} />
        {demo && (
          <>
            <Derivation room={demo.rooms[demo.hero]} roomId={demo.hero} counts={demo.counts} />
            <LightSection room={demo.rooms[demo.light_demo]} roomId={demo.light_demo} />
            <Rooms rooms={demo.rooms} />
          </>
        )}
        <Close />
      </main>
      <Footer />
    </div>
  )
}

/* ---------- hero ---------- */

function Hero({ demo }: { demo: { id: string; room: DemoRoom } | null }) {
  const [active, setActive] = useState(HERO_DEFAULT)
  const palette = demo?.room.palettes[active]

  return (
    <section className="hero" aria-labelledby="hero-title">
      <div className="hero__top">
        <h1 id="hero-title" className="hero__title">
          Repaint the room you already have.
        </h1>
        <div className="hero__offer">
          <p className="hero__lede">
            Upload one photo. Roomtone measures your walls, floor and light, builds palettes around what stays, and
            shows each real paint on your own walls, under your own light.
          </p>
          <Dropzone />
          <div className="hero__samples">
            <span>Or open a sample room:</span>
            {SAMPLE_ORDER.slice(0, 3).map((id) => (
              <button key={id} type="button" className="link-btn" onClick={() => openSample(id, sampleTitle(id))}>
                {sampleTitle(id)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {demo && palette && (
        <figure className="hero__stage">
          <BeforeAfter
            className="hero__ba"
            before={roomImage(demo.id, 'original.jpg')}
            after={roomImage(demo.id, `palette_${active}.jpg`)}
            alt="A Victorian dining room with terracotta walls"
            beforeLabel="Before"
            afterLabel={`After · ${palette.roles.dominant.name || palette.roles.dominant.code}`}
            intro
            rest={26}
          />
          <figcaption className="hero__caption">
            <div className="hero__caption-text">
              <p className="hero__paint">
                <span className="swatch" style={{ background: palette.roles.dominant.hex }} />
                {paintName(palette.roles.dominant)}
              </p>
              <p className="hero__note">
                <span className="num">LRV {palette.roles.dominant.lrv.toFixed(0)}</span> · {SCHEME_LABEL[palette.scheme]} scheme, built around the walnut table. Panelling, table and floor kept as
                they are with {demo.room.excluded_clicks} clicks.
              </p>
            </div>
            <div className="hero__switch" role="group" aria-label="Show another suggested palette">
              {demo.room.palettes.map((p, i) => (
                <button
                  key={p.roles.dominant.hex}
                  type="button"
                  className={`chip${i === active ? ' is-active' : ''}`}
                  aria-pressed={i === active}
                  onClick={() => setActive(i)}
                >
                  <span className="swatch" style={{ background: p.roles.dominant.hex }} />
                  {p.roles.dominant.name || p.roles.dominant.code}
                </button>
              ))}
            </div>
          </figcaption>
        </figure>
      )}
    </section>
  )
}

function sampleTitle(id: string) {
  return { dining_room: 'Dining room', bedroom_historic: 'Canopy bedroom', bedroom_modern: 'City bedroom', living_luxury: 'Open-plan living' }[id] ?? id
}

/* ---------- derivation ---------- */

function Derivation({ room, roomId, counts }: { room: DemoRoom; roomId: string; counts: { paints: number; room_classes: number; harmony_schemes: number } }) {
  const hero = room.palettes[HERO_DEFAULT]
  const cur = room.current_roles
  const wall = hero.roles.dominant

  return (
    <section id="derive" className="derive" aria-labelledby="derive-title">
      <div className="section-head">
        <h2 id="derive-title" className="statement">
          How a palette
          <br />
          is derived
        </h2>
        <p className="section-head__copy">
          No mood board and no guesswork. Five steps run on your photo in about two seconds, and each one leaves
          something you can check. Everything below is the real output for the dining room above.
        </p>
      </div>

      <ol className="steps">
        <li className="step">
          <div className="step__head">
            <h3 className="step__title">Segment</h3>
            <span className="step__n num">01</span>
          </div>
          <p className="step__figure num">{counts.room_classes}</p>
          <p className="step__copy">
            room classes the segmentation model knows. It splits the photo into walls, floor, ceiling and furniture.
          </p>
          <figure className="step__art">
            <img src={roomImage(roomId, 'segmentation.jpg')} alt="The dining room split into wall, floor and furniture regions" loading="lazy" />
            <figcaption>
              <span className="legend legend--wall">Wall</span>
              <span className="legend legend--floor">Floor</span>
              <span className="legend legend--furniture">Furniture</span>
              <span className="step__aside">The raw read counts the panelling as wall; one click fixes it.</span>
            </figcaption>
          </figure>
        </li>

        <li className="step">
          <div className="step__head">
            <h3 className="step__title">Measure</h3>
            <span className="step__n num">02</span>
          </div>
          <p className="step__copy">
            Colours are clustered in CIELAB, a colour space where distance matches what the eye sees. Out comes the
            room as it is today: its 60 / 30 / 10 split and its light.
          </p>
          <div className="step__art">
            <PaintStrip
              size="lg"
              dominant={{ hex: cur.dominant.hex, label: cur.dominant.source, sub: cur.dominant.hex }}
              secondary={{ hex: cur.secondary?.hex ?? cur.dominant.hex, label: cur.secondary?.source ?? '', sub: cur.secondary?.hex }}
              accent={{ hex: cur.accent?.hex ?? cur.dominant.hex, label: 'accent' }}
            />
            <p className="step__aside">
              Light: {room.lighting.brightness}, {room.lighting.temperature}
            </p>
          </div>
        </li>

        <li className="step">
          <div className="step__head">
            <h3 className="step__title">Harmonise</h3>
            <span className="step__n num">03</span>
          </div>
          <p className="step__figure num">{room.candidates_considered}</p>
          <p className="step__copy">
            candidates from {counts.harmony_schemes} colour-wheel schemes in three shades each, all {room.base_hue_reason.replace(/\s*\(#\w+\)/, '')}.
          </p>
          <ul className="step__art schemes">
            {room.palettes.map((p: DemoPalette) => (
              <li key={p.scheme}>
                <span className="schemes__chip" style={{ background: p.roles.dominant.hex }} />
                <span>{SCHEME_LABEL[p.scheme]}</span>
              </li>
            ))}
          </ul>
        </li>

        <li className="step">
          <div className="step__head">
            <h3 className="step__title">Check</h3>
            <span className="step__n num">04</span>
          </div>
          <p className="step__copy">
            Rules drop any palette that would fail in this room: a wall that blends into the floor, an accent too weak
            to read, a wall too dark for a dim room.
          </p>
          <ul className="step__art checks">
            {hero.checks.slice(0, 4).map((c) => (
              <li key={c.rule}>
                <Check size={16} strokeWidth={2} aria-hidden="true" className={c.passed ? 'is-ok' : 'is-warn'} />
                <span>{checkSentence(c)}</span>
              </li>
            ))}
          </ul>
        </li>

        <li className="step">
          <div className="step__head">
            <h3 className="step__title">Real paint</h3>
            <span className="step__n num">05</span>
          </div>
          <p className="step__figure num">{fmt(counts.paints)}</p>
          <p className="step__copy">
            real paints from four colour books. Every colour becomes the nearest one you can buy, matched by CIEDE2000.
          </p>
          <div className="step__art chipcard">
            <span className="chipcard__swatch" style={{ background: wall.hex }} />
            <span className="chipcard__brand">{wall.brand}</span>
            <span className="chipcard__name">{wall.name || wall.code}</span>
            <span className="chipcard__meta num">
              {wall.code} · LRV {wall.lrv.toFixed(0)} · {wall.hex}
            </span>
          </div>
        </li>
      </ol>
    </section>
  )
}

/* ---------- paint x light ---------- */

function LightSection({ room, roomId }: { room: DemoRoom; roomId: string }) {
  const p = room.palettes[1]
  const paint = p.roles.dominant
  return (
    <section id="light" className="light" aria-labelledby="light-title">
      <div className="section-head section-head--walnut">
        <h2 id="light-title" className="statement">
          Paint <span className="times">×</span> the light
          <br />
          already in your photo
        </h2>
        <p className="section-head__copy">
          Roomtone never redraws your room. It lifts the light off the wall, every shadow, lamp pool and bit of
          texture, and multiplies the new paint into it. Only the paint changes, so the preview shows the paint you
          would actually buy.
        </p>
      </div>

      <div className="equation">
        <figure className="eq__item">
          <img src={roomImage(roomId, 'original.jpg')} alt="A bedroom with white walls lit by two bedside lamps" loading="lazy" />
          <figcaption>Your photo</figcaption>
        </figure>
        <span className="eq__op eq__op--arrow" aria-hidden="true">
          <ArrowRight size={36} strokeWidth={1.25} />
        </span>
        <figure className="eq__item">
          <img src={roomImage(roomId, 'light_layer.jpg')} alt="The same wall in neutral grey, showing only its light and shadow" loading="lazy" />
          <figcaption>The wall’s light, measured once</figcaption>
        </figure>
        <span className="eq__op" aria-hidden="true">
          ×
        </span>
        <figure className="eq__item eq__item--paint">
          <span className="eq__paint" style={{ background: paint.hex }} />
          <figcaption>
            {paint.brand} {paint.name || paint.code}
          </figcaption>
        </figure>
        <span className="eq__op" aria-hidden="true">
          =
        </span>
        <figure className="eq__item">
          <img src={roomImage(roomId, 'palette_1.jpg')} alt={`The bedroom walls in ${paint.name}, lamp glow intact`} loading="lazy" />
          <figcaption>The preview</figcaption>
        </figure>
      </div>

      <dl className="light__facts">
        <div>
          <dt className="num">~0.1 s</dt>
          <dd>per colour change. The light is measured once, so trying another paint is one multiplication.</dd>
        </div>
        <div>
          <dt>Same room</dt>
          <dd>Lamp pools stay warm and shadows stay where they fell. Nothing is invented by an image model.</dd>
        </div>
        <div>
          <dt>Click to fix</dt>
          <dd>Mislabelled panelling or a cabinet? Click it and it is left out of the paint, with undo.</dd>
        </div>
      </dl>
    </section>
  )
}

/* ---------- sample rooms ---------- */

function Rooms({ rooms }: { rooms: Record<string, DemoRoom> }) {
  return (
    <section id="rooms" className="rooms" aria-labelledby="rooms-title">
      <div className="rooms__intro">
        <h2 id="rooms-title" className="rooms__title">
          Try it on a real room
        </h2>
        <p className="section-head__copy">
          Four photos from Wikimedia Commons, already analysed. Open one in the studio and change everything: goal,
          locks, warmth, what gets painted.
        </p>
        <button type="button" className="pill" onClick={() => go('studio')}>
          Upload your own <ArrowRight size={16} aria-hidden="true" />
        </button>
      </div>
      <ul className="rooms__grid">
        {SAMPLE_ORDER.map((id, idx) => {
          const r = rooms[id]
          const p = r.palettes[0]
          return (
            <li key={id} className="room-card">
              <button type="button" className="room-card__btn" onClick={() => openSample(id, r.title)}>
                <span className="room-card__img">
                  <img src={roomImage(id, 'palette_0.jpg')} alt="" loading="lazy" />
                  <img className="room-card__before" src={roomImage(id, 'original.jpg')} alt="" loading="lazy" />
                </span>
                <span className="room-card__body">
                  <span className="room-card__title">
                    {r.title}
                    <span className="room-card__n num">{String(idx + 1).padStart(2, '0')}</span>
                  </span>
                  <span className="room-card__paint">
                    {paintName(p.roles.dominant)}
                  </span>
                  <PaintStrip
                    size="sm"
                    showText={false}
                    dominant={{ hex: p.roles.dominant.hex, label: '' }}
                    secondary={{ hex: p.roles.secondary.hex, label: '' }}
                    accent={{ hex: p.roles.accent.hex, label: '' }}
                  />
                  <span className="room-card__open">
                    Open in the studio <ArrowRight size={14} aria-hidden="true" />
                  </span>
                </span>
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}

/* ---------- close + footer ---------- */

function Close() {
  return (
    <section className="close" aria-labelledby="close-title">
      <h2 id="close-title" className="close__title">
        Bring one photo.
      </h2>
      <div className="close__action">
        <p className="section-head__copy">
          A phone photo is enough: daylight, walls in view, the floor and big furniture in frame. Palettes and previews
          arrive in a few seconds.
        </p>
        <Dropzone />
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="footer">
      <div className="footer__brand">
        <Wordmark />
        <p>Built for PS 39, AI Interior Colour Palette Suggester. Roomtone is a working name.</p>
      </div>
      <div className="footer__credits">
        <p className="footer__label">Photographs</p>
        <ul>
          {PHOTO_CREDITS.map((c) => (
            <li key={c.room}>
              <a href={c.href} target="_blank" rel="noreferrer">
                {c.title}
              </a>{' '}
              · Wikimedia Commons · {c.licence}. Shown repainted by Roomtone.
            </li>
          ))}
        </ul>
      </div>
      <p className="footer__fine">
        Paint colours are the brands’ on-screen approximations of Sherwin-Williams, Benjamin Moore, Behr and Farrow &amp;
        Ball colours. Check a physical sample on your wall before you buy.
      </p>
    </footer>
  )
}
