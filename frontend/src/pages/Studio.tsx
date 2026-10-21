import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowLeft, Columns2, Eraser, Lock, LockOpen, Minus, PaintRoller, Plus, RotateCcw, Sofa, Sparkles, Sun, Undo2, Upload } from 'lucide-react'
import { BeforeAfter } from '../components/BeforeAfter'
import { Dropzone } from '../components/Dropzone'
import { Wordmark } from '../components/Mark'
import { PaintStrip } from '../components/PaintStrip'
import type { Adjust, Goal, Locks, MaskState, MoodReading, Palette, PaletteResult, Role, RoleColor, Room, RoomStyle } from '../lib/api'
import {
  ApiError,
  SCHEME_LABEL,
  createRoom,
  editMask,
  getMaskImage,
  getHealth,
  getPalettes,
  getStyle,
  modelName,
  getPreview,
  originalUrl,
  paintLabel,
  resetMask,
  undoMask,
} from '../lib/api'
import { checkSentence } from '../lib/checks'
import { go, takePendingPhoto } from '../lib/router'
import './Studio.css'

const GOALS: { id: Goal; label: string }[] = [
  { id: 'balanced', label: 'Balanced' },
  { id: 'calm', label: 'Calm' },
  { id: 'cosy', label: 'Cosy' },
  { id: 'bright', label: 'Bright' },
  { id: 'bold', label: 'Bold' },
]

const NO_ADJUST: Adjust = { warmth: 0, lightness: 0, saturation: 0 }

const MOOD_EXAMPLES = ['Calm coastal morning', 'Moody library', 'Warm terracotta, cosy', 'Fresh sage nursery']

type Phase = 'empty' | 'measuring' | 'ready' | 'error'
type Mode = 'compare' | 'fix'

export function Studio() {
  const [phase, setPhase] = useState<Phase>('empty')
  const [error, setError] = useState<string | null>(null)
  const [title, setTitle] = useState('Your room')
  const [localPhoto, setLocalPhoto] = useState<string | null>(null)
  const [room, setRoom] = useState<Room | null>(null)

  const [goal, setGoal] = useState<Goal | null>(null) // null: the mood or style decides
  const [mood, setMood] = useState('')
  const [adjust, setAdjust] = useState<Adjust>(NO_ADJUST)
  const [locks, setLocks] = useState<Locks>({})
  const [result, setResult] = useState<PaletteResult | null>(null)
  const [active, setActive] = useState(0)
  const [loadingPalettes, setLoadingPalettes] = useState(false)
  const [llm, setLlm] = useState<string>('none') // which language model the engine can use
  const [writing, setWriting] = useState(false) // the model is writing explanations
  const [style, setStyle] = useState<RoomStyle | null>(null)
  const [styleState, setStyleState] = useState<'idle' | 'reading' | 'done' | 'failed'>('idle')

  const [preview, setPreview] = useState<string | null>(null)
  const previews = useRef(new Map<string, string>())
  const maskVersion = useRef(0)

  const [mode, setMode] = useState<Mode>('compare')
  const [maskUrl, setMaskUrl] = useState<string | null>(null)
  const [maskAction, setMaskAction] = useState<'exclude' | 'include'>('exclude')
  const [mask, setMask] = useState<MaskState | null>(null)
  const [maskBusy, setMaskBusy] = useState(false)
  const [maskNote, setMaskNote] = useState<string | null>(null)

  const fail = (e: unknown) => {
    setError(e instanceof ApiError ? e.message : 'Something went wrong. Try again.')
    setPhase((p) => (p === 'measuring' ? 'error' : p))
  }

  const clearPreviews = () => {
    previews.current.forEach((u) => URL.revokeObjectURL(u))
    previews.current.clear()
  }

  /* ---------- upload ---------- */
  const start = useCallback(async (file: Blob, name: string, label: string) => {
    clearPreviews()
    setError(null)
    setResult(null)
    setRoom(null)
    setPreview(null)
    setMode('compare')
    setMask(null)
    setLocks({})
    setAdjust(NO_ADJUST)
    setGoal(null)
    setMood('')
    setStyle(null)
    setStyleState('idle')
    setTitle(label)
    setLocalPhoto(URL.createObjectURL(file))
    setPhase('measuring')
    try {
      const r = await createRoom(file, name)
      setRoom(r)
      setPhase('ready')
    } catch (e) {
      fail(e)
    }
  }, [])

  // React's development mode runs mount effects twice; uploading twice would also queue
  // the local model's work twice, so the first photo of a visit is started only once.
  const started = useRef(false)
  useEffect(() => {
    if (started.current) return
    started.current = true
    const p = takePendingPhoto()
    if (p) {
      start(p.file, p.name, p.title)
      return
    }
    // Deep link for demos: #/studio?sample=dining_room opens a sample room directly.
    const sample = new URLSearchParams(window.location.hash.split('?')[1] ?? '').get('sample')
    if (sample && /^[a-z_]+$/.test(sample)) {
      fetch(`/demo/rooms/${sample}/original.jpg`)
        .then((r) => (r.ok ? r.blob() : Promise.reject()))
        .then((b) => start(b, `${sample}.jpg`, sample.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())))
        .catch(() => setError('That sample room does not exist. Choose a photo instead.'))
    }
  }, [start])

  /* ---------- language model: is one running, and what does it see? ---------- */
  // Asked again for every room, so a model started (or stopped) after the page opened is noticed.
  useEffect(() => {
    if (!room) return
    getHealth()
      .then((h) => setLlm(h.llm))
      .catch(() => setLlm('none'))
  }, [room])

  useEffect(() => {
    if (!room || llm === 'none') return
    let alive = true
    setStyleState('reading')
    getStyle(room.room_id)
      .then((s) => {
        if (!alive) return
        setStyle(s)
        setStyleState('done')
      })
      .catch(() => alive && setStyleState('failed'))
    return () => {
      alive = false
    }
  }, [room, llm])

  /* ---------- palettes ---------- */
  useEffect(() => {
    if (!room) return
    let alive = true
    setLoadingPalettes(true)
    const t = window.setTimeout(async () => {
      try {
        const opts = { goal: goal ?? undefined, adjust, locks, mood: mood || undefined, use_style: false }
        // 1. Instant: palettes with explanations written from the measured facts.
        const res = await getPalettes(room.room_id, { ...opts, explain_with: 'template' })
        if (!alive) return
        setResult(res)
        setActive(0)
        setLoadingPalettes(false)
        // 2. If a language model is running, ask it for fuller wording and swap it in
        //    when it lands, as long as the palettes on screen are still the same ones.
        if (llm === 'none' || !res.palettes.length) return
        setWriting(true)
        const fuller = await getPalettes(room.room_id, { ...opts, explain_with: 'auto' })
        if (!alive || fuller.explanation_source === 'template') return
        const key = (r: PaletteResult) => r.palettes.map((p) => p.roles.dominant.hex).join()
        setResult((cur) =>
          cur && key(cur) === key(fuller)
            ? { ...cur, explanation_source: fuller.explanation_source,
                palettes: cur.palettes.map((p, i) => ({ ...p, explanation: fuller.palettes[i].explanation })) }
            : cur,
        )
      } catch (e) {
        if (alive) fail(e)
      } finally {
        if (alive) {
          setLoadingPalettes(false)
          setWriting(false)
        }
      }
    }, 180) // let quick stepper clicks settle into one request
    return () => {
      alive = false
      window.clearTimeout(t)
    }
  }, [room, goal, adjust, locks, mood, llm])

  const palette: Palette | undefined = result?.palettes[active]

  /* ---------- previews (cached per colour, invalidated by mask edits) ---------- */
  const previewFor = useCallback(
    async (p: Palette): Promise<string | null> => {
      if (!room) return null
      const ceiling = room.paint_ceiling_by_default ? p.roles.ceiling.hex : undefined
      const key = `${maskVersion.current}|${p.roles.dominant.hex}|${ceiling ?? ''}`
      const hit = previews.current.get(key)
      if (hit) return hit
      const { url } = await getPreview(room.room_id, p.roles.dominant.hex, ceiling)
      previews.current.set(key, url)
      return url
    },
    [room],
  )

  useEffect(() => {
    if (!palette) return
    let alive = true
    previewFor(palette)
      .then((u) => alive && setPreview(u))
      .catch(fail)
    // Warm the cache for the other palettes so switching is instant.
    result?.palettes.forEach((p) => p !== palette && previewFor(p).catch(() => undefined))
    return () => {
      alive = false
    }
  }, [palette, previewFor, result])

  /* ---------- click-to-fix ---------- */
  const refreshMask = useCallback(async () => {
    if (!room) return
    const url = await getMaskImage(room.room_id)
    setMaskUrl((old) => {
      if (old) URL.revokeObjectURL(old)
      return url
    })
  }, [room])

  useEffect(() => {
    if (mode === 'fix') refreshMask().catch(fail)
  }, [mode, refreshMask])

  const afterMaskChange = async (state: MaskState) => {
    setMask(state)
    maskVersion.current += 1
    await refreshMask()
    if (palette) setPreview(await previewFor(palette))
  }

  const onCanvasClick = async (e: React.MouseEvent<HTMLDivElement>) => {
    if (mode !== 'fix' || !room || maskBusy) return
    const rect = e.currentTarget.getBoundingClientRect()
    const x = Math.round(((e.clientX - rect.left) / rect.width) * room.width)
    const y = Math.round(((e.clientY - rect.top) / rect.height) * room.height)
    setMaskBusy(true)
    setMaskNote(null)
    try {
      const state = await editMask(room.room_id, x, y, maskAction)
      if (state.applied === false) setMaskNote(state.reason ?? 'That click was not applied.')
      else await afterMaskChange(state)
    } catch (err) {
      fail(err)
    } finally {
      setMaskBusy(false)
    }
  }

  const maskStep = async (fn: (id: string) => Promise<MaskState>) => {
    if (!room) return
    setMaskBusy(true)
    try {
      await afterMaskChange(await fn(room.room_id))
    } catch (err) {
      fail(err)
    } finally {
      setMaskBusy(false)
    }
  }

  /* ---------- locks ---------- */
  const toggleLock = (role: 'dominant' | 'accent', hex: string) =>
    setLocks((l) => (l[role] === hex ? { ...l, [role]: undefined } : { ...l, [role]: hex }))

  /* ---------- render ---------- */
  const ratio = room ? room.width / room.height : 1.5

  return (
    <div className="studio">
      <header className="studio-bar">
        <a
          className="studio-bar__home"
          href="#/"
          onClick={(e) => {
            e.preventDefault()
            go('landing')
          }}
        >
          <ArrowLeft size={16} aria-hidden="true" />
          <Wordmark />
        </a>
        <p className="studio-bar__title">{phase === 'empty' ? 'Studio' : title}</p>
        {phase !== 'empty' && <NewPhotoButton onFile={(f) => start(f, f.name, f.name.replace(/\.[^.]+$/, ''))} />}
      </header>

      {phase === 'empty' && (
        <section className="studio-empty">
          <h1 className="studio-empty__title">Start with a photo of the room.</h1>
          <p className="studio-empty__copy">
            Stand in a corner, get two walls and some floor in frame, and use daylight if you can. Roomtone does the
            rest in a few seconds.
          </p>
          <Dropzone onFile={(f) => start(f, f.name, f.name.replace(/\.[^.]+$/, ''))} />
          <button type="button" className="link-btn" onClick={() => go('landing')}>
            Or go back and open a sample room
          </button>
        </section>
      )}

      {phase !== 'empty' && (
        <div className="studio-main">
          {/* ---------- canvas ---------- */}
          <section className="canvas" aria-label="Room photo">
            <div className="canvas__tools">
              <div className="seg" role="radiogroup" aria-label="Canvas mode">
                <button type="button" role="radio" aria-checked={mode === 'compare'} className={mode === 'compare' ? 'is-on' : ''} onClick={() => setMode('compare')} disabled={!room}>
                  <Columns2 size={16} aria-hidden="true" /> Compare
                </button>
                <button type="button" role="radio" aria-checked={mode === 'fix'} className={mode === 'fix' ? 'is-on' : ''} onClick={() => setMode('fix')} disabled={!room}>
                  <Eraser size={16} aria-hidden="true" /> Fix what’s painted
                </button>
              </div>
              {mode === 'fix' && (
                <div className="canvas__fix">
                  <div className="seg seg--small" role="radiogroup" aria-label="What a click does">
                    <button type="button" role="radio" aria-checked={maskAction === 'exclude'} className={maskAction === 'exclude' ? 'is-on is-rust' : ''} onClick={() => setMaskAction('exclude')}>
                      Leave out
                    </button>
                    <button type="button" role="radio" aria-checked={maskAction === 'include'} className={maskAction === 'include' ? 'is-on' : ''} onClick={() => setMaskAction('include')}>
                      Paint too
                    </button>
                  </div>
                  <button type="button" className="icon-btn" onClick={() => maskStep(undoMask)} disabled={maskBusy || !mask?.edits} aria-label="Undo last click">
                    <Undo2 size={16} aria-hidden="true" /> Undo
                  </button>
                  <button type="button" className="icon-btn" onClick={() => maskStep(resetMask)} disabled={maskBusy || !mask?.edits} aria-label="Reset all clicks">
                    <RotateCcw size={16} aria-hidden="true" /> Reset
                  </button>
                </div>
              )}
            </div>

            <div className="canvas__stage">
              <div
                className={`canvas__frame${mode === 'fix' ? ' is-fixing' : ''}${maskBusy ? ' is-busy' : ''}`}
                style={{ width: `min(100%, calc((100vh - 230px) * ${ratio}))`, aspectRatio: `${ratio}` }}
                onClick={onCanvasClick}
              >
                {phase === 'measuring' && localPhoto && (
                  <div className="scan">
                    <img src={localPhoto} alt="Your photo, being measured" />
                    <span className="scan__line" aria-hidden="true" />
                    <p className="scan__label" role="status">
                      Measuring walls, floor and light
                    </p>
                  </div>
                )}

                {phase === 'error' && localPhoto && <img src={localPhoto} alt="Your photo" className="canvas__img is-dim" />}

                {room && mode === 'compare' && (
                  <BeforeAfter
                    className="canvas__ba"
                    before={originalUrl(room.room_id)}
                    after={preview ?? originalUrl(room.room_id)}
                    alt={title}
                    beforeLabel="As photographed"
                    afterLabel={palette ? palette.roles.dominant.name || palette.roles.dominant.code || 'Preview' : 'Preview'}
                  />
                )}

                {room && mode === 'fix' && maskUrl && (
                  <img src={maskUrl} alt="Your photo with the area that will be painted shown in magenta" className="canvas__img" draggable={false} />
                )}
              </div>
            </div>

            <p className="canvas__hint" role="status">
              {mode === 'fix'
                ? maskNote ??
                  (maskBusy
                    ? 'Working out the object you clicked…'
                    : `Magenta is what gets painted. Click an object to ${maskAction === 'exclude' ? 'leave it out' : 'paint it too'}.${mask?.edits ? ` ${mask.edits} click${mask.edits > 1 ? 's' : ''} so far.` : ''}`)
                : room
                  ? 'Drag the divider, or focus it and use the arrow keys.'
                  : ''}
            </p>

            {room && <RoomReadout room={room} style={style} styleState={styleState} />}
          </section>

          {/* ---------- panel ---------- */}
          <aside className="panel" aria-label="Palettes">
            {error && (
              <div className="alert" role="alert">
                <p>{error}</p>
                {phase === 'error' && (
                  <button type="button" className="pill" onClick={() => setPhase('empty')}>
                    Choose another photo
                  </button>
                )}
              </div>
            )}

            {phase === 'measuring' && <PanelSkeleton />}

            {room && (
              <>
                <div className="panel__block">
                  <h2 className="panel__h plus">What should the room feel like?</h2>
                  <MoodInput
                    applied={mood}
                    reading={result?.mood ?? null}
                    onApply={(text) => {
                      setMood(text)
                      setGoal(null)
                    }}
                  />
                  <div className="goals" role="radiogroup" aria-label="Goal">
                    {GOALS.map((g) => {
                      const on = (goal ?? result?.goal) === g.id
                      return (
                        <button key={g.id} type="button" role="radio" aria-checked={on} className={`goal${on ? ' is-on' : ''}`} onClick={() => setGoal(g.id)}>
                          {g.label}
                        </button>
                      )
                    })}
                  </div>
                </div>

                <div className={`panel__block palettes${loadingPalettes ? ' is-loading' : ''}`} aria-busy={loadingPalettes}>
                  <h2 className="panel__h plus">
                    Palettes <span className="panel__count num">{result ? `${result.palettes.length} of ${result.candidates_considered} candidates` : ''}</span>
                  </h2>
                  {result?.palettes.length === 0 && <p className="panel__empty">No palette passed the checks with these locks. Unlock a colour or change the goal.</p>}
                  {result && result.palettes.length > 0 && (
                    <div className="switcher" role="tablist" aria-label="Suggested palettes">
                      {result.palettes.map((p, i) => (
                        <button
                          key={`${p.scheme}-${p.roles.dominant.hex}-${i}`}
                          type="button"
                          role="tab"
                          aria-selected={i === active}
                          className={`switcher__tab${i === active ? ' is-on' : ''}`}
                          onClick={() => setActive(i)}
                        >
                          <PaintStrip
                            size="sm"
                            showText={false}
                            dominant={{ hex: p.roles.dominant.hex, label: '' }}
                            secondary={{ hex: p.roles.secondary.hex, label: '' }}
                            accent={{ hex: p.roles.accent.hex, label: '' }}
                          />
                          <span className="switcher__name">{p.roles.dominant.name || p.roles.dominant.code}</span>
                          <span className="switcher__n num">{String(i + 1).padStart(2, '0')}</span>
                        </button>
                      ))}
                    </div>
                  )}
                  {palette && <PaletteDetail palette={palette} locks={locks} onLock={toggleLock} source={result?.explanation_source} writing={writing} />}
                </div>

                <Refine adjust={adjust} onChange={setAdjust} locks={locks} onClearLocks={() => setLocks({})} />
              </>
            )}
          </aside>
        </div>
      )}
    </div>
  )
}

/* ---------- pieces ---------- */

function MoodInput({ applied, reading, onApply }: { applied: string; reading: MoodReading | null; onApply: (text: string) => void }) {
  const [draft, setDraft] = useState(applied)
  useEffect(() => setDraft(applied), [applied])
  const fromModel = reading && reading.source !== 'keywords'
  return (
    <div className="mood">
      <form
        className="mood__form"
        onSubmit={(e) => {
          e.preventDefault()
          onApply(draft.trim())
        }}
      >
        <label htmlFor="mood-input" className="sr-only">
          Describe the mood in your own words
        </label>
        <input
          id="mood-input"
          className="mood__input"
          type="text"
          value={draft}
          maxLength={200}
          placeholder="In your words: calm coastal morning"
          onChange={(e) => setDraft(e.target.value)}
        />
        <button type="submit" className="pill pill--solid mood__go" disabled={draft.trim() === applied}>
          <Sparkles size={16} aria-hidden="true" /> Apply
        </button>
      </form>
      {!applied && (
        <div className="mood__examples" aria-label="Example moods">
          {MOOD_EXAMPLES.map((m) => (
            <button key={m} type="button" className="link-chip" onClick={() => onApply(m)}>
              {m}
            </button>
          ))}
        </div>
      )}
      {applied && reading && (
        <div className="mood__reading" role="status">
          {reading.summary ? (
            <>
              <span className="mood__label">Read as</span>
              <span className="mood__summary">{reading.summary}</span>
              {reading.hues.map((h) => (
                <span key={h.word} className="mood__hue" title={`${h.word} family`}>
                  <span className="swatch" style={{ background: h.hex }} />
                </span>
              ))}
              <span className="mood__source">{fromModel ? `read by ${reading.source.split(':').pop()}` : 'from your words'}</span>
            </>
          ) : (
            <span className="mood__none">
              No mood words recognised{reading.ignored.length ? ` in "${reading.ignored.slice(0, 3).join(', ')}"` : ''}. Try words like calm, cosy, coastal, moody, sage or terracotta.
            </span>
          )}
          <button type="button" className="link-btn mood__clear" onClick={() => onApply('')}>
            Clear
          </button>
        </div>
      )}
    </div>
  )
}

function NewPhotoButton({ onFile }: { onFile: (f: File) => void }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <button type="button" className="pill studio-bar__new" onClick={() => input.current?.click()}>
        <Upload size={16} aria-hidden="true" /> New photo
      </button>
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = ''
        }}
      />
    </>
  )
}

function RoomReadout({ room, style, styleState }: { room: Room; style: RoomStyle | null; styleState: string }) {
  const roles = room.analysis.current_roles
  const order = ['dominant', 'secondary', 'accent'].filter((k) => roles[k])
  return (
    <div className="readout">
      <p className="readout__h">The room today</p>
      <ul className="readout__list">
        {order.map((k) => (
          <li key={k}>
            <span className="swatch" style={{ background: roles[k].hex }} />
            <span>
              {k === 'dominant' ? '60%' : k === 'secondary' ? '30%' : '10%'} {roles[k].source === 'overall' ? 'accent' : roles[k].source}
            </span>
            <span className="num readout__hex">{roles[k].hex}</span>
          </li>
        ))}
        <li>
          <Sun size={16} strokeWidth={1.75} aria-hidden="true" className="readout__icon" />
          <span>
            Light: {room.analysis.lighting.brightness}, {room.analysis.lighting.temperature}
          </span>
        </li>
        <li>
          <PaintRoller size={16} strokeWidth={1.75} aria-hidden="true" className="readout__icon" />
          <span>
            Walls: <span className="num">{Math.round(room.paintable.wall * 100)}%</span> of the photo
          </span>
        </li>
        {styleState !== 'idle' && styleState !== 'failed' && (
          <li className="readout__style">
            <Sofa size={16} strokeWidth={1.75} aria-hidden="true" className="readout__icon" />
            {style ? (
              <span title={style.design_notes}>
                {style.style !== 'other' ? `${style.style} ` : ''}
                {style.room_type}, {style.mood}
                <span className="readout__hex"> · read by {modelName(style.source)}</span>
              </span>
            ) : (
              <span className="readout__pending">Reading the room's style…</span>
            )}
          </li>
        )}
      </ul>
    </div>
  )
}

const ROLE_NAME: Record<Role, string> = { dominant: 'Walls', secondary: 'Keep', accent: 'Accent', ceiling: 'Ceiling & trim' }

function PaletteDetail({
  palette,
  locks,
  onLock,
  source,
  writing,
}: {
  palette: Palette
  locks: Locks
  onLock: (role: 'dominant' | 'accent', hex: string) => void
  source?: string
  writing?: boolean
}) {
  const { roles, explanation: ex } = palette
  const rows: Role[] = ['dominant', 'secondary', 'accent', 'ceiling']
  const passed = palette.checks.filter((c) => c.passed)
  const warnings = palette.checks.filter((c) => !c.passed)
  const fromTemplate = !source || source === 'template'
  const name = ex?.name ?? SCHEME_LABEL[palette.scheme]
  const scheme = SCHEME_LABEL[palette.scheme]
  const showScheme = name.toLowerCase().replace(/[^a-z]/g, '') !== scheme.toLowerCase().replace(/[^a-z]/g, '')

  return (
    <article className="pal" aria-label={name}>
      <div className="pal__rail">
        <header className="pal__head">
          <h3 className="pal__name">{name}</h3>
          {showScheme && <span className="pal__scheme">{scheme} scheme</span>}
        </header>
        <PaintStrip
          dominant={{ hex: roles.dominant.hex, label: 'Walls', sub: roles.dominant.name || roles.dominant.code }}
          secondary={{ hex: roles.secondary.hex, label: roles.secondary.source.replace('existing ', 'Your '), sub: roles.secondary.hex }}
          accent={{ hex: roles.accent.hex, label: 'Accent' }}
        />
      </div>

      <ul className="paints">
        {rows.map((r) => (
          <PaintRow key={r} role={r} c={roles[r]} locked={locks[r as 'dominant' | 'accent'] === roles[r].hex} onLock={r === 'dominant' || r === 'accent' ? () => onLock(r, roles[r].hex) : undefined} />
        ))}
      </ul>

      {ex && (
        <div className="why">
          <p className="why__summary">{ex.summary}</p>
          <details className="why__more">
            <summary>Why this palette</summary>
          <dl className="why__list">
            <div>
              <dt>Why these walls</dt>
              <dd>{ex.why_walls}</dd>
            </div>
            <div>
              <dt>Why this accent</dt>
              <dd>{ex.why_accent}</dd>
            </div>
            <div>
              <dt>Where to use it</dt>
              <dd>{ex.where_to_use}</dd>
            </div>
            <div>
              <dt>Before you buy</dt>
              <dd>{ex.watch_out}</dd>
            </div>
          </dl>
          </details>
          <p className="why__source">
            {writing
              ? 'Written from the measured facts. A fuller explanation from the local model is on its way…'
              : fromTemplate
                ? 'Written from the measured facts. No language model is connected.'
                : `Written by ${modelName(source ?? '')} from the measured facts.`}
          </p>
        </div>
      )}

      <details className="checks-box">
        <summary>
          Passed {passed.length} of {palette.checks.length} checks{warnings.length ? `, ${warnings.length} note${warnings.length > 1 ? 's' : ''}` : ''}
        </summary>
        <ul>
          {palette.checks.map((c) => (
            <li key={c.rule} className={c.passed ? '' : 'is-warn'}>
              {checkSentence(c)}
            </li>
          ))}
        </ul>
      </details>
    </article>
  )
}

function PaintRow({ role, c, locked, onLock }: { role: Role; c: RoleColor; locked: boolean; onLock?: () => void }) {
  return (
    <li className="paint">
      <span className="paint__chip" style={{ background: c.hex }} />
      <span className="paint__text">
        <span className="paint__role">
          {ROLE_NAME[role]}
          {c.brand ? ` · ${c.brand}` : ''}
        </span>
        <span className="paint__name" title={c.brand ? paintLabel(c) : undefined}>
          {c.brand ? [c.name, c.code].filter(Boolean).join(' ') : `${c.source.replace('existing ', 'Your ')}, as it is`}
        </span>
      </span>
      <span className="paint__meta num">
        {c.hex}
        <br />
        LRV {c.lrv.toFixed(0)}
      </span>
      {onLock ? (
        <button type="button" className={`lock${locked ? ' is-locked' : ''}`} aria-pressed={locked} onClick={onLock} aria-label={locked ? `Unlock ${ROLE_NAME[role].toLowerCase()} colour` : `Keep this ${ROLE_NAME[role].toLowerCase()} colour and rebuild the rest`}>
          {locked ? <Lock size={16} aria-hidden="true" /> : <LockOpen size={16} aria-hidden="true" />}
        </button>
      ) : (
        <span className="lock lock--none" aria-hidden="true" />
      )}
    </li>
  )
}

const KNOBS: { key: keyof Adjust; less: string; more: string }[] = [
  { key: 'warmth', less: 'Cooler', more: 'Warmer' },
  { key: 'lightness', less: 'Darker', more: 'Lighter' },
  { key: 'saturation', less: 'Muted', more: 'Vivid' },
]

function Refine({ adjust, onChange, locks, onClearLocks }: { adjust: Adjust; onChange: (a: Adjust) => void; locks: Locks; onClearLocks: () => void }) {
  const lockCount = Object.values(locks).filter(Boolean).length
  const changed = KNOBS.some((k) => adjust[k.key] !== 0)
  return (
    <div className="panel__block refine">
      <h2 className="panel__h plus">Refine every suggestion</h2>
      {KNOBS.map((k) => {
        const v = adjust[k.key]
        return (
          <div className="knob" key={k.key}>
            <button type="button" className="knob__btn" onClick={() => onChange({ ...adjust, [k.key]: Math.max(-3, v - 1) })} disabled={v <= -3} aria-label={`${k.less} by one step`}>
              <Minus size={16} aria-hidden="true" />
            </button>
            <span className="knob__label">
              <span className={v < 0 ? 'is-on' : ''}>{k.less}</span>
              <span className="knob__track" aria-hidden="true">
                {[-3, -2, -1, 0, 1, 2, 3].map((s) => (
                  <i key={s} className={`${s === 0 ? 'mid' : ''}${(v < 0 && s >= v && s < 0) || (v > 0 && s <= v && s > 0) ? ' on' : ''}`} />
                ))}
              </span>
              <span className={v > 0 ? 'is-on' : ''}>{k.more}</span>
              <span className="sr-only" aria-live="polite">
                {v === 0 ? 'unchanged' : `${Math.abs(v)} step${Math.abs(v) > 1 ? 's' : ''} ${v > 0 ? k.more.toLowerCase() : k.less.toLowerCase()}`}
              </span>
            </span>
            <button type="button" className="knob__btn" onClick={() => onChange({ ...adjust, [k.key]: Math.min(3, v + 1) })} disabled={v >= 3} aria-label={`${k.more} by one step`}>
              <Plus size={16} aria-hidden="true" />
            </button>
          </div>
        )
      })}
      <div className="refine__foot">
        <span>{lockCount ? `${lockCount} colour${lockCount > 1 ? 's' : ''} locked. Palettes are rebuilt around ${lockCount > 1 ? 'them' : 'it'}.` : 'Lock a wall or accent colour to rebuild the rest around it.'}</span>
        {(changed || lockCount > 0) && (
          <button
            type="button"
            className="link-btn"
            onClick={() => {
              onChange(NO_ADJUST)
              onClearLocks()
            }}
          >
            Clear
          </button>
        )}
      </div>
    </div>
  )
}

function PanelSkeleton() {
  return (
    <div className="skeleton" aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}
