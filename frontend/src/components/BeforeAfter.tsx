import { useEffect, useRef, useState } from 'react'
import './BeforeAfter.css'

interface Props {
  before: string
  after: string
  alt: string
  beforeLabel?: string
  afterLabel?: string
  /** Sweep the divider in once on mount: the page's one authored motion. */
  intro?: boolean
  /** Where the divider rests (percent from the left; the new paint shows to its right). */
  rest?: number
  className?: string
}

const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

export function BeforeAfter({ before, after, alt, beforeLabel = 'Before', afterLabel = 'After', intro, rest = 50, className }: Props) {
  const [pos, setPos] = useState(intro && !prefersReducedMotion() ? 96 : rest)
  const [afterReady, setAfterReady] = useState(false)
  const swept = useRef(false)

  // One sweep from right to centre, exponential ease-out, after the first image paints.
  useEffect(() => {
    if (!intro || swept.current || !afterReady || prefersReducedMotion()) return
    swept.current = true
    const from = 96
    const to = rest
    const dur = 1600
    const t0 = performance.now()
    let raf = 0
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / dur)
      const e = 1 - Math.pow(2, -10 * k)
      setPos(from + (to - from) * e)
      if (k < 1) raf = requestAnimationFrame(tick)
    }
    const delay = window.setTimeout(() => (raf = requestAnimationFrame(tick)), 450)
    // Frames can be throttled (background tabs, headless capture); always end at rest.
    const settle = window.setTimeout(() => {
      cancelAnimationFrame(raf)
      setPos(to)
    }, 450 + dur + 150)
    return () => {
      window.clearTimeout(delay)
      window.clearTimeout(settle)
      cancelAnimationFrame(raf)
    }
  }, [intro, afterReady, rest])

  return (
    <div className={`ba ${className ?? ''}`} style={{ '--pos': `${pos}%` } as React.CSSProperties}>
      <img className="ba__img" src={before} alt={`${alt}, before`} draggable={false} />
      <img
        key={after}
        className="ba__img ba__after"
        src={after}
        alt={`${alt}, after`}
        draggable={false}
        onLoad={() => setAfterReady(true)}
      />
      <span className="ba__tag ba__tag--before">{beforeLabel}</span>
      <span className="ba__tag ba__tag--after">{afterLabel}</span>
      <span className="ba__divider" aria-hidden="true">
        <span className="ba__knob">
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
            <path d="M6.5 4.5 2 9l4.5 4.5M11.5 4.5 16 9l-4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </span>
      </span>
      <input
        className="ba__range"
        type="range"
        min={0}
        max={100}
        step={0.5}
        value={pos}
        aria-label="Compare before and after: drag or use arrow keys"
        aria-valuetext={`${Math.round(100 - pos)}% showing the new paint`}
        onChange={(e) => setPos(Number(e.target.value))}
      />
    </div>
  )
}
