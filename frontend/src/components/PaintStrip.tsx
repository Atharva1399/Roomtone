// The signature element: a palette drawn at its real 60 / 30 / 10 proportions,
// like a strip cut from a material board, each band labelled in text.
import './PaintStrip.css'

interface Band {
  hex: string
  label: string
  sub?: string
}

interface Props {
  dominant: Band
  secondary: Band
  accent: Band
  size?: 'sm' | 'lg'
  showText?: boolean
}

/** Readable ink for text laid on a swatch (WCAG relative luminance). */
export function inkOn(hex: string): string {
  const c = hex.replace('#', '')
  const ch = [0, 2, 4].map((i) => {
    const v = parseInt(c.slice(i, i + 2), 16) / 255
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)
  })
  const L = 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
  return L > 0.22 ? '#16130F' : '#F9F7F3'
}

export function PaintStrip({ dominant, secondary, accent, size = 'lg', showText = true }: Props) {
  const bands: [Band, number, string][] = [
    [dominant, 60, '60%'],
    [secondary, 30, '30%'],
    [accent, 10, '10%'],
  ]
  return (
    <div className={`strip strip--${size}`} role="img" aria-label={bands.map(([b, , p]) => `${p} ${b.label} ${b.hex}`).join(', ')}>
      {bands.map(([b, w, p]) => (
        <div key={p} className="strip__band" style={{ flexGrow: w, background: b.hex, color: inkOn(b.hex) }}>
          {showText && (
            <>
              <span className="strip__pct num">{p}</span>
              <span className="strip__label">{b.label}</span>
              {b.sub && <span className="strip__sub">{b.sub}</span>}
            </>
          )}
        </div>
      ))}
    </div>
  )
}
