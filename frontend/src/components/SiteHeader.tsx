import { go, jumpTo } from '../lib/router'
import { Wordmark } from './Mark'
import './SiteHeader.css'

const LINKS: [string, string][] = [
  ['derive', 'How it works'],
  ['light', 'The preview'],
  ['rooms', 'Sample rooms'],
]

export function SiteHeader() {
  return (
    <header className="site-header">
      <nav className="site-header__nav" aria-label="Page sections">
        {LINKS.map(([id, label]) => (
          <a
            key={id}
            href={`#${id}`}
            className="site-header__link plus"
            onClick={(e) => {
              e.preventDefault()
              jumpTo(id)
            }}
          >
            {label}
          </a>
        ))}
      </nav>
      <a
        className="site-header__brand"
        href="#/"
        aria-label="Roomtone, home"
        onClick={(e) => {
          e.preventDefault()
          window.scrollTo({ top: 0, behavior: 'smooth' })
        }}
      >
        <Wordmark />
      </a>
      <button type="button" className="pill site-header__cta" onClick={() => go('studio')}>
        Open the studio
      </button>
    </header>
  )
}
