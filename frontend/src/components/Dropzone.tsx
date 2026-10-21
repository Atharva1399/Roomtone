import { useRef, useState } from 'react'
import { ImagePlus } from 'lucide-react'
import { openInStudio } from '../lib/router'
import './Dropzone.css'

const MAX_BYTES = 15 * 1024 * 1024

interface Props {
  tone?: 'light' | 'walnut'
  /** Inside the studio a file is handled in place; elsewhere it opens the studio. */
  onFile?: (file: File) => void
}

export function Dropzone({ tone = 'light', onFile }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const take = (file: File | undefined) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('That file is not a photo. Use a JPEG or PNG of a room.')
      return
    }
    if (file.size > MAX_BYTES) {
      setError('That photo is over 15 MB. Export a smaller copy and try again.')
      return
    }
    setError(null)
    if (onFile) onFile(file)
    else openInStudio({ file, name: file.name, title: file.name.replace(/\.[^.]+$/, '') })
  }

  return (
    <div className={`drop drop--${tone}`}>
      <button
        type="button"
        className={`drop__target${over ? ' is-over' : ''}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          take(e.dataTransfer.files[0])
        }}
      >
        <ImagePlus size={22} strokeWidth={1.5} aria-hidden="true" />
        <span className="drop__main">Drop a photo of your room</span>
        <span className="drop__sub">or click to choose one · JPEG or PNG, up to 15 MB</span>
      </button>
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => take(e.target.files?.[0])}
      />
      {error && (
        <p className="drop__error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}
