// Two screens, hash-routed so the static build works anywhere:
//   #/             landing
//   #/studio       studio (a room arrives via the in-memory handoff below)
import { useEffect, useState } from 'react'

export type Route = 'landing' | 'studio'

function read(): Route {
  return window.location.hash.startsWith('#/studio') ? 'studio' : 'landing'
}

/** In-page jump that leaves the route hash alone. */
export function jumpTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(read)
  useEffect(() => {
    const on = () => {
      setRoute((prev) => {
        const next = read()
        if (next !== prev) window.scrollTo(0, 0)
        return next
      })
    }
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}

export function go(route: Route) {
  window.location.hash = route === 'studio' ? '#/studio' : '#/'
}

// A photo picked on the landing page travels to the studio through here.
export interface PendingPhoto {
  file: Blob
  name: string
  title: string
  sample?: string
}

let pending: PendingPhoto | null = null

export function openInStudio(photo: PendingPhoto) {
  pending = photo
  go('studio')
}

export function takePendingPhoto(): PendingPhoto | null {
  const p = pending
  pending = null
  return p
}

export async function openSample(room: string, title: string) {
  const res = await fetch(`/demo/rooms/${room}/original.jpg`)
  openInStudio({ file: await res.blob(), name: `${room}.jpg`, title, sample: room })
}
