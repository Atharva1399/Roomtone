// Real pipeline output exported by backend/scripts/export_demo.py into /public/demo.
import { useEffect, useState } from 'react'
import type { Role } from './api'

export interface DemoRole {
  hex: string
  lrv: number
  source: string
  use: string
  brand?: string
  name?: string
  code?: string
}

export interface DemoPalette {
  scheme: string
  facts: string[]
  roles: Record<Role, DemoRole>
  checks: { rule: string; passed: boolean; detail: string }[]
}

export interface DemoRoom {
  title: string
  size: [number, number]
  lighting: { brightness: string; temperature: string; median_L: number; median_b: number }
  current_roles: Record<string, { hex: string; source: string }>
  base_hue_reason: string
  candidates_considered: number
  excluded_clicks: number
  palettes: DemoPalette[]
}

export interface Demo {
  counts: { paints: number; room_classes: number; harmony_schemes: number }
  hero: string
  light_demo: string
  rooms: Record<string, DemoRoom>
}

export const roomImage = (room: string, file: string) => `/demo/rooms/${room}/${file}`

export function useDemo(): Demo | null {
  const [demo, setDemo] = useState<Demo | null>(null)
  useEffect(() => {
    let alive = true
    fetch('/demo/demo.json')
      .then((r) => r.json())
      .then((d: Demo) => alive && setDemo(d))
      .catch(() => undefined)
    return () => {
      alive = false
    }
  }, [])
  return demo
}

export const PHOTO_CREDITS: { room: string; title: string; href: string; licence: string }[] = [
  {
    room: 'dining_room',
    title: 'Dining Room 556',
    href: 'https://commons.wikimedia.org/wiki/File:Dining_Room_556.jpg',
    licence: 'CC BY-SA 4.0',
  },
  {
    room: 'bedroom_modern',
    title: 'Bedroom, Interior of apartment in Brisbane, 2025, 01',
    href: 'https://commons.wikimedia.org/wiki/File:Bedroom,_Interior_of_apartment_in_Brisbane,_2025,_01.jpg',
    licence: 'CC BY-SA 4.0',
  },
  {
    room: 'bedroom_historic',
    title: 'Bedroom in Montacute House',
    href: 'https://commons.wikimedia.org/wiki/File:Bedroom_in_Montacute_House_(8601909505).jpg',
    licence: 'CC BY 2.0',
  },
  {
    room: 'living_luxury',
    title: 'Modern luxury living room with kitchen interior',
    href: 'https://commons.wikimedia.org/wiki/File:Modern_luxury_living_room_with_kitchen_interior.jpg',
    licence: 'CC BY 2.0',
  },
]
