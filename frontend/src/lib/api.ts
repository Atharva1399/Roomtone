// Typed client for the FastAPI backend (proxied at /api by Vite in development).

export type Goal = 'balanced' | 'calm' | 'cosy' | 'bright' | 'bold'
export type Role = 'dominant' | 'secondary' | 'accent' | 'ceiling'

export interface RoleColor {
  hex: string
  lab: number[]
  lrv: number
  source: string
  share: number
  use: string
  brand?: string
  name?: string
  code?: string
  match_delta_e?: number
}

export interface Check {
  rule: string
  passed: boolean
  detail: string
  hard: boolean
}

export interface Explanation {
  name: string
  summary: string
  why_walls: string
  why_accent: string
  where_to_use: string
  watch_out: string
  source: string
}

export interface Palette {
  scheme: string
  score: number
  roles: Record<Role, RoleColor>
  checks: Check[]
  facts: string[]
  explanation?: Explanation
}

export interface Adjust {
  warmth: number
  lightness: number
  saturation: number
}

export interface Locks {
  dominant?: string
  accent?: string
}

export interface MoodReading {
  text: string
  goal: Goal | null
  adjust: Partial<Adjust>
  hues: { word: string; hex: string; hue: number }[]
  neutral: boolean
  matched: string[]
  ignored: string[]
  source: string
  summary: string
}

export interface PaletteResult {
  goal: Goal
  goal_source: string
  mood?: MoodReading | null
  style: string | null
  base_hue_reason: string
  palettes: Palette[]
  candidates_considered: number
  paint_ceiling_by_default: boolean
  explanation_source?: string
  explanation_note?: string
}

export interface Swatch {
  hex: string
  lrv: number
  area?: number
}

export interface RoomAnalysis {
  regions: Record<string, Swatch & { area: number }>
  current_roles: Record<string, Swatch & { source: string }>
  lighting: { brightness: string; temperature: string }
}

export interface Room {
  room_id: string
  width: number
  height: number
  paintable: Record<string, number>
  paint_ceiling_by_default: boolean
  analysis: RoomAnalysis
}

export interface MaskState {
  applied?: boolean
  reason?: string
  group: string
  edits: number
  coverage: number
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError(0, 'The Roomtone engine is not reachable. Start the backend on port 8000, then try again.')
  }
  if (!res.ok) {
    let detail = res.statusText
    let fromEngine = false
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
      fromEngine = true
    } catch {
      /* body was not JSON */
    }
    // The dev proxy answers 5xx with no JSON when the engine is down or still starting.
    if (!fromEngine && res.status >= 500) {
      throw new ApiError(0, 'The Roomtone engine is not reachable yet. Start the backend on port 8000 (or wait for it to finish starting), then try again.')
    }
    if (res.status === 404) detail = 'This room is no longer in memory. Upload the photo again.'
    if (res.status >= 500 && res.status !== 503) detail = 'The engine hit an error on this photo. Try again, or try another photo.'
    throw new ApiError(res.status, detail)
  }
  return res
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export async function createRoom(file: Blob, filename = 'room.jpg'): Promise<Room> {
  const form = new FormData()
  form.append('file', file, filename)
  return (await request('/api/rooms', { method: 'POST', body: form })).json()
}

export interface PaletteOptions {
  goal?: Goal
  n?: number
  explain?: boolean
  /** template: instant, from the facts; auto: the language model when one is running */
  explain_with?: 'auto' | 'template'
  use_style?: boolean
  locks?: Locks
  adjust?: Adjust
  mood?: string
}

export async function getPalettes(roomId: string, opts: PaletteOptions): Promise<PaletteResult> {
  return (await request(`/api/rooms/${roomId}/palettes`, json({ n: 4, explain: true, ...opts }))).json()
}

export interface Health {
  status: string
  gpu: boolean
  llm: string // "local:gemma3:4b", "huggingface:…", "claude:…" or "none"
  paints: number
}

export async function getHealth(): Promise<Health> {
  return (await request('/api/health')).json()
}

export interface RoomStyle {
  room_type: string
  style: string
  secondary_style: string
  mood: string
  natural_light: string
  suggested_goal: Goal
  design_notes: string
  source: string
}

export async function getStyle(roomId: string): Promise<RoomStyle> {
  return (await request(`/api/rooms/${roomId}/style`, { method: 'POST' })).json()
}

/** "local:gemma3:4b" -> "gemma3:4b"; "claude:claude-opus-5-5" -> "claude-opus-5-5" */
export function modelName(source: string): string {
  return source.slice(source.indexOf(':') + 1)
}

export async function getPreview(roomId: string, wall: string, ceiling?: string): Promise<{ url: string; deltaE: string }> {
  const res = await request(`/api/rooms/${roomId}/preview`, json({ wall, ceiling, quality: 90 }))
  const blob = await res.blob()
  return { url: URL.createObjectURL(blob), deltaE: res.headers.get('X-Paint-Delta-E') ?? '' }
}

export async function getMaskImage(roomId: string): Promise<string> {
  const res = await request(`/api/rooms/${roomId}/mask?group=wall&t=${Date.now()}`)
  return URL.createObjectURL(await res.blob())
}

export async function editMask(roomId: string, x: number, y: number, mode: 'exclude' | 'include'): Promise<MaskState> {
  return (await request(`/api/rooms/${roomId}/mask`, json({ group: 'wall', x, y, mode, level: 'object' }))).json()
}

export async function undoMask(roomId: string): Promise<MaskState> {
  return (await request(`/api/rooms/${roomId}/mask/undo`, json({ group: 'wall' }))).json()
}

export async function resetMask(roomId: string): Promise<MaskState> {
  return (await request(`/api/rooms/${roomId}/mask/reset`, json({ group: 'wall' }))).json()
}

export function originalUrl(roomId: string): string {
  return `/api/rooms/${roomId}/original`
}

export function paintLabel(c: Pick<RoleColor, 'brand' | 'name' | 'code' | 'source'>): string {
  const parts = [c.brand, c.name, c.code].filter(Boolean)
  return parts.length ? parts.join(' ') : c.source
}

export const SCHEME_LABEL: Record<string, string> = {
  tonal: 'Tonal',
  'analogous-warm': 'Analogous, warm side',
  'analogous-cool': 'Analogous, cool side',
  complementary: 'Complementary',
  'split-complementary': 'Split complementary',
  triadic: 'Triadic',
  'warm-neutral': 'Warm neutral',
  'cool-neutral': 'Cool neutral',
  'mood-match': 'Mood match',
}
