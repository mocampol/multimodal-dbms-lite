// Cliente de la API de detección (apps/audio-copyright/backend/detection_api.py).
const API_URL = import.meta.env.VITE_AUDIO_API_URL ?? 'http://localhost:8001'

export type Verdict = 'posible_coincidencia' | 'sin_coincidencias'
export type IndexType = 'hnsw' | 'ivf'

/** Un tramo del audio subido que se parece a un tramo de una canción de referencia. */
export interface Match {
  match_id: number
  song_id: number
  title?: string | null
  artist?: string | null
  license_family?: string | null
  track_url?: string | null
  query_start_s: number
  query_end_s: number
  reference_start_s: number
  reference_end_s: number
  similarity_pct: number
  max_similarity_pct: number
  segments: number
}

/** Un segmento del audio subido y su mejor candidato, para la línea de tiempo. */
export interface TimelineSegment {
  segment_index: number
  start_s: number
  end_s: number
  best_song_id: number | null
  best_title: string | null
  best_artist: string | null
  best_similarity_pct: number | null
  above_threshold: boolean
  match_id: number | null
}

export interface DetectionSettings {
  threshold: number
  min_segments: number
  max_gap: number
  offset_tolerance_s: number
  margin?: number
}

export interface Detection {
  filename: string
  verdict: Verdict
  message: string
  disclaimer: string
  duration_s: number
  coverage_pct: number
  matches: Match[]
  timeline: TimelineSegment[]
  settings: DetectionSettings
  search: { index_type?: string; metric?: string; k?: number; used_index?: boolean; model_name?: string }
  timings_ms: { extraction_ms?: number; dbms_search_ms?: number; total_ms?: number }
}

export interface Health {
  status: string
  ready: boolean
  error: string | null
  settings: DetectionSettings
}

export interface DetectionOptions {
  index?: IndexType
  threshold?: number   // entre 0 y 1
  minSegments?: number
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : 'Ocurrió un error inesperado'
}

/** FastAPI responde {detail: "..."} o, si falla la validación, {detail: [{msg: "..."}]}. */
async function failure(response: Response): Promise<ApiError> {
  const body = (await response.json().catch(() => null)) as { detail?: unknown } | null
  const detail = body?.detail
  const message = typeof detail === 'string'
    ? detail
    : Array.isArray(detail)
      ? detail.map((item) => (item as { msg?: string }).msg ?? '').filter(Boolean).join('; ')
      : response.statusText
  return new ApiError(response.status, message || `Error ${response.status}`)
}

async function send(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(`${API_URL}${path}`, init)
  } catch {
    throw new ApiError(0, `No se pudo conectar con la API de detección en ${API_URL}`)
  }
}

export async function getHealth(): Promise<Health> {
  const response = await send('/health')
  if (!response.ok) throw await failure(response)
  return response.json() as Promise<Health>
}

/** Envía el audio como cuerpo crudo de la petición, como espera POST /detections. */
export async function detect(file: File, options: DetectionOptions = {}): Promise<Detection> {
  const params = new URLSearchParams({ filename: file.name })
  if (options.index) params.set('index', options.index)
  if (options.threshold !== undefined) params.set('threshold', String(options.threshold))
  if (options.minSegments !== undefined) params.set('min_segments', String(options.minSegments))

  const response = await send(`/detections?${params}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  })
  if (!response.ok) throw await failure(response)
  return response.json() as Promise<Detection>
}

/** Una canción agregada a la base de referencia desde la aplicación. */
export interface UploadedSong {
  song_id: number
  title: string
  artist: string
  album_title: string
  genre: string
  license_family: string
  track_url: string
  duration_s: number
}

export interface SongList {
  count: number
  licenses: string[]
  songs: UploadedSong[]
}

export interface AddedSong extends UploadedSong {
  artist_id: number
  segments: number
  model_name: string
  timings_ms: { extraction_ms?: number; dbms_ms?: number; total_ms?: number }
}

export interface NewSong {
  title: string
  artist: string
  license: string
  album?: string
  genre?: string
  sourceUrl?: string
}

export async function listSongs(): Promise<SongList> {
  const response = await send('/songs')
  if (!response.ok) throw await failure(response)
  return response.json() as Promise<SongList>
}

/** Agrega una canción a la base: el audio va como cuerpo crudo y los datos en la URL. */
export async function addSong(file: File, song: NewSong): Promise<AddedSong> {
  const params = new URLSearchParams({ filename: file.name, title: song.title, artist: song.artist,
                                       license: song.license })
  if (song.album) params.set('album', song.album)
  if (song.genre) params.set('genre', song.genre)
  if (song.sourceUrl) params.set('source_url', song.sourceUrl)

  const response = await send(`/songs?${params}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream' },
    body: file,
  })
  if (!response.ok) throw await failure(response)
  return response.json() as Promise<AddedSong>
}
