import type { MouseEvent } from 'react'
import type { Detection } from './api'
import { formatInterval, formatTime, matchColor } from './format'

// Las similitudes se concentran entre 93 % y 100 % (#113): con una escala de 0 a 100 %
// todas las barras se verían llenas, así que la escala empieza en 80 %.
const SCALE_MIN = 80

interface TimelineProps {
  detection: Detection
  currentTime: number
  onSeek: (seconds: number) => void
}

function position(seconds: number, duration: number): string {
  return `${(100 * seconds) / duration}%`
}

function barHeight(similarity: number | null): number {
  if (similarity === null) return 0
  return Math.max(4, Math.min(100, ((similarity - SCALE_MIN) / (100 - SCALE_MIN)) * 100))
}

function ticks(duration: number): number[] {
  const step = duration <= 40 ? 5 : duration <= 120 ? 15 : duration <= 300 ? 30 : 60
  const values = []
  for (let t = 0; t <= duration + 1e-9; t += step) values.push(t)
  return values
}

function Timeline({ detection, currentTime, onSeek }: TimelineProps) {
  const { duration_s: duration, timeline, matches, settings } = detection
  const thresholdPct = settings.threshold * 100

  function seek(event: MouseEvent<HTMLDivElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    onSeek(Math.max(0, Math.min(duration, ((event.clientX - box.left) / box.width) * duration)))
  }

  const playhead = <div className="playhead" style={{ left: position(currentTime, duration) }} />

  return (
    <section className="card">
      <h2 className="section-title">Línea de tiempo</h2>
      <p className="muted">Haz clic en la línea para escuchar ese instante.</p>

      <div className="timeline"
           role="img"
           aria-label={`Línea de tiempo de ${formatTime(duration)} con ${matches.length} posibles coincidencias`}>
        <div className="timeline-row">
          <span className="timeline-label">Coincidencias</span>
          <div className="timeline-track" onClick={seek}>
            {matches.map((match) => (
              <div key={match.match_id}
                   className="timeline-match"
                   style={{
                     left: position(match.query_start_s, duration),
                     width: position(match.query_end_s - match.query_start_s, duration),
                     background: matchColor(match.match_id),
                   }}
                   title={`#${match.match_id} · ${formatInterval(match.query_start_s, match.query_end_s)} → `
                          + `${match.similarity_pct} % similar a "${match.title ?? 'Sin título'}"`}>
                #{match.match_id}
              </div>
            ))}
            {matches.length === 0 && <span className="timeline-empty">Ninguna</span>}
            {playhead}
          </div>
        </div>

        <div className="timeline-row">
          <span className="timeline-label">Similitud por segmento</span>
          <div className="timeline-track timeline-bars" onClick={seek}>
            {timeline.map((segment, i) => {
              // los segmentos se solapan: cada barra ocupa hasta el inicio del siguiente
              const nextStart = timeline[i + 1]?.start_s ?? segment.end_s
              const color = segment.match_id !== null
                ? matchColor(segment.match_id)
                : segment.above_threshold ? 'var(--bar-strong)' : 'var(--bar-weak)'
              return (
                <div key={segment.segment_index}
                     className="bar-slot"
                     style={{ left: position(segment.start_s, duration), width: position(nextStart - segment.start_s, duration) }}
                     title={`${formatInterval(segment.start_s, segment.end_s)} · ${segment.best_similarity_pct ?? '—'} % con `
                            + `"${segment.best_title ?? 'sin candidato'}"${segment.best_artist ? ` de ${segment.best_artist}` : ''}`}>
                  <div className="bar" style={{ height: `${barHeight(segment.best_similarity_pct)}%`, background: color }} />
                </div>
              )
            })}
            <div className="threshold-line" style={{ bottom: `${barHeight(thresholdPct)}%` }} />
            {playhead}
          </div>
        </div>

        <div className="timeline-row">
          <span className="timeline-label" />
          <div className="timeline-axis">
            {ticks(duration).map((t) => (
              <span key={t} style={{ left: position(t, duration) }}>{formatTime(t)}</span>
            ))}
          </div>
        </div>
      </div>

      <ul className="legend">
        {matches.map((match) => (
          <li key={match.match_id}>
            <span className="swatch" style={{ background: matchColor(match.match_id) }} />
            #{match.match_id} {match.title ?? 'Sin título'}
          </li>
        ))}
        <li><span className="swatch" style={{ background: 'var(--bar-strong)' }} />Sobre el umbral, sin coincidencia</li>
        <li><span className="swatch" style={{ background: 'var(--bar-weak)' }} />Bajo el umbral</li>
        <li><span className="swatch swatch-line" />Umbral ({thresholdPct} %) · escala de {SCALE_MIN} a 100 %</li>
      </ul>
    </section>
  )
}

export default Timeline
