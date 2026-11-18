import type { Detection } from './api'
import { formatTime } from './format'

function VerdictBanner({ detection }: { detection: Detection }) {
  const possible = detection.verdict === 'posible_coincidencia'
  return (
    <section className={`card verdict ${possible ? 'verdict-possible' : 'verdict-none'}`} aria-live="polite">
      <h2>{possible ? 'Posible coincidencia' : 'Sin coincidencias'}</h2>
      <p>{detection.message}</p>
      <p className="muted">
        {detection.filename} · {formatTime(detection.duration_s)} ·{' '}
        {possible ? `las coincidencias cubren el ${detection.coverage_pct} % del audio` : 'ningún tramo superó el umbral'}
      </p>
      <p className="disclaimer">{detection.disclaimer}</p>
    </section>
  )
}

export default VerdictBanner
