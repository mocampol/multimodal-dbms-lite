import type { Detection } from './api'
import { formatInterval, matchColor } from './format'

interface MatchListProps {
  detection: Detection
  onSeek: (seconds: number) => void
}

function MatchList({ detection, onSeek }: MatchListProps) {
  const { matches, settings, search, timings_ms: timings } = detection
  return (
    <section className="card">
      <h2 className="section-title">Detección</h2>

      {matches.length === 0 ? (
        <p className="muted">Ningún tramo se parece a una canción de referencia con los criterios usados.</p>
      ) : (
        <ul className="match-list">
          {matches.map((match) => (
            <li key={match.match_id} className="match">
              <span className="swatch" style={{ background: matchColor(match.match_id) }} />
              <div>
                <button type="button" className="link-button" onClick={() => onSeek(match.query_start_s)}
                        title="Escuchar este tramo">
                  {formatInterval(match.query_start_s, match.query_end_s)}
                </button>
                {' → '}<strong>{match.similarity_pct} %</strong> similar a{' '}
                <strong>"{match.title ?? 'Sin título'}"</strong>{match.artist ? ` de ${match.artist}` : ''}
                <p className="muted">
                  {formatInterval(match.reference_start_s, match.reference_end_s)} de la canción original ·{' '}
                  {match.segments} segmentos · máximo {match.max_similarity_pct} %
                  {match.license_family ? ` · licencia ${match.license_family}` : ''}
                  {match.track_url && (
                    <> · <a href={match.track_url} target="_blank" rel="noreferrer">ver origen</a></>
                  )}
                </p>
              </div>
            </li>
          ))}
        </ul>
      )}

      <p className="conclusion">
        <strong>Conclusión:</strong>{' '}
        {matches.length > 0
          ? 'se detectaron posibles coincidencias. Requieren revisión humana: escucha los tramos y revisa la licencia de cada canción de referencia.'
          : 'no se detectaron coincidencias con la base de referencia.'}
      </p>
      <p className="muted">
        Umbral {settings.threshold * 100} % · al menos {settings.min_segments} segmentos seguidos · índice{' '}
        {search.index_type?.toUpperCase() ?? '—'} ({search.metric ?? '—'})
        {timings.total_ms !== undefined ? ` · ${Math.round(timings.total_ms)} ms` : ''}
      </p>
    </section>
  )
}

export default MatchList
