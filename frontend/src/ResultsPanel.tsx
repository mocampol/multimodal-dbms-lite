import {
  fileUrl,
  isBatchQueryResult,
  isExplainQueryResult,
  type DistanceValue,
  type PointCoordinates,
  type QueryResult,
  type QueryValue,
  type SingleQueryResult,
} from './api'

const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
const AUDIO_EXTENSIONS = ['.wav', '.mp3']
// columnas de SCORE() y SIMILARITY_SCORE(), con o sin alias como "relevancia"
const SCORE_COLUMN = /score|relevan/i

interface ResultsPanelProps {
  result: QueryResult | null
  error: string | null
  running: boolean
}

function ResultsPanel({ result, error, running }: ResultsPanelProps) {
  if (running) return <p className="panel-placeholder">Ejecutando consulta...</p>
  if (error) return <p className="error-text">Error: {error}</p>
  if (!result) return <p className="panel-placeholder">Query results will be shown here.</p>

  if (isBatchQueryResult(result)) {
    return (
      <>
        <p className="meta-line">
          {result.statements.length} sentencia(s) · {result.execution_ms.toFixed(2)} ms
        </p>
        {result.statements.map((statement, index) => (
          <div key={index}>
            <p className="meta-line">#{index + 1} · {statement.type}</p>
            {!isBatchQueryResult(statement) && <ResultContent result={statement} />}
          </div>
        ))}
      </>
    )
  }

  return <ResultContent result={result} />
}

function ResultContent({ result }: { result: SingleQueryResult }) {

  if (isExplainQueryResult(result)) {
    return (
      <p className="meta-line">
        {result.type === 'explain_analyze'
          ? `EXPLAIN ANALYZE · ${result.row_count} fila(s) · ${result.elapsed_ms?.toFixed(3)} ms (ejecución del plan) · ${result.execution_ms.toFixed(2)} ms (total) · ver panel de Execution Plan`
          : `EXPLAIN · plan generado sin ejecutar · ver panel de Execution Plan`}
      </p>
    )
  }

  if (!('row_count' in result)) {
    const affected = 'rows_affected' in result ? `${result.rows_affected} fila(s) afectada(s)` : 'OK'
    return (
      <p className="meta-line">
        {result.type} · {affected} · {result.execution_ms.toFixed(2)} ms
      </p>
    )
  }

  return (
    <>
      <p className="meta-line">
        {result.row_count} fila(s) · {result.execution_ms.toFixed(2)} ms
      </p>
      {result.row_count === 0 ? (
        <p className="panel-placeholder">Sin resultados.</p>
      ) : (
        <div className="results-table-wrap">
          <table className="columns-table">
            <thead>
              <tr>
                {result.columns.map((column, index) => {
                  const isScore = SCORE_COLUMN.test(column)
                  const direction = isScore ? sortDirection(result.rows, index) : ''
                  return (
                    <th key={column} className={isScore ? 'score-column' : undefined}>
                      {column}
                      {direction && (
                        <span title={direction === '↓' ? 'Orden descendente' : 'Orden ascendente'}> {direction}</span>
                      )}
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((value, cellIndex) => {
                    const isScore = SCORE_COLUMN.test(result.columns[cellIndex] ?? '')
                    return (
                      <td key={cellIndex} className={isScore ? 'score-cell' : undefined}>
                        <CellValue value={value} isScore={isScore} />
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

function CellValue({ value, isScore }: { value: QueryValue; isScore: boolean }) {
  if (isScore && typeof value === 'number') return <>{value.toFixed(4)}</>

  const media = mediaFile(value)
  if (media?.kind === 'image') {
    return (
      <a href={fileUrl(media.path)} target="_blank" rel="noreferrer" title={media.path}>
        <img className="result-thumbnail" src={fileUrl(media.path)} alt={media.name} loading="lazy" />
      </a>
    )
  }
  if (media?.kind === 'audio') {
    return <audio className="result-audio" controls preload="none" src={fileUrl(media.path)} title={media.path} />
  }
  return <>{formatCellValue(value)}</>
}

// Solo las rutas absolutas son archivos: la columna "nombre" (foto1.png)
// también termina en .png, pero no es una ruta que la API pueda servir.
function mediaFile(value: QueryValue): { kind: 'image' | 'audio'; path: string; name: string } | null {
  if (typeof value !== 'string' || !/^([a-zA-Z]:[\\/]|\/)/.test(value)) return null
  const lower = value.toLowerCase()
  const name = value.split(/[\\/]/).pop() ?? value
  if (IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext))) return { kind: 'image', path: value, name }
  if (AUDIO_EXTENSIONS.some((ext) => lower.endsWith(ext))) return { kind: 'audio', path: value, name }
  return null
}

function sortDirection(rows: QueryValue[][], index: number): '↓' | '↑' | '' {
  const values = rows.map((row) => row[index]).filter((value): value is number => typeof value === 'number')
  if (values.length < 2) return ''
  if (values.every((value, i) => i === 0 || values[i - 1] >= value)) return '↓'
  if (values.every((value, i) => i === 0 || values[i - 1] <= value)) return '↑'
  return ''
}

function formatCellValue(value: QueryValue): string {
  if (value === null) return 'NULL'
  if (typeof value === 'object' && value !== null) {
    if ('points' in value && Array.isArray((value as any).points)) {
      return `POLYGON(${(value as any).points.length} puntos)`
    }
    if ('west' in value && 'south' in value) {
      const r = value as any
      return `RECTANGLE(${r.west}, ${r.south}, ${r.east}, ${r.north})`
    }
    if (isPointCoordinates(value)) {
      return `lat: ${value.latitude}, lon: ${value.longitude}`
    }
    if (isDistanceValue(value)) return `${value.value} ${value.unit}`
    return JSON.stringify(value)
  }
  return String(value)
}

function isPointCoordinates(value: QueryValue): value is PointCoordinates {
  return typeof value === 'object' && value !== null && 'longitude' in value && 'latitude' in value && !('points' in value)
}

function isDistanceValue(value: QueryValue): value is DistanceValue {
  return typeof value === 'object' && value !== null && 'value' in value && 'unit' in value
}

export default ResultsPanel