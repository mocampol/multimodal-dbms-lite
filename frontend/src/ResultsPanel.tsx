import { isBatchQueryResult, type QueryResult, type SingleQueryResult } from './api'

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
                {result.columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((value, cellIndex) => (
                    <td key={cellIndex}>{value === null ? 'NULL' : String(value)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

export default ResultsPanel
