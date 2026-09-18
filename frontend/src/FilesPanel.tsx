import { useEffect, useState } from 'react'
import { ApiError, getTable, listTables, type TableDetail, type TableSummary } from './api'

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.message : 'No se pudo conectar con la API'
}

interface FilesPanelProps {
  refreshKey: number
}

function FilesPanel({ refreshKey }: FilesPanelProps) {
  const [tables, setTables] = useState<TableSummary[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)
  const [detail, setDetail] = useState<TableDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listTables()
      .then(setTables)
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setLoading(false))
  }, [refreshKey])

  useEffect(() => {
    if (!expanded) {
      setDetail(null)
      return
    }
    setDetailError(null)
    getTable(expanded)
      .then(setDetail)
      .catch((err) => setDetailError(errorMessage(err)))
  }, [expanded, refreshKey])

  function toggleTable(name: string) {
    setExpanded((current) => (current === name ? null : name))
  }

  if (loading) return <p className="panel-placeholder">Cargando tablas...</p>
  if (error) return <p className="error-text">Error: {error}</p>
  if (tables.length === 0) return <p className="panel-placeholder">No hay tablas creadas.</p>

  return (
    <ul className="table-list">
      {tables.map((table) => {
        const isExpanded = expanded === table.name

        return (
          <li key={table.name}>
            <button
              type="button"
              className={isExpanded ? 'table-item selected' : 'table-item'}
              onClick={() => toggleTable(table.name)}
            >
              {table.name}
            </button>

            {isExpanded && (
              <div className="table-detail">
                {!detail && !detailError && <p className="panel-placeholder">Cargando...</p>}
                {detailError && <p className="error-text">Error: {detailError}</p>}
                {detail && (
                  <table className="columns-table">
                    <thead>
                      <tr>
                        <th>Columna</th>
                        <th>Tipo</th>
                        <th>Índice</th>
                      </tr>
                    </thead>
                    <tbody>
                      {detail.columns.map((column) => {
                        const index = detail.indexes.find((i) => i.column_name === column.name)
                        return (
                          <tr key={column.name}>
                            <td>{column.name}{column.is_primary_key ? ' (PK)' : ''}</td>
                            <td>{column.type}</td>
                            <td>{index ? index.index_type : '—'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </li>
        )
      })}
    </ul>
  )
}

export default FilesPanel
