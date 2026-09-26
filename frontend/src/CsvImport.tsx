import { useRef, useState } from 'react'
import {
  ApiError,
  confirmCsv,
  discardCsv,
  previewCsv,
  type CsvColumn,
  type CsvImportResult,
  type CsvOnError,
  type CsvPreview,
} from './api'

const SIZED_TYPES = new Set(['char', 'varchar'])
const DEFAULT_SIZE = 32

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.message : 'No se pudo conectar con la API'
}

interface CsvImportProps {
  onImported: () => void
}

function CsvImport({ onImported }: CsvImportProps) {
  const fileInput = useRef<HTMLInputElement>(null)
  const [preview, setPreview] = useState<CsvPreview | null>(null)
  const [tableName, setTableName] = useState('')
  const [columns, setColumns] = useState<CsvColumn[]>([])
  const [onError, setOnError] = useState<CsvOnError>('stop')
  const [result, setResult] = useState<CsvImportResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handleFile(file: File | undefined) {
    if (!file) return
    setBusy(true)
    setError(null)
    setResult(null)
    previewCsv(file)
      .then((data) => {
        setPreview(data)
        setTableName(data.table_name)
        setColumns(data.columns)
        setOnError('stop')
      })
      .catch((err) => setError(errorMessage(err)))
      .finally(() => {
        setBusy(false)
        if (fileInput.current) fileInput.current.value = ''
      })
  }

  function updateColumn(index: number, changes: Partial<CsvColumn>) {
    setColumns((current) =>
      current.map((column, i) => {
        if (changes.is_primary_key && i !== index) return { ...column, is_primary_key: false }
        if (i !== index) return column
        const next = { ...column, ...changes }
        if (changes.type !== undefined) {
          next.size = SIZED_TYPES.has(changes.type) ? (column.size ?? DEFAULT_SIZE) : null
        }
        return next
      }),
    )
  }

  function handleConfirm() {
    if (!preview) return
    setBusy(true)
    setError(null)
    confirmCsv(preview.upload_id, tableName, columns, onError)
      .then((data) => {
        setResult(data)
        onImported()
      })
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setBusy(false))
  }

  function handleClose() {
    if (preview && !result) discardCsv(preview.upload_id).catch(() => {})
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const hasPrimaryKey = columns.some((column) => column.is_primary_key)

  return (
    <>
      <div className="csv-import-trigger">
        <input
          ref={fileInput}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(event) => handleFile(event.target.files?.[0])}
        />
        <button
          type="button"
          className="secondary-button"
          onClick={() => fileInput.current?.click()}
          disabled={busy}
        >
          {busy && !preview ? 'Analizando...' : 'Importar CSV'}
        </button>
        {error && !preview && <p className="error-text">Error: {error}</p>}
      </div>

      {preview && (
        <div className="modal-backdrop">
          <div className="modal" role="dialog" aria-modal="true" aria-label="Importar CSV">
            <h3>Importar CSV</h3>

            {result ? (
              <>
                <p className="meta-line">
                  Tabla <strong>{result.table_name}</strong> creada · {result.inserted} fila(s) insertada(s)
                  {result.skipped.length > 0 && ` · ${result.skipped.length} omitida(s)`}
                </p>
                {result.skipped.length > 0 && (
                  <ul className="skipped-list">
                    {result.skipped.map((s) => (
                      <li key={s.row}>{s.reason}</li>
                    ))}
                  </ul>
                )}
                <div className="modal-actions">
                  <button type="button" className="run-button" onClick={handleClose}>
                    Cerrar
                  </button>
                </div>
              </>
            ) : (
              <>
                <label className="field">
                  <span>Nombre de la tabla</span>
                  <input value={tableName} onChange={(event) => setTableName(event.target.value)} />
                </label>

                <div className="results-table-wrap">
                  <table className="columns-table schema-editor">
                    <thead>
                      <tr>
                        <th>Columna</th>
                        <th>Tipo</th>
                        <th>Tamaño</th>
                        <th>PK</th>
                        <th>Nullable</th>
                        <th>Unique</th>
                      </tr>
                    </thead>
                    <tbody>
                      {columns.map((column, index) => (
                        <tr key={index}>
                          <td>
                            <input
                              value={column.name}
                              onChange={(event) => updateColumn(index, { name: event.target.value })}
                            />
                          </td>
                          <td>
                            <select
                              value={column.type}
                              onChange={(event) => updateColumn(index, { type: event.target.value })}
                            >
                              {preview.data_types.map((type) => (
                                <option key={type} value={type}>{type}</option>
                              ))}
                            </select>
                          </td>
                          <td>
                            {SIZED_TYPES.has(column.type) ? (
                              <input
                                type="number"
                                min={1}
                                className="size-input"
                                value={column.size ?? ''}
                                onChange={(event) =>
                                  updateColumn(index, {
                                    size: event.target.value === '' ? null : Number(event.target.value),
                                  })
                                }
                              />
                            ) : (
                              '—'
                            )}
                          </td>
                          <td>
                            <input
                              type="checkbox"
                              checked={column.is_primary_key}
                              onChange={(event) => updateColumn(index, { is_primary_key: event.target.checked })}
                            />
                          </td>
                          <td>
                            <input
                              type="checkbox"
                              checked={column.is_primary_key ? false : column.nullable}
                              disabled={column.is_primary_key}
                              onChange={(event) => updateColumn(index, { nullable: event.target.checked })}
                            />
                          </td>
                          <td>
                            <input
                              type="checkbox"
                              checked={column.is_primary_key || column.is_unique}
                              disabled={column.is_primary_key}
                              onChange={(event) => updateColumn(index, { is_unique: event.target.checked })}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <p className="meta-line">Vista previa ({preview.preview_rows.length} fila(s))</p>
                <div className="results-table-wrap">
                  <table className="columns-table">
                    <thead>
                      <tr>
                        {columns.map((column, index) => (
                          <th key={index}>{column.name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.preview_rows.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {row.map((value, cellIndex) => (
                            <td key={cellIndex}>{value === '' ? 'NULL' : value}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <label className="field">
                  <span>Si una fila falla</span>
                  <select value={onError} onChange={(event) => setOnError(event.target.value as CsvOnError)}>
                    <option value="stop">Cancelar toda la importación</option>
                    <option value="ignore">Omitir la fila y continuar</option>
                  </select>
                </label>

                {error && <p className="error-text">Error: {error}</p>}

                <div className="modal-actions">
                  {!hasPrimaryKey && (
                    <span className="hint-text">Marca una columna como Primary Key para continuar</span>
                  )}
                  <button type="button" className="secondary-button" onClick={handleClose} disabled={busy}>
                    Cancelar
                  </button>
                  <button
                    type="button"
                    className="run-button"
                    onClick={handleConfirm}
                    disabled={busy || tableName.trim() === '' || !hasPrimaryKey}
                  >
                    {busy ? 'Importando...' : 'Importar'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  )
}

export default CsvImport
