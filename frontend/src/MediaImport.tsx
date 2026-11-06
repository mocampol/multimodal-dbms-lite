import { useState, type FormEvent } from 'react'
import { ApiError, importMedia, type MediaImportResult, type MediaIndex, type MediaKind } from './api'

const KINDS: { value: MediaKind; label: string; extensions: string; indexes: MediaIndex[] }[] = [
  { value: 'image', label: 'Imágenes', extensions: '.jpg, .jpeg, .png', indexes: ['HNSW', 'IVF'] },
  { value: 'audio', label: 'Audios', extensions: '.wav, .mp3', indexes: ['IVF', 'HNSW'] },
  { value: 'text', label: 'Documentos', extensions: '.txt', indexes: ['SPIMI'] },
]

function errorMessage(err: unknown): string {
  return err instanceof ApiError ? err.message : 'No se pudo conectar con la API'
}

interface MediaImportProps {
  onImported: () => void
}

function MediaImport({ onImported }: MediaImportProps) {
  const [open, setOpen] = useState(false)
  const [directory, setDirectory] = useState('')
  const [kind, setKind] = useState<MediaKind>('image')
  const [tableName, setTableName] = useState('')
  const [index, setIndex] = useState<MediaIndex | ''>('HNSW')
  const [result, setResult] = useState<MediaImportResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const kindInfo = KINDS.find((item) => item.value === kind)!

  function chooseKind(next: MediaKind) {
    setKind(next)
    setIndex(KINDS.find((item) => item.value === next)!.indexes[0])
  }

  function handleOpen() {
    setOpen(true)
    setResult(null)
    setError(null)
  }

  function handleClose() {
    if (result) {
      setDirectory('')
      setTableName('')
    }
    setOpen(false)
    setResult(null)
    setError(null)
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError(null)
    importMedia(directory.trim(), tableName.trim(), kind, index || null)
      .then((data) => {
        setResult(data)
        onImported()
      })
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setBusy(false))
  }

  return (
    <>
      <button type="button" className="secondary-button" onClick={handleOpen}>
        Importar carpeta
      </button>

      {open && (
        <div className="modal-backdrop">
          <div className="modal" role="dialog" aria-modal="true" aria-label="Importar carpeta multimedia">
            <h3>Importar carpeta multimedia</h3>

            {result ? (
              <>
                <p className="meta-line">
                  Tabla <strong>{result.table_name}</strong> creada · {result.inserted} archivo(s) cargado(s)
                  {result.skipped.length > 0 && ` · ${result.skipped.length} omitido(s)`}
                  {' · '}
                  {result.index
                    ? `índice ${result.index.index_type.toUpperCase()} sobre ${result.index.column_name}`
                    : 'sin índice'}
                </p>
                {result.skipped.length > 0 && (
                  <ul className="skipped-list">
                    {result.skipped.map((s) => (
                      <li key={s.file}>
                        <strong>{s.file}</strong>: {s.reason}
                      </li>
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
              <form className="media-import-form" onSubmit={handleSubmit}>
                <label className="field">
                  <span>Carpeta en el servidor</span>
                  <input
                    value={directory}
                    onChange={(event) => setDirectory(event.target.value)}
                    placeholder="C:\datos\imagenes"
                    autoFocus
                  />
                </label>

                <label className="field">
                  <span>Tipo</span>
                  <select value={kind} onChange={(event) => chooseKind(event.target.value as MediaKind)}>
                    {KINDS.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label} ({item.extensions})
                      </option>
                    ))}
                  </select>
                </label>

                <label className="field">
                  <span>Nombre de la tabla</span>
                  <input value={tableName} onChange={(event) => setTableName(event.target.value)} />
                </label>

                <label className="field">
                  <span>Índice</span>
                  <select value={index} onChange={(event) => setIndex(event.target.value as MediaIndex | '')}>
                    {kindInfo.indexes.map((item) => (
                      <option key={item} value={item}>{item}</option>
                    ))}
                    <option value="">Sin índice</option>
                  </select>
                </label>

                <p className="meta-line">
                  La carpeta debe estar en la máquina donde corre la API. Con muchos archivos, construir el
                  índice puede tardar.
                </p>

                {error && <p className="error-text">Error: {error}</p>}

                <div className="modal-actions">
                  <button type="button" className="secondary-button" onClick={handleClose} disabled={busy}>
                    Cancelar
                  </button>
                  <button
                    type="submit"
                    className="run-button"
                    disabled={busy || directory.trim() === '' || tableName.trim() === ''}
                  >
                    {busy ? 'Importando...' : 'Importar'}
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </>
  )
}

export default MediaImport
