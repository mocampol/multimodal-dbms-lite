import { useState, type FormEvent } from 'react'
import type { DetectionOptions, DetectionSettings, IndexType } from './api'

const ACCEPT = '.mp3,.wav,.flac,.ogg,.m4a,audio/*'

interface UploadFormProps {
  file: File | null
  defaults: DetectionSettings | null
  busy: boolean
  onFileChange: (file: File | null) => void
  onSubmit: (options: DetectionOptions) => void
}

function UploadForm({ file, defaults, busy, onFileChange, onSubmit }: UploadFormProps) {
  const [index, setIndex] = useState<IndexType>('hnsw')
  const [thresholdPct, setThresholdPct] = useState('')  // vacío: el umbral calibrado de la API
  const [minSegments, setMinSegments] = useState('')

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    onSubmit({
      index,
      threshold: thresholdPct === '' ? undefined : Number(thresholdPct) / 100,
      minSegments: minSegments === '' ? undefined : Number(minSegments),
    })
  }

  return (
    <form className="card upload-form" onSubmit={handleSubmit}>
      <div className="upload-row">
        <label className="file-picker">
          <input type="file" accept={ACCEPT} onChange={(event) => onFileChange(event.target.files?.[0] ?? null)} />
          <span className="secondary-button">Elegir audio</span>
          <span className="file-name">{file ? file.name : 'MP3, WAV, FLAC, OGG o M4A, de 5 s a 10 min'}</span>
        </label>
        <button type="submit" className="primary-button" disabled={!file || busy}>
          {busy ? 'Analizando...' : 'Analizar'}
        </button>
      </div>

      <details className="advanced">
        <summary>Opciones avanzadas</summary>
        <div className="advanced-fields">
          <label className="field">
            Índice
            <select value={index} onChange={(event) => setIndex(event.target.value as IndexType)}>
              <option value="hnsw">HNSW</option>
              <option value="ivf">IVF</option>
            </select>
          </label>
          <label className="field">
            Umbral de similitud (%)
            <input type="number" min={50} max={100} step={0.5} value={thresholdPct}
                   placeholder={defaults ? String(defaults.threshold * 100) : ''}
                   onChange={(event) => setThresholdPct(event.target.value)} />
          </label>
          <label className="field">
            Segmentos seguidos mínimos
            <input type="number" min={1} step={1} value={minSegments}
                   placeholder={defaults ? String(defaults.min_segments) : ''}
                   onChange={(event) => setMinSegments(event.target.value)} />
          </label>
        </div>
        <p className="muted">
          Si los dejas vacíos, se usan los valores calibrados de la API. Un umbral más bajo encuentra
          más coincidencias, pero también más coincidencias falsas.
        </p>
      </details>
    </form>
  )
}

export default UploadForm
