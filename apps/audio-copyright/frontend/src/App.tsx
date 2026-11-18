import { useEffect, useRef, useState } from 'react'
import { detect, errorMessage, getHealth, type Detection, type DetectionOptions, type Health } from './api'
import MatchList from './MatchList'
import Timeline from './Timeline'
import UploadForm from './UploadForm'
import VerdictBanner from './VerdictBanner'

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [detection, setDetection] = useState<Detection | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const audioRef = useRef<HTMLAudioElement>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch((err) => setHealthError(errorMessage(err)))
  }, [])

  // Libera la URL del audio anterior al elegir otro.
  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl)
  }, [audioUrl])

  function chooseFile(next: File | null) {
    setFile(next)
    setDetection(null)
    setError(null)
    setCurrentTime(0)
    setAudioUrl(next ? URL.createObjectURL(next) : null)
  }

  function analyze(options: DetectionOptions) {
    if (!file) return
    setBusy(true)
    setError(null)
    setDetection(null)
    detect(file, options)
      .then(setDetection)
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setBusy(false))
  }

  function seek(seconds: number) {
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = seconds
    setCurrentTime(seconds)
    void audio.play().catch(() => {})
  }

  return (
    <main className="app">
      <header className="app-header">
        <h1>Detección de copyright en audio</h1>
        <p className="muted">
          Sube una canción y busca tramos parecidos a las canciones de la base de referencia
          (1,000 clips de Free Music Archive).
        </p>
      </header>

      <p className="notice" role="note">
        <strong>Resultados orientativos.</strong> Esta herramienta reporta posibles coincidencias de audio
        según su similitud. No determina si hubo una infracción de copyright: cualquier conclusión requiere
        revisión humana.
      </p>

      {healthError && <p className="card error-text">{healthError}</p>}
      {health && !health.ready && (
        <p className="card error-text">
          La API todavía no puede analizar: {health.error ?? 'falta el codebook'}. Corre antes el servicio audio-data.
        </p>
      )}

      <UploadForm file={file} defaults={health?.settings ?? null} busy={busy}
                  onFileChange={chooseFile} onSubmit={analyze} />

      {audioUrl && (
        <audio ref={audioRef} className="player" src={audioUrl} controls
               onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} />
      )}

      {busy && (
        <p className="muted">
          Analizando: el audio se divide en segmentos y se busca cada uno en la base. Puede tardar unos segundos.
        </p>
      )}
      {error && <p className="card error-text">Error: {error}</p>}

      {detection && (
        <>
          <VerdictBanner detection={detection} />
          <Timeline detection={detection} currentTime={currentTime} onSeek={seek} />
          <MatchList detection={detection} onSeek={seek} />
        </>
      )}
    </main>
  )
}

export default App
