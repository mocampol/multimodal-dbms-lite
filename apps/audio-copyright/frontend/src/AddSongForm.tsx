import { useEffect, useState, type FormEvent } from 'react'
import { addSong, errorMessage, listSongs, type AddedSong, type UploadedSong } from './api'
import { formatTime } from './format'

const ACCEPT = '.mp3,.wav,.flac,.ogg,.m4a,audio/*'
const LICENSE_HELP: Record<string, string> = {
  Propia: 'Eres el autor del audio',
}

/** Agrega una canción a la base de referencia y lista las que ya se agregaron. */
function AddSongForm() {
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [artist, setArtist] = useState('')
  const [license, setLicense] = useState('Propia')
  const [album, setAlbum] = useState('')
  const [genre, setGenre] = useState('')
  const [sourceUrl, setSourceUrl] = useState('')
  const [licenses, setLicenses] = useState<string[]>(['Propia'])
  const [songs, setSongs] = useState<UploadedSong[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [added, setAdded] = useState<AddedSong | null>(null)

  function refresh() {
    listSongs()
      .then((list) => {
        setSongs(list.songs)
        setLicenses(list.licenses)
      })
      .catch((err) => setError(errorMessage(err)))
  }

  useEffect(refresh, [])

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (!file) return
    setBusy(true)
    setError(null)
    setAdded(null)
    addSong(file, { title, artist, license, album, genre, sourceUrl })
      .then((song) => {
        setAdded(song)
        setFile(null)
        setTitle('')
        setAlbum('')
        setSourceUrl('')
        refresh()
      })
      .catch((err) => setError(errorMessage(err)))
      .finally(() => setBusy(false))
  }

  return (
    <section className="card add-song">
      <h2>Agregar una canción a la base</h2>
      <p className="muted">
        La canción se procesa igual que las de referencia: se divide en segmentos, se describe con las palabras
        de audio ya aprendidas (el codebook no cambia) y sus segmentos se agregan a los índices HNSW e IVF.
        Desde ese momento, las detecciones también la buscan.
      </p>

      <form className="add-song-form" onSubmit={handleSubmit}>
        <label className="file-picker">
          <input type="file" accept={ACCEPT} onChange={(event) => setFile(event.target.files?.[0] ?? null)} />
          <span className="secondary-button">Elegir audio</span>
          <span className="file-name">{file ? file.name : 'MP3, WAV, FLAC, OGG o M4A, de 5 s a 10 min'}</span>
        </label>
        <div className="advanced-fields">
          <label className="field">
            Título *
            <input value={title} maxLength={256} required onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label className="field">
            Artista *
            <input value={artist} maxLength={128} required onChange={(event) => setArtist(event.target.value)} />
          </label>
          <label className="field">
            Licencia
            <select value={license} onChange={(event) => setLicense(event.target.value)}>
              {licenses.map((name) => (
                <option key={name} value={name}>{LICENSE_HELP[name] ? `${name} (${LICENSE_HELP[name]})` : name}</option>
              ))}
            </select>
          </label>
          <label className="field">
            Álbum
            <input value={album} maxLength={256} onChange={(event) => setAlbum(event.target.value)} />
          </label>
          <label className="field">
            Género
            <input value={genre} maxLength={32} onChange={(event) => setGenre(event.target.value)} />
          </label>
          <label className="field">
            URL de origen
            <input type="url" value={sourceUrl} maxLength={512} onChange={(event) => setSourceUrl(event.target.value)} />
          </label>
        </div>
        <p className="muted">
          Agrega solo audio que puedas usar: propio o con una licencia Creative Commons que permita obras derivadas.
        </p>
        <button type="submit" className="primary-button" disabled={!file || !title.trim() || !artist.trim() || busy}>
          {busy ? 'Agregando...' : 'Agregar a la base'}
        </button>
      </form>

      {error && <p className="error-text">Error: {error}</p>}
      {added && (
        <p className="success-text">
          Se agregó «{added.title}» de {added.artist}: {added.segments} segmentos de {formatTime(added.duration_s)},
          ya en los índices (id {added.song_id}).
        </p>
      )}

      {songs.length > 0 && (
        <table className="song-table">
          <caption>Canciones agregadas ({songs.length})</caption>
          <thead>
            <tr><th>Id</th><th>Título</th><th>Artista</th><th>Duración</th><th>Licencia</th></tr>
          </thead>
          <tbody>
            {songs.map((song) => (
              <tr key={song.song_id}>
                <td>{song.song_id}</td>
                <td>{song.title}</td>
                <td>{song.artist}</td>
                <td>{formatTime(song.duration_s)}</td>
                <td>{song.license_family}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

export default AddSongForm
