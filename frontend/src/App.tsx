import { useState } from 'react'
import { ApiError, runQuery, type QueryResult } from './api'
import FilesPanel from './FilesPanel'
import QueryPanel from './QueryPanel'
import ResultsPanel from './ResultsPanel'

function App() {
  const [sql, setSql] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function handleRun() {
    setRunning(true)
    setError(null)
    runQuery(sql)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : 'No se pudo conectar con la API'))
      .finally(() => setRunning(false))
  }

  return (
    <>
      <header className="app-header">
        <h1>Multimodal DBMS Lite</h1>
      </header>

      <div className="app-layout">
        <section className="panel" aria-label="Files">
          <h2>Files</h2>
          <FilesPanel />
        </section>

        <div className="main-column">
          <section className="panel" aria-label="Query">
            <h2>Query</h2>
            <QueryPanel sql={sql} onSqlChange={setSql} onRun={handleRun} running={running} />
          </section>

          <section className="panel" aria-label="Results">
            <h2>Results</h2>
            <ResultsPanel result={result} error={error} running={running} />
          </section>

          <section className="panel" aria-label="Execution Plan">
            <h2>Execution Plan</h2>
            <p className="panel-placeholder">Execution steps will be listed here.</p>
          </section>
        </div>
      </div>
    </>
  )
}

export default App
