import { useState } from 'react'
import { ApiError, runQuery, type QueryResult } from './api'
import CsvImport from './CsvImport'
import ExecutionPlanPanel from './ExecutionPlanPanel'
import FilesPanel from './FilesPanel'
import QueryPanel from './QueryPanel'
import ResultsPanel from './ResultsPanel'
import SpatialMapPanel from './SpatialMapPanel'

function App() {
  const [sql, setSql] = useState('')
  const [result, setResult] = useState<QueryResult | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [searchCenter, setSearchCenter] = useState<{ latitude: number; longitude: number } | null>(null)

  function handleRun() {
    setSearchCenter(null)
    runSql(sql)
  }

  function handleSpatialRun(query: string, center: { latitude: number; longitude: number }) {
    setSearchCenter(center)
    setSql(query)
    runSql(query)
  }

  function runSql(query: string) {
    setRunning(true)
    setError(null)
    setResult(null)
    runQuery(query)
      .then((data) => {
        setResult(data)
        setRefreshKey((key) => key + 1)
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : 'No se pudo conectar con la API'))
      .finally(() => setRunning(false))
  }

  return (
    <>
      <header className="app-header">
        <h1>Multimodal DBMS Lite</h1>
        <CsvImport onImported={() => setRefreshKey((key) => key + 1)} />
      </header>

      <div className="app-layout">
        <section className="panel" aria-label="Files">
          <h2>Files</h2>
          <FilesPanel refreshKey={refreshKey} />
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

          <section className="panel" aria-label="Spatial search and map">
            <h2>Spatial Search & Map</h2>
            <SpatialMapPanel
              result={result}
              running={running}
              searchCenter={searchCenter}
              refreshKey={refreshKey}
              onRunQuery={handleSpatialRun}
            />
          </section>

          <section className="panel" aria-label="Execution Plan">
            <h2>Execution Plan</h2>
            <ExecutionPlanPanel result={result} error={error} running={running} />
          </section>
        </div>
      </div>
    </>
  )
}

export default App
