import { useState } from 'react'
import { ApiError, runQuery, type QueryResult } from './api'
import CsvImport from './CsvImport'
import ExecutionPlanPanel from './ExecutionPlanPanel'
import FilesPanel from './FilesPanel'
import Panel from './Panel'
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
        <Panel id="files" title="Files">
          <FilesPanel refreshKey={refreshKey} />
        </Panel>

        <div className="main-column">
          <Panel id="query" title="Query">
            <QueryPanel sql={sql} onSqlChange={setSql} onRun={handleRun} running={running} />
          </Panel>

          <Panel id="results" title="Results">
            <ResultsPanel result={result} error={error} running={running} />
          </Panel>

          <Panel id="spatial" title="Spatial Search & Map" defaultCollapsed>
            <SpatialMapPanel
              result={result}
              running={running}
              searchCenter={searchCenter}
              refreshKey={refreshKey}
              onRunQuery={handleSpatialRun}
            />
          </Panel>

          <Panel id="plan" title="Execution Plan">
            <ExecutionPlanPanel result={result} error={error} running={running} />
          </Panel>
        </div>
      </div>
    </>
  )
}

export default App
