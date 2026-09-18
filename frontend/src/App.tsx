import FilesPanel from './FilesPanel'

function App() {
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
            <textarea
              className="query-editor"
              placeholder="SELECT * FROM ..."
              disabled
            />
            <button type="button" className="run-button" disabled>
              Ejecutar
            </button>
          </section>

          <section className="panel" aria-label="Results">
            <h2>Results</h2>
            <p className="panel-placeholder">Query results will be shown here.</p>
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
