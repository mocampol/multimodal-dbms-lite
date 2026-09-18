interface QueryPanelProps {
  sql: string
  onSqlChange: (sql: string) => void
  onRun: () => void
  running: boolean
}

function QueryPanel({ sql, onSqlChange, onRun, running }: QueryPanelProps) {
  return (
    <>
      <textarea
        className="query-editor"
        placeholder="SELECT * FROM ..."
        value={sql}
        onChange={(event) => onSqlChange(event.target.value)}
      />
      <button
        type="button"
        className="run-button"
        onClick={onRun}
        disabled={running || sql.trim() === ''}
      >
        {running ? 'Ejecutando...' : 'Ejecutar'}
      </button>
    </>
  )
}

export default QueryPanel
