import { useState, type ReactNode } from 'react'

const STORAGE_KEY = 'collapsed-panels'

function readCollapsed(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
  } catch {
    return {}
  }
}

interface PanelProps {
  id: string
  title: string
  defaultCollapsed?: boolean
  children: ReactNode
}

// El contenido se oculta pero no se desmonta: así el editor, los filtros
// espaciales y el mapa conservan su estado al minimizar y expandir.
function Panel({ id, title, defaultCollapsed = false, children }: PanelProps) {
  const [collapsed, setCollapsed] = useState(() => readCollapsed()[id] ?? defaultCollapsed)
  const contentId = `panel-${id}`

  function toggle() {
    const next = !collapsed
    setCollapsed(next)
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...readCollapsed(), [id]: next }))
  }

  return (
    <section className={collapsed ? 'panel collapsed' : 'panel'} aria-label={title}>
      <div className="panel-header">
        <h2>{title}</h2>
        <button
          type="button"
          className="panel-toggle"
          onClick={toggle}
          aria-expanded={!collapsed}
          aria-controls={contentId}
        >
          {collapsed ? 'Expandir' : 'Minimizar'}
        </button>
      </div>
      <div id={contentId} hidden={collapsed}>
        {children}
      </div>
    </section>
  )
}

export default Panel
