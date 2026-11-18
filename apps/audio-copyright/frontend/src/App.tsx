import { useEffect, useState } from 'react'
import { errorMessage, getHealth } from './api'

function App() {
  const [status, setStatus] = useState('Conectando con la API de detección...')

  useEffect(() => {
    getHealth()
      .then((health) => setStatus(health.ready ? 'API lista' : `La API no tiene el codebook: ${health.error ?? ''}`))
      .catch((error) => setStatus(errorMessage(error)))
  }, [])

  return (
    <main className="app">
      <header className="app-header">
        <h1>Detección de copyright en audio</h1>
        <p className="muted">{status}</p>
      </header>
    </main>
  )
}

export default App
