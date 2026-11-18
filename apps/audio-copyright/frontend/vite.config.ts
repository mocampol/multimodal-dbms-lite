import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 5173 es del frontend principal y 5174 del reconocimiento facial.
export default defineConfig({
  plugins: [react()],
  server: { port: 5175 },
})
