/** 75.5 -> "1:15.5" y 75 -> "1:15": los segmentos avanzan de a 2.5 s, así que a veces hay décimas. */
export function formatTime(seconds: number): string {
  const tenths = Math.round(Math.max(0, seconds) * 10)
  const minutes = Math.floor(tenths / 600)
  const rest = tenths - minutes * 600
  const secs = String(Math.floor(rest / 10)).padStart(2, '0')
  const decimal = rest % 10
  return decimal ? `${minutes}:${secs}.${decimal}` : `${minutes}:${secs}`
}

export function formatInterval(start: number, end: number): string {
  return `${formatTime(start)}–${formatTime(end)}`
}

/** Un color fijo por coincidencia, en orden: el #1 siempre tiene el mismo color. */
const MATCH_COLORS = ['#7c3aed', '#0f9f6e', '#c2410c', '#2563eb', '#be185d', '#0e7490', '#4d7c0f', '#a16207']

export function matchColor(matchId: number): string {
  return MATCH_COLORS[(matchId - 1) % MATCH_COLORS.length]
}
