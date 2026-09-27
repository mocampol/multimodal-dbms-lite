import { useEffect, useMemo, useState } from 'react'
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet'
import { listTables, type PointCoordinates, type QueryResult, type TableSummary } from './api'
import 'leaflet/dist/leaflet.css'

interface SpatialMapPanelProps {
  result: QueryResult | null
  running: boolean
  searchCenter: PointCoordinates | null
  onRunQuery: (sql: string, center: PointCoordinates) => void
}

type SearchMode = 'radius' | 'knn'
type DistanceMetric = 'HAVERSINE' | 'EUCLIDEAN'

interface MapPoint extends PointCoordinates {
  id: string
  label: string
}

const RADIUS_PRESETS = [1000, 5000, 10000]
const KNN_PRESETS = [10, 50, 100]
const INITIAL_CENTER: PointCoordinates = { latitude: -12.0464, longitude: -77.0428 }

function collectPoints(result: QueryResult | null): MapPoint[] {
  if (!result) return []
  if ('statements' in result) {
    return result.statements.flatMap((statement) => collectPoints(statement))
  }
  if (!('rows' in result)) return []

  return result.rows.flatMap((row, rowIndex) =>
    row.flatMap((value, columnIndex) => {
      if (
        typeof value !== 'object' || value === null ||
        !('latitude' in value) || !('longitude' in value) ||
        typeof value.latitude !== 'number' || typeof value.longitude !== 'number'
      ) return []

      return [{
        id: `${result.columns[columnIndex] ?? 'point'}-${rowIndex}`,
        label: result.columns[columnIndex] ?? 'POINT',
        latitude: value.latitude,
        longitude: value.longitude,
      }]
    }),
  )
}

function FitMap({ points, target }: { points: MapPoint[]; target: PointCoordinates | null }) {
  const map = useMap()
  const pointKey = points.map((point) => `${point.latitude},${point.longitude}`).join('|')

  useEffect(() => {
    const coordinates = points.map((point) => [point.latitude, point.longitude] as [number, number])
    if (target) coordinates.push([target.latitude, target.longitude])
    if (coordinates.length > 1) {
      map.fitBounds(coordinates, { padding: [30, 30], maxZoom: 14 })
    } else if (coordinates.length === 1) {
      map.setView(coordinates[0], 13)
    }
  }, [map, pointKey, target?.latitude, target?.longitude])

  return null
}

function SpatialMapPanel({ result, running, searchCenter, onRunQuery }: SpatialMapPanelProps) {
  const [tables, setTables] = useState<TableSummary[]>([])
  const [tablesError, setTablesError] = useState<string | null>(null)
  const [tableName, setTableName] = useState('')
  const [columnName, setColumnName] = useState('')
  const [mode, setMode] = useState<SearchMode>('radius')
  const [metric, setMetric] = useState<DistanceMetric>('HAVERSINE')
  const [radius, setRadius] = useState('5000')
  const [neighbors, setNeighbors] = useState('10')
  const [latitude, setLatitude] = useState(String(INITIAL_CENTER.latitude))
  const [longitude, setLongitude] = useState(String(INITIAL_CENTER.longitude))

  useEffect(() => {
    listTables()
      .then((items) => {
        setTables(items)
        const firstSpatial = items.find((table) => table.columns.some((column) => column.type === 'point'))
        if (firstSpatial) {
          setTableName(firstSpatial.name)
          setColumnName(firstSpatial.columns.find((column) => column.type === 'point')?.name ?? '')
        }
      })
      .catch((error: unknown) => {
        setTablesError(error instanceof Error ? error.message : 'No se pudieron cargar las tablas')
      })
  }, [])

  const table = tables.find((item) => item.name === tableName)
  const pointColumns = table?.columns.filter((column) => column.type === 'point') ?? []
  const selectedIndex = table?.indexes.find(
    (index) => index.column_name === columnName && index.index_type === 'rtree',
  )
  const points = useMemo(() => collectPoints(result), [result])
  const center = useMemo(() => ({
    latitude: Number(latitude),
    longitude: Number(longitude),
  }), [latitude, longitude])
  const centerIsValid = Number.isFinite(center.latitude) && Number.isFinite(center.longitude)
    && Math.abs(center.latitude) <= 90 && Math.abs(center.longitude) <= 180
  const radiusIsValid = Number.isFinite(Number(radius)) && Number(radius) > 0

  function chooseTable(name: string) {
    setTableName(name)
    const nextTable = tables.find((item) => item.name === name)
    setColumnName(nextTable?.columns.find((column) => column.type === 'point')?.name ?? '')
  }

  function generateQuery() {
    if (!tableName || !columnName || !centerIsValid) return
    const point = `POINT(${center.longitude}, ${center.latitude})`
    const metricArg = `, ${metric}`
    const sql = mode === 'radius'
      ? `SELECT * FROM ${tableName} WHERE distancia(${columnName}, ${point}${metricArg}) < ${Number(radius)};`
      : `SELECT * FROM ${tableName} ORDER BY distancia(${columnName}, ${point}${metricArg}) LIMIT ${Number(neighbors)};`
    onRunQuery(sql, center)
  }

  return (
    <div className="spatial-workbench">
      <div className="spatial-controls">
        <div className="spatial-control-row">
          <fieldset className="segmented-control">
            <legend>Consulta</legend>
            <label>
              <input type="radio" name="spatial-mode" checked={mode === 'radius'} onChange={() => setMode('radius')} />
              Radio
            </label>
            <label>
              <input type="radio" name="spatial-mode" checked={mode === 'knn'} onChange={() => setMode('knn')} />
              k-NN
            </label>
          </fieldset>

          <label className="spatial-field">
            Tabla
            <select value={tableName} onChange={(event) => chooseTable(event.target.value)}>
              {tables.filter((item) => item.columns.some((column) => column.type === 'point')).map((item) => (
                <option key={item.name} value={item.name}>{item.name}</option>
              ))}
            </select>
          </label>

          <label className="spatial-field">
            Punto
            <select value={columnName} onChange={(event) => setColumnName(event.target.value)}>
              {pointColumns.map((column) => <option key={column.name} value={column.name}>{column.name}</option>)}
            </select>
          </label>

          <label className="spatial-field">
            Métrica
            <select
              value={metric}
              onChange={(event) => {
                const nextMetric = event.target.value as DistanceMetric
                setMetric(nextMetric)
                setRadius(nextMetric === 'HAVERSINE' ? '5000' : '0.05')
              }}
            >
              <option value="HAVERSINE">Haversine</option>
              <option value="EUCLIDEAN">Euclidiana</option>
            </select>
          </label>
        </div>

        <div className="spatial-control-row">
          <label className="spatial-field">
            Latitud
            <input type="number" min="-90" max="90" step="0.0001" value={latitude} onChange={(event) => setLatitude(event.target.value)} />
          </label>
          <label className="spatial-field">
            Longitud
            <input type="number" min="-180" max="180" step="0.0001" value={longitude} onChange={(event) => setLongitude(event.target.value)} />
          </label>

          {mode === 'radius' ? (
            <div className="spatial-presets" role="group" aria-label={metric === 'HAVERSINE' ? 'Radio en kilómetros' : 'Radio en grados'}>
              <span>{metric === 'HAVERSINE' ? 'Radio (km)' : 'Radio (grados)'}</span>
              {(metric === 'HAVERSINE' ? RADIUS_PRESETS : [0.01, 0.05, 0.1]).map((value) => {
                const queryRadius = metric === 'HAVERSINE' ? value : value
                const display = metric === 'HAVERSINE' ? `${value / 1000} km` : String(value)
                return (
                  <button
                    type="button"
                    key={value}
                    className={Number(radius) === queryRadius ? 'spatial-preset selected' : 'spatial-preset'}
                    onClick={() => setRadius(String(queryRadius))}
                  >
                    {display}
                  </button>
                )
              })}
              <input
                aria-label={metric === 'HAVERSINE' ? 'Radio personalizado en metros' : 'Radio personalizado en grados'}
                type="number"
                min={metric === 'HAVERSINE' ? '0.001' : '0.001'}
                step={metric === 'HAVERSINE' ? '0.1' : '0.001'}
                value={metric === 'HAVERSINE' ? Number(radius) / 1000 : radius}
                onChange={(event) => {
                  const inputRadius = Number(event.target.value)
                  setRadius(String(metric === 'HAVERSINE' ? inputRadius * 1000 : inputRadius))
                }}
              />
            </div>
          ) : (
            <div className="spatial-presets" role="group" aria-label="Cantidad de vecinos">
              <span>Vecinos</span>
              {KNN_PRESETS.map((value) => (
                <button
                  type="button"
                  key={value}
                  className={Number(neighbors) === value ? 'spatial-preset selected' : 'spatial-preset'}
                  onClick={() => setNeighbors(String(value))}
                >
                  {value}
                </button>
              ))}
            </div>
          )}

          <button
            type="button"
            className="run-button spatial-run"
            disabled={running || !tableName || !columnName || !centerIsValid || (mode === 'radius' && !radiusIsValid)}
            onClick={generateQuery}
          >
            {running ? 'Buscando...' : 'Buscar'}
          </button>
        </div>

        {tablesError && <p className="error-text">{tablesError}</p>}
        {!tablesError && pointColumns.length === 0 && (
          <p className="meta-line">No hay tablas con columnas POINT disponibles.</p>
        )}
        {tableName && columnName && (
          <p className="spatial-index-status">
            {selectedIndex ? `R-Tree #${selectedIndex.index_id}` : 'Scan secuencial'} · {points.length} puntos visibles
          </p>
        )}
      </div>

      <div className="spatial-map-frame">
        <MapContainer center={[INITIAL_CENTER.latitude, INITIAL_CENTER.longitude]} zoom={11} scrollWheelZoom>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitMap points={points} target={searchCenter} />
          {searchCenter && (
            <CircleMarker
              center={[searchCenter.latitude, searchCenter.longitude]}
              radius={8}
              pathOptions={{ color: '#bd3f32', fillColor: '#e26148', fillOpacity: 0.9, weight: 2 }}
            >
              <Popup>Punto de búsqueda</Popup>
            </CircleMarker>
          )}
          {points.map((point) => (
            <CircleMarker
              key={point.id}
              center={[point.latitude, point.longitude]}
              radius={7}
              pathOptions={{ color: '#176b66', fillColor: '#35a89a', fillOpacity: 0.88, weight: 2 }}
            >
              <Popup>
                <strong>{point.label}</strong><br />
                Lat {point.latitude.toFixed(5)}, lon {point.longitude.toFixed(5)}
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  )
}

export default SpatialMapPanel