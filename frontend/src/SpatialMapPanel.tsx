import { useEffect, useMemo, useState, Fragment } from 'react'
import { Circle, CircleMarker, MapContainer, Popup, Rectangle, TileLayer, Polygon, Polyline, useMap } from 'react-leaflet'
import { listTables, type PointCoordinates, type QueryResult, type TableSummary, type PlanNode } from './api'
import 'leaflet/dist/leaflet.css'

interface SpatialMapPanelProps {
  result: QueryResult | null
  running: boolean
  searchCenter: PointCoordinates | null
  refreshKey: number
  onRunQuery: (sql: string, center: PointCoordinates) => void
}

type SearchMode = 'radius' | 'knn'
type DistanceMetric = 'HAVERSINE' | 'EUCLIDEAN'

interface MapFeature {
  id: string
  label: string
  rowData: { column: string; value: string }[]
  type: 'point' | 'rectangle' | 'polygon'
  latitude?: number
  longitude?: number
  bounds?: [[number, number], [number, number]] // para rectángulos
  positions?: [number, number][] // para polígonos
}

const RADIUS_PRESETS = [1000, 5000, 10000]
const KNN_PRESETS = [10, 50, 100]
const INITIAL_CENTER: PointCoordinates = { latitude: -12.0464, longitude: -77.0428 }

function collectFeatures(result: QueryResult | null): MapFeature[] {
  if (!result) return []
  if ('statements' in result) {
    return result.statements.flatMap((statement) => collectFeatures(statement))
  }
  if (!('rows' in result)) return []

  return result.rows.flatMap((row, rowIndex) =>
    row.flatMap((value, columnIndex) => {
      if (typeof value !== 'object' || value === null) return []
      const geomObj = value as any

      let isPoint = 'latitude' in geomObj && 'longitude' in geomObj
      let isRect = 'west' in geomObj && 'south' in geomObj && 'east' in geomObj && 'north' in geomObj
      let isPoly = 'points' in geomObj && Array.isArray(geomObj.points)

      if (!isPoint && !isRect && !isPoly) return []

      const rowData = row.map((val, i) => {
        let displayValue = String(val)
        if (val === null) displayValue = 'NULL'
        else if (typeof val === 'object') {
          const anyVal = val as any
          if ('longitude' in anyVal && 'latitude' in anyVal) {
            displayValue = `POINT(${anyVal.longitude}, ${anyVal.latitude})`
          } else if ('west' in anyVal && 'south' in anyVal) {
            displayValue = `RECTANGLE(${anyVal.west}, ${anyVal.south}, ${anyVal.east}, ${anyVal.north})`
          } else if ('points' in anyVal && Array.isArray(anyVal.points)) {
            displayValue = `POLYGON(${anyVal.points.length} puntos)`
          } else if ('value' in anyVal && 'unit' in anyVal) {
            displayValue = `${anyVal.value} ${anyVal.unit}`
          } else {
            displayValue = JSON.stringify(val)
          }
        }
        return {
          column: result.columns[i] ?? `col${i}`,
          value: displayValue
        }
      }).filter(item => item.column !== (result.columns[columnIndex] ?? 'GEOMETRY'))

      const feature: MapFeature = {
        id: `${result.columns[columnIndex] ?? 'geom'}-${rowIndex}`,
        label: result.columns[columnIndex] ?? 'GEOMETRY',
        rowData,
        type: isPoint ? 'point' : isRect ? 'rectangle' : 'polygon',
      }

      if (isPoint) {
        feature.latitude = Number(geomObj.latitude)
        feature.longitude = Number(geomObj.longitude)
      } else if (isRect) {
        const south = Number(geomObj.south)
        const north = Number(geomObj.north)
        const west = Number(geomObj.west)
        const east = Number(geomObj.east)
        feature.bounds = [[south, west], [north, east]]
        feature.latitude = (south + north) / 2.0
        feature.longitude = (west + east) / 2.0
      } else if (isPoly) {
        feature.positions = geomObj.points.map((p: any) => [Number(p.latitude), Number(p.longitude)])
        const pts = geomObj.points
        const unique = pts.length > 1 && pts[0].latitude === pts[pts.length - 1].latitude && pts[0].longitude === pts[pts.length - 1].longitude
          ? pts.slice(0, -1)
          : pts
        if (unique.length > 0) {
          feature.latitude = unique.reduce((sum: number, p: any) => sum + Number(p.latitude), 0) / unique.length
          feature.longitude = unique.reduce((sum: number, p: any) => sum + Number(p.longitude), 0) / unique.length
        }
      }

      return [feature]
    }),
  )
}

function extractBoxes(plan: PlanNode | null | undefined): { candidate: [number, number, number, number][], index: [number, number, number, number][] } {
  if (!plan) return { candidate: [], index: [] }
  const boxes = {
    candidate: plan.candidate_boxes ? [...plan.candidate_boxes] : [],
    index: plan.index_mbrs ? [...plan.index_mbrs] : []
  }
  if (plan.children) {
    for (const child of plan.children) {
      const childBoxes = extractBoxes(child)
      boxes.candidate.push(...childBoxes.candidate)
      boxes.index.push(...childBoxes.index)
    }
  }
  return boxes
}

function FitMap({ features, target }: { features: MapFeature[]; target: PointCoordinates | null }) {
  const map = useMap()
  const pointKey = features.map((f) => {
    if (f.type === 'point') return `${f.latitude},${f.longitude}`
    if (f.type === 'rectangle') return `${f.bounds?.[0][0]},${f.bounds?.[0][1]}`
    if (f.type === 'polygon') return `${f.positions?.[0][0]},${f.positions?.[0][1]}`
    return ''
  }).join('|')

  useEffect(() => {
    const coordinates: [number, number][] = []
    features.forEach((f) => {
      if (f.type === 'point' && f.latitude !== undefined && f.longitude !== undefined) {
        coordinates.push([f.latitude, f.longitude])
      } else if (f.type === 'rectangle' && f.bounds) {
        coordinates.push(f.bounds[0], f.bounds[1])
      } else if (f.type === 'polygon' && f.positions) {
        coordinates.push(...f.positions)
      }
    })
    if (target) coordinates.push([target.latitude, target.longitude])
    if (coordinates.length > 1) {
      map.fitBounds(coordinates, { padding: [30, 30], maxZoom: 14 })
    } else if (coordinates.length === 1) {
      map.setView(coordinates[0], 13)
    }
  }, [map, pointKey, target?.latitude, target?.longitude])

  return null
}

function SpatialMapPanel({ result, running, searchCenter, refreshKey, onRunQuery }: SpatialMapPanelProps) {
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
  const [lastSearch, setLastSearch] = useState<{ mode: SearchMode, radius: number, metric: DistanceMetric } | null>(null)

  useEffect(() => {
    listTables()
      .then((items) => {
        setTables(items)
        const isSpatialCol = (c: { type: string }) => c.type === 'point' || c.type === 'geometry' || c.type === 'polygon'
        const firstSpatial = items.find((table) => table.columns.some(isSpatialCol))
        if (firstSpatial) {
          setTableName((current) => current || firstSpatial.name)
          setColumnName((current) => current || firstSpatial.columns.find(isSpatialCol)?.name || '')
        }
      })
      .catch((error: unknown) => {
        setTablesError(error instanceof Error ? error.message : 'No se pudieron cargar las tablas')
      })
  }, [refreshKey])

  const table = tables.find((item) => item.name === tableName)
  const isSpatialCol = (c: { type: string }) => c.type === 'point' || c.type === 'geometry' || c.type === 'polygon'
  const pointColumns = table?.columns.filter(isSpatialCol) ?? []
  const selectedIndex = table?.indexes.find(
    (index) => index.column_name === columnName && index.index_type === 'rtree',
  )
  const features = useMemo(() => collectFeatures(result), [result])
  const { candidateBoxes, indexMbrs } = useMemo(() => {
    if (!result || !('plan' in result) || !result.plan) return { candidateBoxes: [], indexMbrs: [] }
    const boxes = extractBoxes(result.plan)
    return { candidateBoxes: boxes.candidate, indexMbrs: boxes.index }
  }, [result])
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
    setColumnName(nextTable?.columns.find(isSpatialCol)?.name ?? '')
  }

  function generateQuery() {
    if (!tableName || !columnName || !centerIsValid) return
    const point = `POINT(${center.longitude}, ${center.latitude})`
    const metricArg = `, ${metric}`
    const sql = mode === 'radius'
      ? `SELECT * FROM ${tableName} WHERE distancia(${columnName}, ${point}${metricArg}) < ${Number(radius)};`
      : `SELECT * FROM ${tableName} ORDER BY distancia(${columnName}, ${point}${metricArg}) LIMIT ${Number(neighbors)};`
    setLastSearch({ mode, radius: Number(radius), metric })
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
              {tables.filter((item) => item.columns.some(isSpatialCol)).map((item) => (
                <option key={item.name} value={item.name}>{item.name}</option>
              ))}
            </select>
          </label>

          <label className="spatial-field">
            Geometría
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
          <div className="spatial-field-group">
            <label className="spatial-field">
              Latitud
              <input type="number" min="-90" max="90" step="0.0001" value={latitude} onChange={(event) => setLatitude(event.target.value)} />
            </label>
            <label className="spatial-field">
              Longitud
              <input type="number" min="-180" max="180" step="0.0001" value={longitude} onChange={(event) => setLongitude(event.target.value)} />
            </label>
            {!centerIsValid && (latitude !== '' || longitude !== '') && (
              <p className="error-text" style={{ gridColumn: '1 / -1', margin: '4px 0 0' }}>Lat: -90 a 90, Lon: -180 a 180</p>
            )}
          </div>

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
                aria-label={metric === 'HAVERSINE' ? 'Radio personalizado en kilómetros' : 'Radio personalizado en grados'}
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
          <p className="meta-line">No hay tablas con columnas espaciales disponibles.</p>
        )}
        {tableName && columnName && (
          <p className="spatial-index-status">
            {selectedIndex ? `R-Tree #${selectedIndex.index_id}` : 'Scan secuencial'} · {features.length} geometrías visibles
          </p>
        )}
      </div>

      <div className="spatial-map-frame">
        <MapContainer center={[INITIAL_CENTER.latitude, INITIAL_CENTER.longitude]} zoom={11} scrollWheelZoom>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitMap features={features} target={searchCenter} />
          {indexMbrs.map((box, i) => (
            <Rectangle
              key={`index-${i}`}
              bounds={[[box[1], box[0]], [box[3], box[2]]]}
              pathOptions={{ color: '#9ca3af', fillColor: '#d1d5db', fillOpacity: 0.2, weight: 1 }}
            />
          ))}
          {candidateBoxes.map((box, i) => (
            <Rectangle
              key={`box-${i}`}
              bounds={[[box[1], box[0]], [box[3], box[2]]]}
              pathOptions={{ color: '#f59e0b', fillColor: 'transparent', weight: 2, dashArray: '4 4' }}
            />
          ))}
          {searchCenter && lastSearch?.mode === 'radius' && (
            <Circle
              center={[searchCenter.latitude, searchCenter.longitude]}
              radius={lastSearch.metric === 'HAVERSINE' ? lastSearch.radius : lastSearch.radius * 111320}
              pathOptions={{ color: '#e26148', fillColor: '#e26148', fillOpacity: 0.1, weight: 1, dashArray: '4 4' }}
            />
          )}
          {searchCenter && (
            <CircleMarker
              center={[searchCenter.latitude, searchCenter.longitude]}
              radius={8}
              pathOptions={{ color: '#bd3f32', fillColor: '#e26148', fillOpacity: 0.9, weight: 2 }}
            >
              <Popup>Punto de búsqueda</Popup>
            </CircleMarker>
          )}
          {features.map((feature) => {
            const PopupContent = (
              <Popup>
                <div className="spatial-popup-content">
                  <strong>Geometría: {feature.label}</strong>
                  <table className="spatial-popup-table">
                    <tbody>
                      {feature.type === 'point' && (
                        <>
                          <tr><th>Latitud</th><td>{feature.latitude?.toFixed(5)}</td></tr>
                          <tr><th>Longitud</th><td>{feature.longitude?.toFixed(5)}</td></tr>
                        </>
                      )}
                      {feature.rowData.map((data) => (
                        <tr key={data.column}>
                          <th>{data.column}</th>
                          <td>{data.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Popup>
            )

            const centerPos: [number, number] | null =
              feature.latitude !== undefined && feature.longitude !== undefined
                ? [feature.latitude, feature.longitude]
                : null

            const knnLine = searchCenter && lastSearch?.mode === 'knn' && centerPos ? (
              <Polyline
                positions={[
                  [searchCenter.latitude, searchCenter.longitude],
                  centerPos,
                ]}
                pathOptions={{ color: '#7c3aed', weight: 2, dashArray: '4 6', opacity: 0.7 }}
              />
            ) : null

            if (feature.type === 'point' && centerPos) {
              return (
                <Fragment key={feature.id}>
                  <CircleMarker
                    center={centerPos}
                    radius={7}
                    pathOptions={{ color: '#176b66', fillColor: '#35a89a', fillOpacity: 0.88, weight: 2 }}
                  >
                    {PopupContent}
                  </CircleMarker>
                  {knnLine}
                </Fragment>
              )
            } else if (feature.type === 'rectangle' && feature.bounds) {
              return (
                <Fragment key={feature.id}>
                  <Rectangle
                    bounds={feature.bounds}
                    pathOptions={{ color: '#e26148', fillColor: 'transparent', weight: 2, dashArray: '5 5' }}
                  >
                    {PopupContent}
                  </Rectangle>
                  {centerPos && (
                    <CircleMarker
                      center={centerPos}
                      radius={5}
                      pathOptions={{ color: '#bd3f32', fillColor: '#e26148', fillOpacity: 0.85, weight: 1.5 }}
                    >
                      {PopupContent}
                    </CircleMarker>
                  )}
                  {knnLine}
                </Fragment>
              )
            } else if (feature.type === 'polygon' && feature.positions) {
              return (
                <Fragment key={feature.id}>
                  <Polygon
                    positions={feature.positions}
                    pathOptions={{ color: '#176b66', fillColor: '#35a89a', fillOpacity: 0.2, weight: 2 }}
                  >
                    {PopupContent}
                  </Polygon>
                  {centerPos && (
                    <CircleMarker
                      center={centerPos}
                      radius={5}
                      pathOptions={{ color: '#176b66', fillColor: '#35a89a', fillOpacity: 0.85, weight: 1.5 }}
                    >
                      {PopupContent}
                    </CircleMarker>
                  )}
                  {knnLine}
                </Fragment>
              )
            }
            return null
          })}
        </MapContainer>
      </div>
    </div>
  )
}

export default SpatialMapPanel