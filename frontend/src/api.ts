const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export interface ColumnInfo {
  name: string
  type: string
  size?: number
  nullable?: boolean
  is_unique?: boolean
  is_primary_key: boolean
}

export interface IndexInfo {
  index_id: number
  column_name: string
  index_type: 'btree' | 'hash' | string
  root_page_id?: number
}

export interface TableSummary {
  name: string
  storage_type: string
  columns: Pick<ColumnInfo, 'name' | 'type' | 'is_primary_key'>[]
}

export interface TableDetail {
  name: string
  storage_type: string
  columns: ColumnInfo[]
  indexes: IndexInfo[]
  // Not returned by the backend yet; panels must tolerate undefined.
  record_count?: number
  page_count?: number
}

export interface PlanNode {
  node: string
  access?: string
  table?: string
  index_type?: string
  key?: unknown
  condition?: string
  columns?: string[]
  order_by?: string[]
  group_by?: string[]
  join_type?: string
  children?: PlanNode[]
}

export interface SelectQueryResult {
  type: 'SELECT'
  columns: string[]
  rows: Record<string, unknown>[]
  row_count: number
  execution_ms: number
  plan: PlanNode
}

export interface MutationQueryResult {
  type: 'INSERT' | 'UPDATE' | 'DELETE'
  rows_affected: number
  execution_ms: number
  plan: null
}

export interface OtherQueryResult {
  type: string
  ok: true
  execution_ms: number
  plan: null
  transaction_id?: number
}

export type QueryResult = SelectQueryResult | MutationQueryResult | OtherQueryResult

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { error?: string } | null
    throw new ApiError(response.status, body?.error ?? response.statusText)
  }

  return response.json() as Promise<T>
}

export function getHealth() {
  return request<{ status: string; tables: number }>('/health')
}

export function listTables() {
  return request<TableSummary[]>('/tables')
}

export function getTable(name: string) {
  return request<TableDetail>(`/tables/${encodeURIComponent(name)}`)
}

export function runQuery(sql: string) {
  return request<QueryResult>('/query', {
    method: 'POST',
    body: JSON.stringify({ sql }),
  })
}
