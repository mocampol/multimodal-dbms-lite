import { isBatchQueryResult, isExplainQueryResult, type PlanNode, type QueryResult, type SingleQueryResult } from './api'

interface ExecutionPlanPanelProps {
  result: QueryResult | null
  error: string | null
  running: boolean
}

function describeNode(node: PlanNode): string {
  const base = describeNodeShape(node)
  if (node.rows !== undefined && node.time_ms !== undefined) {
    return `${base} — ${node.rows} fila(s), ${node.time_ms.toFixed(3)} ms`
  }
  return base
}

function describeNodeShape(node: PlanNode): string {
  switch (node.node) {
    case 'SeqScan':
      return `Sequential Scan sobre "${node.table}"`
    case 'ClusteredScan':
      return `Clustered Index Scan (clave = ${JSON.stringify(node.key)})`
    case 'IndexScan':
      return `Index Scan usando ${node.index_type ?? 'índice'} (clave = ${JSON.stringify(node.key)})`
    case 'BitmapScan':
      return `Bitmap Scan (clave = ${JSON.stringify(node.key)})`
    case 'Filter':
      return `Filter ${node.condition ?? ''}`
    case 'Sort':
      return `Sort por ${node.order_by?.join(', ') ?? ''}`
    case 'Limit':
      return `Limit ${node.limit ?? ''}`
    case 'HashAggregate':
      return `Hash Aggregate agrupado por ${node.group_by?.join(', ') ?? ''}`
    case 'GroupAggregate':
      return `Group Aggregate agrupado por ${node.group_by?.join(', ') ?? ''}`
    case 'SortAggregate':
      return `Sort Aggregate agrupado por ${node.group_by?.join(', ') ?? ''}`
    case 'HashJoin':
      return 'Hash Join'
    case 'NestedLoopJoin':
      return 'Nested Loop Join'
    case 'SortMergeJoin':
      return 'Sort Merge Join'
    case 'Projection':
      return `Projection ${node.columns?.join(', ') ?? '*'}`
    default:
      return node.node
  }
}

function flattenPlan(node: PlanNode): string[] {
  const steps: string[] = []
  for (const child of node.children ?? []) {
    steps.push(...flattenPlan(child))
  }
  steps.push(describeNode(node))
  return steps
}

function explainHeader(result: SingleQueryResult): string | null {
  if (!isExplainQueryResult(result)) return null
  if (result.type === 'explain_analyze') {
    return `EXPLAIN ANALYZE — ${result.row_count} fila(s), ${result.elapsed_ms?.toFixed(3)} ms`
  }
  return 'EXPLAIN (plan sin ejecutar)'
}

function ExecutionPlanPanel({ result, error, running }: ExecutionPlanPanelProps) {
  if (running) return <p className="panel-placeholder">Ejecutando consulta...</p>
  if (error) return <p className="error-text">Error: {error}</p>
  if (!result) return <p className="panel-placeholder">Execution steps will be listed here.</p>
  if (isBatchQueryResult(result)) {
    const planned = result.statements.filter(
      (statement): statement is SingleQueryResult => !isBatchQueryResult(statement) && statement.plan !== null,
    )
    if (planned.length === 0) return <p className="panel-placeholder">Estas operaciones no tienen plan de ejecución.</p>
    return (
      <ol className="plan-list">
        {planned.map((statement, index) => (
          <li key={index}>
            {statement.type}
            {explainHeader(statement) && <span className="meta-line"> · {explainHeader(statement)}</span>}
            <ol className="plan-list">
              {flattenPlan(statement.plan!).map((step, stepIndex) => <li key={stepIndex}>{step}</li>)}
            </ol>
          </li>
        ))}
      </ol>
    )
  }
  if (!result.plan) return <p className="panel-placeholder">Esta operación no tiene plan de ejecución.</p>

  const steps = flattenPlan(result.plan)
  const header = explainHeader(result)

  return (
    <>
      {header && <p className="meta-line">{header}</p>}
      <ol className="plan-list">
        {steps.map((step, index) => (
          <li key={index}>{step}</li>
        ))}
      </ol>
    </>
  )
}

export default ExecutionPlanPanel