import type { PlanNode, QueryResult } from './api'

interface ExecutionPlanPanelProps {
  result: QueryResult | null
  error: string | null
  running: boolean
}

function describeNode(node: PlanNode): string {
  switch (node.node) {
    case 'SeqScan':
      return `Sequential Scan sobre "${node.table}"`
    case 'ClusteredScan':
      return `Clustered Index Scan (clave = ${JSON.stringify(node.key)})`
    case 'IndexScan':
      return `Index Scan usando ${node.index_type ?? 'índice'} (clave = ${JSON.stringify(node.key)})`
    case 'Filter':
      return `Filter ${node.condition ?? ''}`
    case 'Sort':
      return `Sort por ${node.order_by?.join(', ') ?? ''}`
    case 'HashAggregate':
      return `Hash Aggregate agrupado por ${node.group_by?.join(', ') ?? ''}`
    case 'GroupAggregate':
      return 'Group Aggregate'
    case 'HashJoin':
      return `Hash Join`
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

function ExecutionPlanPanel({ result, error, running }: ExecutionPlanPanelProps) {
  if (running) return <p className="panel-placeholder">Ejecutando consulta...</p>
  if (error) return <p className="panel-placeholder">Error: {error}</p>
  if (!result) return <p className="panel-placeholder">Execution steps will be listed here.</p>
  if (!result.plan) return <p className="panel-placeholder">Esta operación no tiene plan de ejecución.</p>

  const steps = flattenPlan(result.plan)

  return (
    <ol className="plan-list">
      {steps.map((step, index) => (
        <li key={index}>{step}</li>
      ))}
    </ol>
  )
}

export default ExecutionPlanPanel
