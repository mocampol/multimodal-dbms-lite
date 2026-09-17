"""
WHERE clause: passes through only the records from its child that
satisfy a condition expression from the AST (a BinaryExp), already
validated by SemanticVisitor.
"""

from typing import Optional

from common.record import Record
from query.executor.plan_node import PlanNode
from query.parser.ast_nodes import BinaryExp, IdExp, NumExp, StringExp, BinaryOp


class Filter(PlanNode):
    """
    Filters child's output using condition.

    Attributes:
        child (PlanNode): Source of candidate records.
        condition (BinaryExp): Condition each record must satisfy.
        schema: Schema of the table being scanned, used to resolve the
            condition's column name to a positional index in the Record.
    """

    def __init__(self, child: PlanNode, condition: BinaryExp, schema):
        self.child = child
        self.condition = condition
        self.schema = schema

    def open(self) -> None:
        self.child.open()

    def next(self) -> Optional[Record]:
        while True:
            record = self.child.next()
            if record is None:
                return None
            if self._matches(record):
                return record

    def close(self) -> None:
        self.child.close()

    def _matches(self, record: Record) -> bool:
        left_idx = self.schema.column_index(self.condition.left.value)
        left_value = record[left_idx].data

        right_exp = self.condition.right
        if isinstance(right_exp, IdExp):
            right_idx = self.schema.column_index(right_exp.value)
            right_value = record[right_idx].data
        elif isinstance(right_exp, (NumExp, StringExp)):
            right_value = right_exp.value
        else:
            raise ValueError(f"Expresión de valor no soportada: {right_exp!r}")

        op = self.condition.op
        if op == BinaryOp.EQ_OP:
            return left_value == right_value
        if op == BinaryOp.NEQ_OP:
            return left_value != right_value
        if op == BinaryOp.LE_OP:
            return left_value < right_value
        if op == BinaryOp.LEQ_OP:
            return left_value <= right_value
        if op == BinaryOp.GT_OP:
            return left_value > right_value
        if op == BinaryOp.GEQ_OP:
            return left_value >= right_value

        raise ValueError(f"Operador no soportado: {op}")
