"""
Database record representation.

Defines the Record abstraction used to represent one row of data in a database table.
"""


from .value import Value
from .schema import Schema


class Record:
    def __init__(self, values: list):
        """
        Represents one row of a table.

        A Record contains Values in the same order as the columns
        in its associated Schema.
        """
        if not all(isinstance(value, Value) for value in values):
            raise TypeError("Todos los elementos del Record deben ser Value")

        self.values = values


    def validate(self, schema: Schema) -> bool:
        """
        Validates the record against a Schema.

        Checks:
            1. Number of values equals number of columns.
            2. Each value is compatible with its column (type, NULL, size).
        """
        if len(self.values) != len(schema.columns):
            return False

        for value, column in zip(self.values, schema.columns):
            if not column.validate(value):
                return False

        return True


    def byte_size(self) -> int:
        """
        Returns the physical size of the values in the record.
        Does not include page-level or slot-level metadata.
        """
        return sum(value.byte_size() for value in self.values)


    def __getitem__(self, i: int):
        return self.values[i]


    def __len__(self):
        return len(self.values)


    def __iter__(self):
        return iter(self.values)


    def __repr__(self):
        return f"Record({[v.data for v in self.values]})"