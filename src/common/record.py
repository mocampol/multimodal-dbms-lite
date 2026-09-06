"""
Database record representation.

This module defines the Record abstraction used to represent a single 
row of data in a database table.
"""

from .value import Value

class Record:
    def __init__(self, values: list):
        """
        Initializes a Record with the given list of values.
        Each value corresponds to a column in the associated table schema.
        """
        self.values = values

    def __getitem__(self, i: int):
        """
        Returns the value at the given index in the record.
        """
        return self.values[i]
