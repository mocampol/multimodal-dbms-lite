"""
Catalog-specific exceptions.
"""


class TableAlreadyExistsError(Exception):
    pass


class TableNotFoundError(Exception):
    pass


class ColumnNotFoundError(Exception):
    pass


class UniqueConstraintError(Exception):
    pass
