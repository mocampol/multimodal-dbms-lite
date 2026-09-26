class EmptyCSVError(Exception):
    pass


class InconsistentRowError(Exception):
    pass


class UnsupportedTypeError(Exception):
    pass


class DuplicateColumnNameError(Exception):
    pass