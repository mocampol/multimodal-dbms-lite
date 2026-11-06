class EmptyCSVError(Exception):
    pass


class InconsistentRowError(Exception):
    pass


class UnsupportedTypeError(Exception):
    pass


class DuplicateColumnNameError(Exception):
    pass


class MediaDirectoryNotFoundError(Exception):
    pass


class EmptyMediaDirectoryError(Exception):
    pass
