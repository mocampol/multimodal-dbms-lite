class RTreeError(Exception):
    """Base exception for persistent R-Tree page operations."""


class RTreePageCorruptionError(RTreeError):
    """Raised when a page fails its format, bounds, or checksum validation."""


class RTreeNodeCapacityError(RTreeError):
    """Raised when an entry set cannot fit in the physical page format."""


class RTreeNodeSplitError(RTreeError):
    """Raised when a node cannot be split into two valid pages."""