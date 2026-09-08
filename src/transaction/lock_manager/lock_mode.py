from enum import Enum

class LockMode(Enum):
    SHARED = "S"
    EXCLUSIVE = "X"

def compatible(held: LockMode, requested: LockMode) -> bool:
    return held == LockMode.SHARED and requested == LockMode.SHARED