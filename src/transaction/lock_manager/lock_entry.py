from dataclasses import dataclass, field
from .lock_mode import LockMode


#RID temporal, se definira posteriormente en common
@dataclass(frozen=True)
class RID:
    page_id: int
    slot_number: int


@dataclass
class LockEntry:
    holders: dict[int, LockMode] = field(default_factory=dict)
    waiting: list[tuple[int, LockMode]] = field(default_factory=list)