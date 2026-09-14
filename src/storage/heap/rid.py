"""
Record ID: the stable logical identifier of a record within a Heap File.
"""


class RID:
    """
    A Record ID uniquely and stably identifies a record: (page_id, slot).

    Stability matters because other structures (B+ Tree secondary indexes,
    Extendible Hash buckets) store RIDs to point at heap records. The RID
    itself never changes even if the record's physical bytes move within
    or across pages — see heap_file.py's forwarding pointer mechanism for
    how cross-page moves stay transparent to whoever holds an RID.
    """

    def __init__(self, page_id: int, slot: int):
        self.page_id = page_id
        self.slot = slot

    def __eq__(self, other):
        return (
            isinstance(other, RID)
            and self.page_id == other.page_id
            and self.slot == other.slot
        )

    def __hash__(self):
        return hash((self.page_id, self.slot))

    def __repr__(self):
        return f"RID({self.page_id}, {self.slot})"
