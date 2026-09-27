"""Persistent R-Tree node operations."""

import math

from storage.heap.rid import RID
from storage.page import Page
from spatial.geometry import BoundingBox

from .exceptions import RTreeNodeCapacityError, RTreeNodeSplitError
from .page import InternalEntry, LeafEntry, NodeType, RTreePage


class RTreeNode:
    def __init__(self, page: Page):
        self.page = page
        self.node_type, self._entries = RTreePage.deserialize(page)

    @classmethod
    def init_leaf(cls, page: Page) -> "RTreeNode":
        RTreePage.initialize(page, NodeType.LEAF)
        return cls(page)

    @classmethod
    def init_internal(cls, page: Page) -> "RTreeNode":
        RTreePage.initialize(page, NodeType.INTERNAL)
        return cls(page)

    @property
    def is_leaf(self) -> bool:
        return self.node_type == NodeType.LEAF

    @property
    def entries(self) -> list[LeafEntry | InternalEntry]:
        return list(self._entries)

    @property
    def capacity(self) -> int:
        return RTreePage.max_entries(self.page.size, self.node_type)

    @property
    def bounding_box(self) -> BoundingBox | None:
        if not self._entries:
            return None
        result = self._entries[0].mbr
        for entry in self._entries[1:]:
            result = result.union(entry.mbr)
        return result

    def insert_leaf_entry(self, mbr: BoundingBox, rid: RID) -> bool:
        if not self.is_leaf:
            raise ValueError("insert_leaf_entry() solo aplica a nodos hoja")
        entry = LeafEntry(mbr, rid)
        if len(self._entries) >= self.capacity:
            return False
        self._replace_entries(self._entries + [entry])
        return True

    def insert_internal_entry(self, mbr: BoundingBox, child_page_id: int) -> bool:
        if self.is_leaf:
            raise ValueError("insert_internal_entry() solo aplica a nodos internos")
        entry = InternalEntry(mbr, child_page_id)
        if len(self._entries) >= self.capacity:
            return False
        self._replace_entries(self._entries + [entry])
        return True

    def split(
        self,
        right_page: Page,
        pending_entry: LeafEntry | InternalEntry | None = None,
    ) -> "RTreeNode":
        """Split this node and write its sibling to right_page.

        A pending entry lets callers split a full page plus the entry that
        overflowed it without ever writing an over-capacity page.
        """
        if not isinstance(right_page, Page):
            raise TypeError("right_page debe ser un Page")
        if right_page.page_id == self.page.page_id:
            raise ValueError("right_page debe tener un page_id distinto")
        if right_page.size != self.page.size:
            raise ValueError("las páginas del split deben tener el mismo tamaño")

        all_entries = list(self._entries)
        if pending_entry is not None:
            expected_type = LeafEntry if self.is_leaf else InternalEntry
            if not isinstance(pending_entry, expected_type):
                raise TypeError("pending_entry no coincide con el tipo del nodo")
            all_entries.append(pending_entry)
        if len(all_entries) < 2:
            raise RTreeNodeSplitError("se requieren al menos dos entradas para dividir")

        if len(all_entries) > 2 * self.capacity:
            raise RTreeNodeCapacityError(
                "las entradas no caben en dos páginas del mismo tamaño"
            )

        left_entries, right_entries = self._linear_split(all_entries)
        if not left_entries or not right_entries:
            raise RTreeNodeSplitError("el split produjo un nodo vacío")

        right_node = (
            RTreeNode.init_leaf(right_page)
            if self.is_leaf
            else RTreeNode.init_internal(right_page)
        )
        self._replace_entries(left_entries)
        right_node._replace_entries(right_entries)
        return right_node

    def replace_entries(self, entries: list[LeafEntry | InternalEntry]) -> None:
        """Replace all entries while enforcing the node kind and page capacity."""
        expected_type = LeafEntry if self.is_leaf else InternalEntry
        if any(not isinstance(entry, expected_type) for entry in entries):
            raise TypeError("las entradas no coinciden con el tipo del nodo")
        self._replace_entries(entries)

    def _linear_split(
        self,
        entries: list[LeafEntry | InternalEntry],
    ) -> tuple[list[LeafEntry | InternalEntry], list[LeafEntry | InternalEntry]]:
        seeds = self._linear_split_seeds(entries)
        left_entries = [entries[seeds[0]]]
        right_entries = [entries[seeds[1]]]
        remaining = [
            entry for index, entry in enumerate(entries) if index not in seeds
        ]
        minimum_fill = max(1, math.ceil(self.capacity * 0.4))

        while remaining:
            if len(left_entries) + len(remaining) == minimum_fill:
                left_entries.extend(remaining)
                break
            if len(right_entries) + len(remaining) == minimum_fill:
                right_entries.extend(remaining)
                break

            left_mbr = self._entries_mbr(left_entries)
            right_mbr = self._entries_mbr(right_entries)
            next_index = max(
                range(len(remaining)),
                key=lambda index: abs(
                    self._enlargement(left_mbr, remaining[index].mbr)
                    - self._enlargement(right_mbr, remaining[index].mbr)
                ),
            )
            entry = remaining.pop(next_index)
            left_enlargement = self._enlargement(left_mbr, entry.mbr)
            right_enlargement = self._enlargement(right_mbr, entry.mbr)
            if left_enlargement < right_enlargement:
                left_entries.append(entry)
            elif right_enlargement < left_enlargement:
                right_entries.append(entry)
            elif left_mbr.area < right_mbr.area:
                left_entries.append(entry)
            elif right_mbr.area < left_mbr.area:
                right_entries.append(entry)
            elif len(left_entries) <= len(right_entries):
                left_entries.append(entry)
            else:
                right_entries.append(entry)

        return left_entries, right_entries

    @staticmethod
    def _linear_split_seeds(entries: list[LeafEntry | InternalEntry]) -> tuple[int, int]:
        best_pair = None
        best_separation = float("-inf")
        for minimum, maximum in (
            ("min_x", "max_x"),
            ("min_y", "max_y"),
        ):
            highest_low_index = max(
                range(len(entries)), key=lambda index: getattr(entries[index].mbr, minimum)
            )
            lowest_high_index = min(
                range(len(entries)), key=lambda index: getattr(entries[index].mbr, maximum)
            )
            lowest_low = min(getattr(entry.mbr, minimum) for entry in entries)
            highest_high = max(getattr(entry.mbr, maximum) for entry in entries)
            width = highest_high - lowest_low
            separation = (
                (
                    getattr(entries[highest_low_index].mbr, minimum)
                    - getattr(entries[lowest_high_index].mbr, maximum)
                )
                / width
                if width
                else 0
            )
            if separation > best_separation:
                best_pair = highest_low_index, lowest_high_index
                best_separation = separation

        if best_pair[0] != best_pair[1]:
            return best_pair

        centers = [
            ((entry.mbr.min_x + entry.mbr.max_x) / 2,
             (entry.mbr.min_y + entry.mbr.max_y) / 2)
            for entry in entries
        ]
        return max(
            (
                (first_index, second_index)
                for first_index in range(len(entries))
                for second_index in range(first_index + 1, len(entries))
            ),
            key=lambda pair: math.dist(centers[pair[0]], centers[pair[1]]),
        )

    @staticmethod
    def _entries_mbr(entries: list[LeafEntry | InternalEntry]) -> BoundingBox:
        result = entries[0].mbr
        for entry in entries[1:]:
            result = result.union(entry.mbr)
        return result

    @staticmethod
    def _enlargement(current: BoundingBox, added: BoundingBox) -> float:
        return current.union(added).area - current.area

    def _replace_entries(self, entries: list[LeafEntry | InternalEntry]) -> None:
        RTreePage.serialize(self.page, self.node_type, entries)
        self._entries = list(entries)
