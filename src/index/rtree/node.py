"""R-Tree node operations and root-page lifecycle."""

from storage.buffer_manager import BufferManager
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

        centers_x = [(entry.mbr.min_x + entry.mbr.max_x) / 2 for entry in all_entries]
        centers_y = [(entry.mbr.min_y + entry.mbr.max_y) / 2 for entry in all_entries]
        spread_x = max(centers_x) - min(centers_x)
        spread_y = max(centers_y) - min(centers_y)
        center_axis = 0 if spread_x >= spread_y else 1
        ordered = sorted(
            all_entries,
            key=lambda entry: (
                (entry.mbr.min_x + entry.mbr.max_x) / 2
                if center_axis == 0
                else (entry.mbr.min_y + entry.mbr.max_y) / 2,
                entry.mbr.min_x,
                entry.mbr.min_y,
            ),
        )
        split_at = len(ordered) // 2
        left_entries, right_entries = ordered[:split_at], ordered[split_at:]
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

    def _replace_entries(self, entries: list[LeafEntry | InternalEntry]) -> None:
        RTreePage.serialize(self.page, self.node_type, entries)
        self._entries = list(entries)


class RTree:
    """Owns the root page ID and allocates node pages through a BufferManager.

    As with BTree, callers persist root_page_id in catalog metadata and pass it
    when reopening an existing index. Node pages themselves live in the file.
    """

    def __init__(self, buffer_manager: BufferManager, root_page_id: int | None = None):
        self.bm = buffer_manager
        if root_page_id is not None:
            self._root_page_id = root_page_id
            page = self.bm.fetch_page(root_page_id)
            try:
                RTreeNode(page)
            finally:
                self.bm.unpin_page(root_page_id, is_dirty=False)
        elif self.bm.file_manager.page_count() == 0:
            self._root_page_id = self.create_node(NodeType.LEAF)
        else:
            raise ValueError(
                "El archivo R-Tree ya existe pero no se indicó root_page_id"
            )

    @property
    def root_page_id(self) -> int:
        return self._root_page_id

    def create_node(self, node_type: NodeType) -> int:
        page_id = self.bm.allocate_page()
        page = self.bm.fetch_page(page_id)
        try:
            if node_type == NodeType.LEAF:
                RTreeNode.init_leaf(page)
            elif node_type == NodeType.INTERNAL:
                RTreeNode.init_internal(page)
            else:
                raise ValueError(f"Tipo de nodo desconocido: {node_type!r}")
        except Exception:
            self.bm.unpin_page(page_id, is_dirty=False)
            raise
        self.bm.unpin_page(page_id, is_dirty=True)
        return page_id

    def fetch_node(self, page_id: int) -> RTreeNode:
        """Fetch and pin a node; caller must unpin its page when finished."""
        page = self.bm.fetch_page(page_id)
        try:
            return RTreeNode(page)
        except Exception:
            self.bm.unpin_page(page_id, is_dirty=False)
            raise

    def split_node(
        self,
        page_id: int,
        pending_entry: LeafEntry | InternalEntry | None = None,
    ) -> int:
        """Split a node and return its sibling page ID.

        Splitting the root also creates and publishes a new internal root. A
        non-root caller is responsible for adding the sibling to its parent.
        """
        page = self.bm.fetch_page(page_id)
        right_page_id = None
        right_page_pinned = False
        left_dirty = False
        try:
            node = RTreeNode(page)
            if page_id == self._root_page_id and RTreePage.max_entries(
                page.size, NodeType.INTERNAL
            ) < 2:
                raise RTreeNodeCapacityError(
                    "el tamaño de página no permite crear un root interno con dos hijos"
                )
            right_page_id = self.bm.allocate_page()
            right_page = self.bm.fetch_page(right_page_id)
            right_page_pinned = True
            right_node = node.split(right_page, pending_entry)
            left_mbr = node.bounding_box
            right_mbr = right_node.bounding_box
            self.bm.unpin_page(right_page_id, is_dirty=True)
            right_page_pinned = False
            left_dirty = True
        finally:
            if right_page_pinned:
                self.bm.unpin_page(right_page_id, is_dirty=False)
            self.bm.unpin_page(page_id, is_dirty=left_dirty)

        if page_id == self._root_page_id:
            new_root_page_id = self.create_node(NodeType.INTERNAL)
            root_page = self.bm.fetch_page(new_root_page_id)
            try:
                root = RTreeNode(root_page)
                if not root.insert_internal_entry(left_mbr, page_id):
                    raise RTreeNodeCapacityError("el root interno no tiene capacidad")
                if not root.insert_internal_entry(right_mbr, right_page_id):
                    raise RTreeNodeCapacityError("el root interno no tiene capacidad")
            except Exception:
                self.bm.unpin_page(new_root_page_id, is_dirty=False)
                raise
            self.bm.unpin_page(new_root_page_id, is_dirty=True)
            self._root_page_id = new_root_page_id
        return right_page_id