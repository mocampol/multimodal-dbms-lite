"""Persistent R-Tree index over geographic Point2D values."""

from storage.buffer_manager import BufferManager
from storage.heap.rid import RID
from spatial.geometry import BoundingBox, Point2D

from .exceptions import RTreeNodeCapacityError
from .node import RTreeNode
from .page import InternalEntry, LeafEntry, NodeType, RTreePage


class RTree:
    """R-Tree supporting point insert, exact range search, and delete.

    The tree persists node pages through ``buffer_manager``. As with the
    B-Tree, callers store ``root_page_id`` in catalog metadata and pass it
    when reopening an existing index.
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
            self._root_page_id = self._create_node(NodeType.LEAF)
        else:
            raise ValueError(
                "El archivo R-Tree ya existe pero no se indicó root_page_id"
            )

    @property
    def root_page_id(self) -> int:
        return self._root_page_id

    def insert(self, geom, rid: RID) -> None:
        """Insert one point/RID pair. Duplicate points and RIDs are allowed."""
        if not isinstance(geom, (Point2D, BoundingBox)):
            raise TypeError("geom debe ser un Point2D o un BoundingBox")
        if not isinstance(rid, RID):
            raise TypeError("rid debe ser un RID")

        if isinstance(geom, Point2D):
            point_mbr = BoundingBox(geom.x, geom.y, geom.x, geom.y)
        else:
            point_mbr = geom
        left_mbr, split = self._insert_recursive(
            self._root_page_id,
            LeafEntry(point_mbr, rid),
        )
        if split is None:
            return

        right_page_id, right_mbr = split
        self._promote_root(
            self._root_page_id,
            left_mbr,
            right_page_id,
            right_mbr,
        )

    def dump_all_mbrs(self) -> list[BoundingBox]:
        """Return the bounding boxes of all nodes in the tree."""
        mbrs = []
        stack = [self._root_page_id]
        
        while stack:
            page_id = stack.pop()
            page = self.bm.fetch_page(page_id)
            try:
                node = RTreeNode(page)
                if len(node.entries) > 0:
                    mbrs.append(node.bounding_box)
                    if node.node_type == NodeType.INTERNAL:
                        for entry in node.entries:
                            stack.append(entry.child_page_id)
            finally:
                self.bm.unpin_page(page_id, is_dirty=False)
                
        return mbrs

    def fetch_node(self, page_id: int) -> RTreeNode:
        """Fetch and pin a node; callers must unpin its page when finished."""
        return self._fetch_node(page_id)

    def split_node(
        self,
        page_id: int,
        pending_entry: LeafEntry | InternalEntry | None = None,
    ) -> int:
        """Split one node; promote a new root when splitting the current root."""
        page = self.bm.fetch_page(page_id)
        dirty = False
        try:
            node = RTreeNode(page)
            right_page_id, left_mbr, right_mbr = self._split_page(
                node,
                pending_entry,
            )
            dirty = True
        finally:
            self.bm.unpin_page(page_id, is_dirty=dirty)

        if page_id == self._root_page_id:
            self._promote_root(page_id, left_mbr, right_page_id, right_mbr)
        return right_page_id

    def _promote_root(
        self,
        left_page_id: int,
        left_mbr: BoundingBox,
        right_page_id: int,
        right_mbr: BoundingBox,
    ) -> None:
        new_root_page_id = self._create_node(NodeType.INTERNAL)
        page = self.bm.fetch_page(new_root_page_id)
        try:
            new_root = RTreeNode(page)
            if not new_root.insert_internal_entry(left_mbr, left_page_id):
                raise RTreeNodeCapacityError("el root interno no tiene capacidad")
            if not new_root.insert_internal_entry(right_mbr, right_page_id):
                raise RTreeNodeCapacityError("el root interno no tiene capacidad")
        except Exception:
            self.bm.unpin_page(new_root_page_id, is_dirty=False)
            raise
        self.bm.unpin_page(new_root_page_id, is_dirty=True)
        self._root_page_id = new_root_page_id

    def search_bbox(self, min_point: Point2D, max_point: Point2D) -> list[RID]:
        """Return RIDs whose exact indexed points lie within the closed box."""
        if not isinstance(min_point, Point2D) or not isinstance(max_point, Point2D):
            raise TypeError("search_bbox requiere dos Point2D")
        return self.range_query(
            BoundingBox(min_point.x, min_point.y, max_point.x, max_point.y)
        )

    def range_query(self, bbox: BoundingBox) -> list[RID]:
        """Return every matching RID after exact filtering of leaf candidates."""
        if not isinstance(bbox, BoundingBox):
            raise TypeError("range_query requiere un BoundingBox")

        results = []
        pending_page_ids = [self._root_page_id]
        while pending_page_ids:
            page_id = pending_page_ids.pop()
            node = self._fetch_node(page_id)
            if node.is_leaf:
                for entry in node.entries:
                    if not entry.mbr.intersects(bbox):
                        continue
                    exact_point = Point2D(entry.mbr.min_x, entry.mbr.min_y)
                    if bbox.contains(exact_point):
                        results.append(entry.rid)
                self.bm.unpin_page(page_id, is_dirty=False)
                continue

            child_page_ids = [
                entry.child_page_id
                for entry in node.entries
                if entry.mbr.intersects(bbox)
            ]
            self.bm.unpin_page(page_id, is_dirty=False)
            pending_page_ids.extend(reversed(child_page_ids))
        return results

    def search_point(self, point: Point2D) -> list[RID]:
        """Return all RIDs stored at exactly ``point``."""
        if not isinstance(point, Point2D):
            raise TypeError("search_point requiere un Point2D")
        return self.range_query(BoundingBox(point.x, point.y, point.x, point.y))

    def delete(self, point: Point2D, rid: RID) -> bool:
        """Delete one matching point/RID pair, returning whether it existed."""
        if not isinstance(point, Point2D):
            raise TypeError("point debe ser un Point2D")
        if not isinstance(rid, RID):
            raise TypeError("rid debe ser un RID")

        removed, _ = self._delete_recursive(self._root_page_id, point, rid)
        if removed:
            self._shrink_root()
        return removed

    def _insert_recursive(
        self,
        page_id: int,
        pending_entry: LeafEntry,
    ) -> tuple[BoundingBox, tuple[int, BoundingBox] | None]:
        page = self.bm.fetch_page(page_id)
        dirty = False
        try:
            node = RTreeNode(page)
            if node.is_leaf:
                if node.insert_leaf_entry(pending_entry.mbr, pending_entry.rid):
                    dirty = True
                    return node.bounding_box, None
                right_page_id, left_mbr, right_mbr = self._split_page(
                    node, pending_entry
                )
                dirty = True
                return left_mbr, (right_page_id, right_mbr)

            entries = node.entries
            selected = min(
                entries,
                key=lambda entry: (
                    self._enlargement(entry.mbr, pending_entry.mbr),
                    entry.mbr.area,
                    entry.child_page_id,
                ),
            )
        finally:
            self.bm.unpin_page(page_id, is_dirty=dirty)

        child_mbr, child_split = self._insert_recursive(
            selected.child_page_id,
            pending_entry,
        )

        page = self.bm.fetch_page(page_id)
        dirty = False
        try:
            node = RTreeNode(page)
            entries = node.entries
            selected_index = next(
                index
                for index, entry in enumerate(entries)
                if entry.child_page_id == selected.child_page_id
            )
            entries[selected_index] = InternalEntry(
                child_mbr,
                selected.child_page_id,
            )
            if child_split is None:
                node.replace_entries(entries)
                dirty = True
                return node.bounding_box, None

            right_child_page_id, right_child_mbr = child_split
            new_child_entry = InternalEntry(right_child_mbr, right_child_page_id)
            if len(entries) < node.capacity:
                node.replace_entries(entries + [new_child_entry])
                dirty = True
                return node.bounding_box, None

            node.replace_entries(entries)
            right_page_id, left_mbr, right_mbr = self._split_page(
                node,
                new_child_entry,
            )
            dirty = True
            return left_mbr, (right_page_id, right_mbr)
        finally:
            self.bm.unpin_page(page_id, is_dirty=dirty)

    def _split_page(
        self,
        node: RTreeNode,
        pending_entry: LeafEntry | InternalEntry,
    ) -> tuple[int, BoundingBox, BoundingBox]:
        right_page_id = self.bm.allocate_page()
        right_page = self.bm.fetch_page(right_page_id)
        right_dirty = False
        try:
            right_node = node.split(right_page, pending_entry)
            right_dirty = True
            left_mbr = node.bounding_box
            right_mbr = right_node.bounding_box
        finally:
            self.bm.unpin_page(right_page_id, is_dirty=right_dirty)
        if left_mbr is None or right_mbr is None:
            raise RTreeNodeCapacityError("el split produjo un nodo sin MBR")
        return right_page_id, left_mbr, right_mbr

    def _delete_recursive(
        self,
        page_id: int,
        point: Point2D,
        rid: RID,
    ) -> tuple[bool, BoundingBox | None]:
        page = self.bm.fetch_page(page_id)
        node = self._deserialize_page(page, page_id)
        if node.is_leaf:
            entries = node.entries
            matching_index = next(
                (
                    index
                    for index, entry in enumerate(entries)
                    if entry.rid == rid
                    and entry.mbr.min_x == point.x
                    and entry.mbr.min_y == point.y
                    and entry.mbr.max_x == point.x
                    and entry.mbr.max_y == point.y
                ),
                None,
            )
            if matching_index is None:
                self.bm.unpin_page(page_id, is_dirty=False)
                return False, node.bounding_box
            del entries[matching_index]
            node.replace_entries(entries)
            mbr = node.bounding_box
            self.bm.unpin_page(page_id, is_dirty=True)
            return True, mbr

        candidates = [
            entry.child_page_id
            for entry in node.entries
            if entry.mbr.contains(point)
        ]
        self.bm.unpin_page(page_id, is_dirty=False)

        for child_page_id in candidates:
            removed, child_mbr = self._delete_recursive(child_page_id, point, rid)
            if not removed:
                continue

            page = self.bm.fetch_page(page_id)
            node = self._deserialize_page(page, page_id)
            entries = node.entries
            child_index = next(
                index
                for index, entry in enumerate(entries)
                if entry.child_page_id == child_page_id
            )
            if child_mbr is None:
                del entries[child_index]
            else:
                entries[child_index] = InternalEntry(child_mbr, child_page_id)
            node.replace_entries(entries)
            mbr = node.bounding_box
            self.bm.unpin_page(page_id, is_dirty=True)
            return True, mbr

        return False, None

    def _shrink_root(self) -> None:
        page = self.bm.fetch_page(self._root_page_id)
        node = self._deserialize_page(page, self._root_page_id)
        if node.is_leaf or len(node.entries) > 1:
            self.bm.unpin_page(page.page_id, is_dirty=False)
            return
        if not node.entries:
            RTreeNode.init_leaf(page)
            self.bm.unpin_page(page.page_id, is_dirty=True)
            return

        new_root_page_id = node.entries[0].child_page_id
        self.bm.unpin_page(page.page_id, is_dirty=False)
        self._root_page_id = new_root_page_id

    def _create_node(self, node_type: NodeType) -> int:
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

    def _fetch_node(self, page_id: int) -> RTreeNode:
        page = self.bm.fetch_page(page_id)
        return self._deserialize_page(page, page_id)

    def _deserialize_page(self, page, page_id: int) -> RTreeNode:
        try:
            return RTreeNode(page)
        except Exception:
            self.bm.unpin_page(page_id, is_dirty=False)
            raise

    @staticmethod
    def _enlargement(current: BoundingBox, added: BoundingBox) -> float:
        return current.union(added).area - current.area