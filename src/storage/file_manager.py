"""
Raw disk access layer, one physical file per table.

Translates logical page_ids into physical byte offsets within a file.
Performs actual disk I/O on every call — no caching happens here.
Caching and pinning are the Buffer Manager's responsibility, layered
on top of this module.

Classes:
    FileManager: Manages a single table's physical file on disk.
"""

import os

from storage.page import Page, PAGE_SIZE


class FileManager:
    """
    Manages allocation and physical I/O for one table's data file.

    Each table gets its own FileManager instance pointing at its own
    file on disk (e.g. data/users.tbl). Page numbering is 0-based and
    contiguous: page_id N lives at byte offset N * PAGE_SIZE.

    Attributes:
        file_path (str): Path to the physical file on disk.
        page_size (int): Size of each page in bytes.
    """

    def __init__(self, file_path: str, page_size: int = PAGE_SIZE):
        self.file_path = file_path
        self.page_size = page_size

        # Create the file if it doesn't exist yet
        if not os.path.exists(self.file_path):
            os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
            open(self.file_path, "wb").close()

    def _page_count(self) -> int:
        """
        Returns how many pages currently exist in the file, based on
        its size on disk.
        """
        size_on_disk = os.path.getsize(self.file_path)
        return size_on_disk // self.page_size

    def allocate_page(self) -> int:
        """
        Extends the file by one zero-filled page and returns its page_id.
        This is the only way new pages come into existence — page_ids
        are never reused, even after logical deletion (heap/sequential
        deletes are lazy, per spec).
        """
        page_id = self._page_count()
        blank = bytearray(self.page_size)

        with open(self.file_path, "r+b") as f:
            f.seek(page_id * self.page_size)
            f.write(blank)

        return page_id

    def reset(self):
        """Truncate this file so a derived structure can be rebuilt."""
        with open(self.file_path, "wb"):
            pass

    def read_page(self, page_id: int) -> Page:
        """
        Reads page_id from disk and returns it as a Page object.
        Raises ValueError if the page_id doesn't exist in the file yet.
        """
        if page_id < 0 or page_id >= self._page_count():
            raise ValueError(
                f"page_id={page_id} no existe en '{self.file_path}' "
                f"(el archivo tiene {self._page_count()} páginas)"
            )

        with open(self.file_path, "rb") as f:
            f.seek(page_id * self.page_size)
            raw = f.read(self.page_size)

        return Page(page_id=page_id, size=self.page_size, data=raw)

    def write_page(self, page: Page):
        """
        Persists page to its corresponding offset on disk.
        The page must already have been allocated (via allocate_page)
        before it can be written — this method does not extend the file.
        """
        if page.page_id < 0 or page.page_id >= self._page_count():
            raise ValueError(
                f"page_id={page.page_id} no existe en '{self.file_path}'; "
                "debe ser creado con allocate_page() antes de escribirlo"
            )
        if page.size != self.page_size:
            raise ValueError(
                f"El tamaño de la página ({page.size}) no coincide con "
                f"el page_size del archivo ({self.page_size})"
            )

        with open(self.file_path, "r+b") as f:
            f.seek(page.page_id * self.page_size)
            f.write(page.data)

    def __repr__(self):
        return f"FileManager(file_path={self.file_path!r}, pages={self._page_count()})"

    def page_count(self) -> int:
        """Public wrapper: how many pages currently exist in the file."""
        return self._page_count()
