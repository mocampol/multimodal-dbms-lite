"""
Physical page representation.

Classes:
    Page: A fixed-size block of raw bytes, the minimum unit of I/O
          between disk and the Buffer Manager.
"""

PAGE_SIZE = 4096  # bytes. Single source of truth, imported by every
                   # module that needs to agree on the page size
                   # (file_manager, buffer_manager, heap_file's slotted page).


class Page:
    """
    Represents a single fixed-size page of raw bytes.

    A Page knows nothing about records, schemas, or SQL — it is purely
    a byte-addressable block, identical in spirit to a disk block.
    Interpreting its contents (e.g. as a Slotted Page) is the
    responsibility of higher-level modules like heap_file.

    Attributes:
        page_id (int): Logical identifier of this page within its file.
        size (int): Size of the page in bytes (defaults to PAGE_SIZE).
        data (bytearray): The raw bytes of the page, mutable in place.
    """

    def __init__(self, page_id: int, size: int = PAGE_SIZE, data: bytes = None):
        self.page_id = page_id
        self.size = size

        if data is not None:
            if len(data) != size:
                raise ValueError(
                    f"El buffer recibido ({len(data)} bytes) no coincide "
                    f"con el tamaño de página declarado ({size} bytes)"
                )
            self.data = bytearray(data)
        else:
            self.data = bytearray(size)  # zero-filled by default

    def read_bytes(self, offset: int, length: int) -> bytes:
        """
        Returns `length` bytes starting at `offset` within this page.
        Raises ValueError if the requested range is out of bounds.
        """
        if offset < 0 or offset + length > self.size:
            raise ValueError(
                f"Lectura fuera de rango: offset={offset}, length={length}, "
                f"page_size={self.size}"
            )
        return bytes(self.data[offset:offset + length])

    def write_bytes(self, offset: int, content: bytes):
        """
        Writes `content` into this page starting at `offset`.
        Raises ValueError if the write would exceed the page boundary.
        """
        if offset < 0 or offset + len(content) > self.size:
            raise ValueError(
                f"Escritura fuera de rango: offset={offset}, "
                f"len(content)={len(content)}, page_size={self.size}"
            )
        self.data[offset:offset + len(content)] = content

    def clear(self):
        """
        Resets the page's contents to all zero bytes, keeping its page_id.
        Useful when a freshly allocated page is handed to a caller.
        """
        self.data = bytearray(self.size)

    def __repr__(self):
        return f"Page(page_id={self.page_id}, size={self.size})"
