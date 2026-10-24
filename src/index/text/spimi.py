"""
SPIMI (Single-Pass In-Memory Indexing) indexer.

Builds an inverted index from an iterable of documents without loading the
entire corpus into memory. Accumulates postings in memory blocks, sorts and spills
each block to disk, and deterministically merges all blocks using a Min-Heap.
"""

from __future__ import annotations

import heapq
import os
import pickle
import tempfile
import time
from collections import defaultdict
from typing import Iterable

from index.text.inverted_index import InvertedIndex
from index.text.text_processor import TextProcessor


class SPIMIStats:
    """Statistics and telemetry of the SPIMI indexing process."""

    def __init__(self):
        self.blocks_created: int = 0
        self.total_tokens_processed: int = 0
        self.ram_time_sec: float = 0.0
        self.spill_time_sec: float = 0.0
        self.merge_time_sec: float = 0.0
        self.total_time_sec: float = 0.0
        self.block_sizes: list[int] = []

    def to_dict(self) -> dict[str, float | int | list[int]]:
        return {
            "blocks_created": self.blocks_created,
            "total_tokens_processed": self.total_tokens_processed,
            "ram_time_sec": round(self.ram_time_sec, 6),
            "spill_time_sec": round(self.spill_time_sec, 6),
            "merge_time_sec": round(self.merge_time_sec, 6),
            "total_time_sec": round(self.total_time_sec, 6),
            "block_sizes": self.block_sizes,
        }


class SPIMIIndexer:
    """
    Constructs an InvertedIndex using the Single-Pass In-Memory Indexing (SPIMI) algorithm.
    """

    def __init__(
        self,
        processor: TextProcessor | None = None,
        max_records_per_block: int = 10000,
        temp_dir: str | None = None,
    ):
        if max_records_per_block <= 0:
            raise ValueError("max_records_per_block debe ser mayor que 0")

        self.processor = processor or TextProcessor()
        self.max_records_per_block = max_records_per_block
        self._temp_dir_override = temp_dir
        self.stats = SPIMIStats()
        self.last_block_paths: list[str] = []
        self._retained_temp_dir: tempfile.TemporaryDirectory | None = None

    def index_corpus(
        self,
        documents: Iterable[tuple[int, str]],
        keep_blocks: bool = False,
    ) -> InvertedIndex:
        """
        Indexes a corpus of (doc_id, text) tuples.
        The input documents can be an iterator or streaming generator.
        """
        start_total = time.perf_counter()
        self.stats = SPIMIStats()
        self.last_block_paths = []

        temp_dir_obj = None
        if self._temp_dir_override is None:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="spimi-blocks-")
            work_dir = temp_dir_obj.name
        else:
            work_dir = self._temp_dir_override
            os.makedirs(work_dir, exist_ok=True)

        block_paths: list[str] = []
        doc_lengths: dict[int, int] = {}
        total_docs = 0

        # In-memory accumulator: term -> { doc_id: count }
        current_block: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        records_in_current_block = 0

        try:
            for doc_id, text in documents:
                ram_start = time.perf_counter()
                total_docs += 1
                tokens = self.processor.process(text)
                doc_lengths[doc_id] = len(tokens)
                self.stats.total_tokens_processed += len(tokens)

                # Accumulate term frequencies for this document
                for token in tokens:
                    current_block[token][doc_id] += 1

                records_in_current_block += 1
                self.stats.ram_time_sec += time.perf_counter() - ram_start

                # Check if block limit has been reached
                if records_in_current_block >= self.max_records_per_block:
                    t_spill = time.perf_counter()
                    block_path = self._spill_block(
                        current_block, work_dir, len(block_paths)
                    )
                    self.stats.spill_time_sec += time.perf_counter() - t_spill
                    self.stats.blocks_created += 1
                    self.stats.block_sizes.append(records_in_current_block)

                    block_paths.append(block_path)
                    current_block = defaultdict(lambda: defaultdict(int))
                    records_in_current_block = 0

            # Spill any remaining records in the last block
            if records_in_current_block:
                t_spill = time.perf_counter()
                block_path = self._spill_block(
                    current_block, work_dir, len(block_paths)
                )
                self.stats.spill_time_sec += time.perf_counter() - t_spill
                self.stats.blocks_created += 1
                self.stats.block_sizes.append(records_in_current_block)

                block_paths.append(block_path)
                current_block.clear()

            # Handle empty corpus
            if not block_paths:
                self.stats.total_time_sec = time.perf_counter() - start_total
                return InvertedIndex(index={}, doc_lengths={}, total_docs=0)

            # Perform K-way multi-way merge on the spilled blocks
            t_merge = time.perf_counter()
            final_index = self._merge_blocks(block_paths)
            self.stats.merge_time_sec += time.perf_counter() - t_merge

            self.stats.total_time_sec = time.perf_counter() - start_total
            self.last_block_paths = list(block_paths)

            return InvertedIndex(
                index=final_index,
                doc_lengths=doc_lengths,
                total_docs=total_docs,
            )

        finally:
            if not keep_blocks:
                # Clean up block files
                for path in block_paths:
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except OSError:
                        pass
                if temp_dir_obj is not None:
                    temp_dir_obj.cleanup()
            else:
                self._retained_temp_dir = temp_dir_obj

    def _spill_block(
        self,
        block_dict: dict[str, dict[int, int]],
        work_dir: str,
        block_index: int,
    ) -> str:
        """
        Sorts the terms of the block lexicographically, orders each posting list
        by doc_id, and writes to a temporary block file on disk.
        """
        sorted_terms = sorted(block_dict.keys())
        block_file = os.path.join(work_dir, f"block-{block_index}.bin")

        with open(block_file, "wb") as handle:
            for term in sorted_terms:
                # Postings sorted deterministically by doc_id
                postings = sorted(block_dict[term].items(), key=lambda x: x[0])
                pickle.dump((term, postings), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return block_file

    def _merge_blocks(
        self, block_paths: list[str]
    ) -> dict[str, list[tuple[int, int]]]:
        """
        Merges multiple sorted block files using a Min-Heap.
        Returns a consolidated dictionary { term: [(doc_id, tf), ...] }.
        """
        handles = [open(path, "rb") for path in block_paths]
        # Min-heap stores: (term, block_idx, postings)
        heap: list[tuple[str, int, list[tuple[int, int]]]] = []

        try:
            # Prime the heap with the first entry from each block
            for idx, handle in enumerate(handles):
                try:
                    term, postings = pickle.load(handle)
                    heapq.heappush(heap, (term, idx, postings))
                except EOFError:
                    continue

            final_index: dict[str, list[tuple[int, int]]] = {}

            while heap:
                current_term, block_idx, postings = heapq.heappop(heap)
                merged_postings_map: dict[int, int] = dict(postings)

                # Refill from the block that just yielded
                try:
                    next_term, next_postings = pickle.load(handles[block_idx])
                    heapq.heappush(heap, (next_term, block_idx, next_postings))
                except EOFError:
                    pass

                # Merge duplicates: other blocks with the EXACT same term
                while heap and heap[0][0] == current_term:
                    _, other_idx, other_postings = heapq.heappop(heap)
                    for d_id, tf in other_postings:
                        merged_postings_map[d_id] = (
                            merged_postings_map.get(d_id, 0) + tf
                        )

                    try:
                        next_term, next_postings = pickle.load(handles[other_idx])
                        heapq.heappush(heap, (next_term, other_idx, next_postings))
                    except EOFError:
                        pass

                # Final sorted postings for this term
                final_postings = sorted(
                    merged_postings_map.items(), key=lambda x: x[0]
                )
                final_index[current_term] = final_postings

            return final_index

        finally:
            for handle in handles:
                handle.close()
