"""Independent packed-byte oracle for selected-row staging, without Torch."""
from __future__ import annotations

from dataclasses import dataclass

from contracts import PAGE_SIZE, PACKED_ROW_BYTES, SCALE_ROW_BYTES, ContractError


@dataclass(frozen=True)
class ReferenceStage:
    remapped_indices: tuple[tuple[int, ...], ...]
    packed_rows: dict[int, bytes]
    scale_rows: dict[int, bytes]
    logical_to_hot: dict[int, int]


def stage_reference(indices, *, source_pages, physical_packed_rows,
                    physical_scale_rows, slot_generations,
                    expected_generations, row_visible_limits, max_slots=16640):
    rows = len(indices)
    width = len(indices[0]) if rows else 0
    if (not 1 <= rows <= 8 or not 0 < width <= 2080 or width % 32 or
            any(len(row) != width for row in indices) or rows * width > max_slots or
            len(row_visible_limits) != rows):
        raise ContractError("invalid bounded decode shape")
    if len(source_pages) != len(expected_generations):
        raise ContractError("mapping/generation layout mismatch")
    mapping, packed, scales, remap = {}, {}, {}, []
    for query_row, selections in enumerate(indices):
        out = []
        for column, logical in enumerate(selections):
            if logical == -1:
                out.append(-1)
                continue
            if logical < 0 or logical >= row_visible_limits[query_row]:
                raise ContractError("invalid or future selection")
            page, offset = divmod(logical, PAGE_SIZE)
            if page >= len(source_pages):
                raise ContractError("selection exceeds logical mapping")
            physical_page = source_pages[page]
            if not 0 <= physical_page < len(slot_generations):
                raise ContractError("unavailable physical source")
            if slot_generations[physical_page] != expected_generations[page]:
                raise ContractError("stale physical source generation")
            physical_row = physical_page * PAGE_SIZE + offset
            if logical not in mapping:
                # GPU insertion order may differ; flat winner slots make no
                # dense compaction or output shape readback necessary.
                slot = query_row * width + column
                pq, ps = physical_packed_rows[physical_row], physical_scale_rows[physical_row]
                if len(pq) != PACKED_ROW_BYTES or len(ps) != SCALE_ROW_BYTES:
                    raise ContractError("wrong packed row geometry")
                mapping[logical] = slot
                packed[slot], scales[slot] = bytes(pq), bytes(ps)
            out.append(mapping[logical])
        remap.append(tuple(out))
    return ReferenceStage(tuple(remap), packed, scales, mapping)
