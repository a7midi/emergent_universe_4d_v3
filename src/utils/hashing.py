"""
utils/hashing.py  — v2
──────────────────────────────────────────────────────────────────────────────
A *deterministic* and *order‑independent* fingerprint for the tag pattern that
appears in a single detector grid cell.

Changes w.r.t. v1
─────────────────
1. **Order independence** is now real – we sort the tag array before hashing,
   so node‑index ordering and Python’s set iteration order can never change
   the hash.
2. **Collisions are harder** – we mix the tag count and dtype into the hash
   pre‑image.
3. **Speed‑up** –   SHA‑256 is still cryptographically strong, but we feed it
   one contiguous `memoryview` instead of building temporary Python objects.
"""

from __future__ import annotations
import hashlib
import numpy as np


def hash_cell_state(cell_tags: np.ndarray) -> str | None:
    """
    Compute an order‑independent SHA‑256 digest of ``cell_tags``.

    Parameters
    ----------
    cell_tags :
        1‑D ``np.ndarray`` of *integers* (dtype is arbitrary) that stores the
        current tag of **every** node located in a single spatial grid cell.

    Returns
    -------
    str | None
        • 64‑character hexadecimal SHA‑256 digest  
        • ``None`` if the array is empty (caller can skip the cell)

    Notes
    -----
    *   Sorting is **O(N log N)** but the arrays are tiny (≤ R·grid_size²).
    *   We *do* preserve multiplicity – i.e. `[1,1,2]``≠``[1,2,2]`.
    """
    # Fast‑path for empty cells
    if cell_tags.size == 0:
        return None

    # 1. Order‑independent canonical representation
    canonical = np.sort(cell_tags, axis=None)

    # 2. Feed a single contiguous memoryview into SHA‑256
    hasher = hashlib.sha256()
    hasher.update(canonical.view(np.uint8))        # raw little‑endian bytes
    hasher.update(len(canonical).to_bytes(4, "little"))
    hasher.update(str(canonical.dtype).encode())

    return hasher.hexdigest()


# ──────────────────────────────────────────────────────────────────────────
# Self‑test (python -m src.utils.hashing)
if __name__ == "__main__":
    a  = np.array([5, 8, 0, 12, 3], dtype=np.int32)
    a2 = a.copy()
    b  = np.array([5, 8,  1, 12, 3], dtype=np.int32)   # one tag differs
    c  = a[::-1]                                       # same multiset, diff. order

    assert hash_cell_state(a)  == hash_cell_state(a2), "identical arrays must hash equally"
    assert hash_cell_state(a)  != hash_cell_state(b),  "different arrays must hash differently"
    assert hash_cell_state(a)  == hash_cell_state(c),  "order must NOT change the hash"
    print("✓ hashing.py self‑test passed")
