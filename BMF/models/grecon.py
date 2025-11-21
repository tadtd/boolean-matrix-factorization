import numpy as np
from .base import BMFAlgorithm
from ..utils import BMFResult
from typing import List, Set, Tuple, Optional
import time

class GreCon(BMFAlgorithm):
  def __init__(self):
    super().__init__()

  @property
  def name(self) -> str:
    return "GreCon"

  def closure(self, A: np.ndarray, attrs: Set[int]) -> Set[int]:
    """Compute closure attrs'' (common attributes of all objects having attrs)."""
    m, n = A.shape
    if not attrs:
      mask = np.ones(m, dtype=bool)
    else:
      mask = np.all(A[:, list(attrs)] == 1, axis=1)
    if np.any(mask):
      rows = np.nonzero(mask)[0]
      mask_attr = np.all(A[rows, :] == 1, axis=0)
      return set(np.nonzero(mask_attr)[0])
    else:
      return set()

  def is_lectic_greater(self, current: Set[int], candidate: Set[int], attributes: List[int], i: int) -> bool:
    """Check lectic order: current <_lect candidate."""
    for k in range(i):
      if (attributes[k] in candidate) != (attributes[k] in current):
        return False
    return attributes[i] in candidate

  def generate_all_formal_concepts(self, A: np.ndarray) -> List[Tuple[Set[int], Set[int]]]:
    """Generate all formal concepts using the NextClosure algorithm."""
    m, n = A.shape
    attributes = list(range(n))
    concepts = []

    C = set()  # start from empty set
    while True:
      # extent = C', intent = C
      extent_mask = np.all(A[:, list(C)] == 1, axis=1) if C else np.ones(m, dtype=bool)
      extent = set(np.nonzero(extent_mask)[0])
      concepts.append((extent, C))

      found = False
      for i in reversed(range(n)):
        if attributes[i] not in C:
          prefix = {a for a in C if a < attributes[i]} | {attributes[i]}
          D = self.closure(A, prefix)
          if self.is_lectic_greater(C, D, attributes, i):
            C = D
            found = True
            break
      if not found:
        break

    return concepts

  def select_max_cover(
    self,
    universe: Set[Tuple[int, int]],
    concepts_with_pairs: List[Tuple[Tuple[Set[int], Set[int]], Set[Tuple[int, int]]]]
  ) -> Tuple[Optional[int], Optional[Tuple[Set[int], Set[int]]], int]:
    best_idx = None
    best_cover = 0
    best_concept = None
    for idx, ((C, D), cover_pairs) in enumerate(concepts_with_pairs):
      intersection_count = len(cover_pairs & universe)
      if intersection_count > best_cover:
        best_cover = intersection_count
        best_concept = (C, D)
        best_idx = idx
    return best_idx, best_concept, best_cover

  def solve(self, A: np.ndarray) -> BMFResult:
    self._validate_input(A)
    start_time = time.time()
    m, n = A.shape
    A_copy = A.copy()

    # Step 1: Generate all formal concepts using NextClosure
    S_list = self.generate_all_formal_concepts(A_copy)
    print(f"Generated {len(S_list)} formal concepts")

    # Precompute cover pairs for each concept
    concepts_with_pairs = []
    for C, D in S_list:
      cover_pairs = {(i, j) for i in C for j in D if A_copy[i, j] == 1}
      if cover_pairs:
        concepts_with_pairs.append(((C, D), cover_pairs))

    # Step 2: Build universe of (i, j) with A[i, j] = 1
    universe: Set[Tuple[int, int]] = {(i, j) for i in range(m) for j in range(n) if A_copy[i, j] == 1}
    factor_concepts: List[Tuple[Set[int], Set[int]]] = []

    factor_id: int = 0

    # Step 3: GreCon greedy loop
    while universe and concepts_with_pairs:
      idx, concept, cover = self.select_max_cover(universe, concepts_with_pairs)
      if concept is None or cover == 0:
        break

      C_set, D_set = concept
      factor_concepts.append((C_set, D_set))

      print(f"[GreCon] Found factor concept #{factor_id} with |C|={len(C_set)}, |D|={len(D_set)}, gain={cover}")
      factor_id += 1

      for i in C_set:
        for j in D_set:
          universe.discard((i, j))

      if idx is not None:
        concepts_with_pairs.pop(idx)

    run_time = time.time() - start_time
    print(f"[GreCon] factorize runtime: {run_time:.6f} seconds")    
    
    # Step 4: Construct factor matrices
    rank = len(factor_concepts)
    B = np.zeros((m, rank), dtype=int)
    C = np.zeros((rank, n), dtype=int)
    for l, (C_l, D_l) in enumerate(factor_concepts):
      for i in C_l:
        B[i, l] = 1
      for j in D_l:
        C[l, j] = 1

    return BMFResult(A=A, B=B, C=C, time_taken=run_time)