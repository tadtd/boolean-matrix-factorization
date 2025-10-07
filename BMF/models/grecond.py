import numpy as np
from .base import BMFAlgorithm
from ..utils import BMFResult
from typing import List, Tuple, Set
import time

class GreConD(BMFAlgorithm):
  def __init__(self):
    super().__init__()

  @property
  def name(self) -> str:
    return 'GreConD'

  def attributes_to_objects(self, A: np.ndarray, D: Set[int]) -> Set[int]:
    """Convert set of attributes to corresponding objects (rows)."""
    objects: Set[int] = set()
    m, _ = A.shape
    if not D:
      return set(range(m))
    for i in range(m):
      if all(A[i, j] == 1 for j in D):
        objects.add(i)
    return objects

  def objects_to_attributes(self, A: np.ndarray, C: Set[int]) -> Set[int]:
    """Convert set of objects to corresponding attributes (columns)."""
    attributes: Set[int] = set()
    _, n = A.shape
    if not C:
      return set(range(n))
    for j in range(n):
      if all(A[i, j] == 1 for i in C):
        attributes.add(j)
    return attributes

  def solve(self, A: np.ndarray) -> BMFResult:
    """GreConD algorithm for Boolean Matrix Factorization."""
    self._validate_input(A)

    start_time = time.time()

    m, n = A.shape
    A_copy = A.copy()

    universe: Set[Tuple[int, int]] = {(i, j) for i in range(m) for j in range(n) if A[i, j] == 1}
    factor_concepts: List[Tuple[Set[int], Set[int]]] = []

    factor_id: int = 0

    while universe:
      D: Set[int] = set()
      V: int = 0
      while True:
        best_j, best_gain, best_D = None, 0, D
        for j in range(n):
          if j not in D:
            D_candidate = D | {j}
            C_candidate = self.attributes_to_objects(A_copy, D_candidate)
            D_candidate = self.objects_to_attributes(A_copy, C_candidate)
            concept_pairs = {(i, jj) for i in C_candidate for jj in D_candidate}
            gain = len(concept_pairs & universe)
            if gain > best_gain:
              best_gain = gain
              best_j = j
              best_D = D_candidate
        
        if best_j is None or best_gain <= V:
          break

        V = best_gain
        D = best_D
      
      C = self.attributes_to_objects(A_copy, D)
      factor_concepts.append((C, D))

      print(f"[GreConD] Found factor concept #{factor_id} with |C|={len(C)}, |D|={len(D)}, gain={V}")
      factor_id += 1
      
      for i in C:
        for j in D:
          universe.discard((i, j))
    
    run_time = time.time() - start_time
    print(f"[GreConD] factorize runtime: {run_time:.6f} seconds")

    rank = len(factor_concepts)
    B = np.zeros((m, rank), dtype=int)
    C = np.zeros((rank, n), dtype=int)
    for l, (C_l, D_l) in enumerate(factor_concepts):
      for i in C_l:
        B[i, l] = 1
      for j in D_l:
        C[l, j] = 1

    return BMFResult(A=A, B=B, C=C, time_taken=run_time)