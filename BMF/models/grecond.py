import numpy as np
from .base import BMFAlgorithm
from ..utils import BMFResult
from typing import List, Tuple, Set

class GreConD(BMFAlgorithm):
  def __init__(self):
    super().__init__()

  @property
  def name(self) -> str:
    return 'GreConD'

  def attributes_to_objects(self, A: np.ndarray, D: Set[int]) -> Set[int]:
    objects: Set[int] = set()
    m, _ = A.shape
    if not D:
      return set(range(m))
    for i in range(m):
      if all(A[i, j] == 1 for j in D):
        objects.add(i)
    return objects

  def objects_to_attributes(self, A: np.ndarray, C: Set[int]) -> Set[int]:
    attributes: Set[int] = set()
    _, n = A.shape
    if not C:
      return set(range(n))
    for j in range(n):
      if all(A[i, j] == 1 for i in C):
        attributes.add(j)
    return attributes

  def solve(self, A: np.ndarray) -> BMFResult:
    self._validate_input(A)
    m, n = A.shape
    A_copy = A.copy()

    universe: Set[Tuple[int, int]] = {(i, j) for i in range(m) for j in range(n) if A[i, j] == 1}
    factor_concepts: List[Tuple[Set[int], Set[int]]] = []
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
        if best_j is None:
          break
        if best_gain > V:
          V = best_gain
          D = best_D
        else:
          break
      
      C = self.attributes_to_objects(A_copy, D)
      factor_concepts.append((C, D))
      for (i, j) in {(i, j) for i in C for j in D}:
        universe.discard((i, j))
    
    B = np.zeros((m, len(factor_concepts)), dtype=int)
    C = np.zeros((len(factor_concepts), n), dtype=int)
    for l, (C_l, D_l) in enumerate(factor_concepts):
      for i in C_l:
        B[i, l] = 1
      for j in D_l:
        C[l, j] = 1

    return BMFResult(A=A, B=B, C=C)