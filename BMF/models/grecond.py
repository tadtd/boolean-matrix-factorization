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
    for i in range(m):
      if all(A[i, j] == 1 for j in D):
        objects.add(i)
    return objects

  def objects_to_attributes(self, A: np.ndarray, C: Set[int]) -> Set[int]:
    attributes: Set[int] = set()
    m, _ = A.shape
    for j in C:
      if all(A[i, j] == 1 for i in range(m)):
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
      for j in range(n):
        if j not in D:
          D_j = D.union({j})
          C_j = self.attributes_to_objects(A_copy, D_j)
          D_j = self.objects_to_attributes(A_copy, C_j)
          formal_concepts = (C_j, D_j)
          D_oplus_j = formal_concepts.intersection(universe)
          if D_oplus_j and len(D_oplus_j) > V:
            V = len(D_oplus_j)
            D = D_j
      C = self.attributes_to_objects(A_copy, D)
      factor_concepts.append((C, D))
      for (i, j) in list(universe):
        if i in C and j in D:
          universe.remove((i, j))
    
    B = np.zeros((m, len(factor_concepts)), dtype=int)
    C = np.zeros((len(factor_concepts), n), dtype=int)
    for l, (C_l, D_l) in enumerate(factor_concepts):
      for i in C_l:
        B[i, l] = 1
      for j in D_l:
        C[l, j] = 1

    return BMFResult(B, C)