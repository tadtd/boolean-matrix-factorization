import numpy as np
from .base import BMFAlgorithm
from ..utils import BMFResult
from typing import List, Tuple, Set, FrozenSet, Optional

class GreCon(BMFAlgorithm):
  def __init__(self):
    super().__init__()

  @property
  def name(self) -> str:
    return 'GreCon'

  def objects_to_attributes(self, A: np.ndarray, C: Set[int]) -> Set[int]:
    _, n = A.shape
    if not C:
      return set(range(n))
    attributes: Set[int] = set()
    for j in range(n):
      if all(A[i, j] == 1 for i in C):
        attributes.add(j)
    return attributes

  def attributes_to_objects(self, A: np.ndarray, D: Set[int]) -> Set[int]:
    m, _ = A.shape
    if not D:
      return set(range(m))
    objects: Set[int] = set()
    for i in range(m):
      if all(A[i, j] for j in D):
        objects.add(i)
    return objects

  def all_object_concepts(self, A: np.ndarray) -> Set[Tuple[FrozenSet[int], FrozenSet[int]]]:
    m, n = A.shape
    object_concepts: Set[Tuple[FrozenSet[int], FrozenSet[int]]] = set()
    for x in range(m):
      intent = {j for j in range(n) if A[x, j] == 1}
      extent = self.attributes_to_objects(A, intent)
      object_concepts.add((frozenset(extent), frozenset(intent)))
    return object_concepts

  def all_attributes_concept(self, A: np.ndarray) -> Set[Tuple[int, int]]:
    m, n = A.shape
    attribute_concepts: Set[Tuple[FrozenSet[int], FrozenSet[int]]] = set()
    for y in range(n):
      extent = {i for i in range(m) if A[i, y] == 1}
      intent = self.objects_to_attributes(A, extent)
      attribute_concepts.add((frozenset(extent), frozenset(intent)))
    return attribute_concepts

  # Ganter algorithm for generating all formal concepts
  def generate_all_formal_concepts(self, A: np.ndarray) -> List[Tuple[Set[int], Set[int]]]:
    _, n = A.shape
    A_copy = A.copy()
    concepts: List[Tuple[Set[int], Set[int]]] = []

    D: Set[int] = set()  # start with empty intent

    def next_closure(D: Set[int]) -> Optional[Set[int]]:
      """Compute next closure after D using NextClosure algorithm."""
      for j in reversed(range(n)):
        if j not in D:
          E = {k for k in D if k < j} | {j}
          extent = self.attributes_to_objects(A_copy, E)
          closure = self.objects_to_attributes(A_copy, extent)
          if all(k in closure for k in D if k < j):
            return closure
      return None

    while True:
      extent = self.attributes_to_objects(A_copy, D)
      intent = self.objects_to_attributes(A_copy, extent)
      concepts.append((extent, intent))
      D_next = next_closure(D)
      if D_next is None:
        break
      D = D_next

    return concepts

  def solve(self, A: np.ndarray) -> BMFResult:
    self._validate_input(A)
    m, n = A.shape
    A_copy = A.copy()

    S_all = self.generate_all_formal_concepts(A_copy)
    S_list: List[Tuple[Set[int], Set[int]]] = [(set(C), set(D)) for (C, D) in S_all]

    universe: Set[Tuple[int, int]] = {(i, j) for i in range(m) for j in range(n) if A_copy[i, j] == 1}
    obj_concepts = self.all_object_concepts(A=A_copy)
    attr_concepts = self.all_attributes_concept(A=A_copy)
    initial_hashable = obj_concepts & attr_concepts

    factor_concepts: List[Tuple[Set[int], Set[int]]] = []

    if initial_hashable:
      for (ext_frz, int_frz) in initial_hashable:
        C_set = set(ext_frz)
        D_set = set(int_frz)
        factor_concepts.append((C_set, D_set))

        for i in C_set:
          for j in D_set:
            universe.discard((i, j))
        
        S_list = [(C, D) for (C, D) in S_list if (frozenset(C), frozenset(D)) not in initial_hashable]
    
    while universe and S_list:
      best_score = 0
      best_idx: Optional[int] = None
      for idx, (C, D) in enumerate(S_list):
        concept_pairs = {(i, j) for i in C for j in D}
        score = len(concept_pairs & universe)
        if score > best_score:
          best_score = score
          best_idx = idx
      
      if best_idx is None or best_score == 0:
        break

      C_best, D_best = S_list.pop(best_idx)
      factor_concepts.append((C_best, D_best))

      for i in C_best:
        for j in D_best:
          universe.discard((i, j))
    
    rank = len(factor_concepts)
    B = np.zeros((m, rank), dtype=int)
    C = np.zeros((rank, n), dtype=int)
    for l, (C_l, D_l) in enumerate(factor_concepts):
      for i in C_l:
        B[i, l] = 1
      for j in D_l:
        C[l, j] = 1

    return BMFResult(A=A, B=B, C=C)