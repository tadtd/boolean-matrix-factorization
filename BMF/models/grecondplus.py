from .base import BMFAlgorithm
from ..utils import BMFResult
import numpy as np
from typing import Set, Tuple
import time

class GreConDPlus(BMFAlgorithm):
  def __init__(self, w: int = 1):
    super().__init__()
    self.w = w
  
  @property
  def name(self) -> str:
    return "GreConDPlus"
  
  def attributes_to_objects(self, A: np.ndarray, attributes: Set[int]) -> Set[int]:
    """Convert set of attributes to corresponding objects (rows)."""
    m, _ = A.shape
    if not attributes:
      return set(range(m))
    
    objects = set()
    for i in range(m):
      if all(A[i, j] == 1 for j in attributes):
        objects.add(i)
    return objects
  
  def objects_to_attributes(self, A: np.ndarray, objects: Set[int]) -> Set[int]:
    """Convert set of objects to corresponding attributes (columns)."""
    _, n = A.shape
    if not objects:
      return set(range(n))
    
    attributes = set()
    for j in range(n):
      if all(A[i, j] == 1 for i in objects):
        attributes.add(j)
    return attributes

  def gain(self, C: Set[int], F: Set[int], candidate_attrs: Set[int],
           U: Set[Tuple[int, int]], O: Set[Tuple[int, int]],
           w: int) -> int:
    """
    Compute the gain of adding candidate attributes.
    """
    F_j = F | candidate_attrs

    # candidate = C × D_F_j (cartesian product)
    candidate = {(i, j) for i in C for j in F_j}
    
    # cover = candidate ∩ U (true positives)
    cover = candidate & U
    
    # undercover = candidate ∩ O (false positives)
    undercover = candidate & O
    
    score = len(cover) - w * len(undercover)
    return score
  
  def expansion(self, A: np.ndarray, concept: Tuple[Set[int], Set[int]], 
                w: int, U: Set[Tuple[int, int]]) -> Tuple[Set[int], Set[int]]:
    """
    Expand the concept by adding attributes.
    Args:
      A: Input matrix
      concept: Current concept (C, D)
      w: Weight for false positives
      U: Universe of uncovered positive entries
    Returns:
      E: Extra objects added
      F: Extra attributes added
    """
    C, D = concept
    m, n = A.shape
    
    # O = set of zeros in A
    O = {(i, j) for i in range(m) for j in range(n) if A[i, j] == 0}
    
    E = set()  # extra objects (initially empty)
    F = set()  # extra attributes (initially empty)
    
    changed = True
    while changed:
      changed = False
      best_attr = -1
      best_gain = 0
      
      # Loop over j ∉ D ∪ F
      for j in range(n):
        if j in D or j in F:
          continue
        
        # Compute gain(j)
        g = self.gain(C, F, {j}, U, O, w)
        
        if g > best_gain:
          best_gain = g
          best_attr = j
      
      # If best_attr found and gain > 0 → add to F
      if best_attr != -1 and best_gain > 0:
        F.add(best_attr)
        changed = True
    
    return E, F
  
  def solve(self, A: np.ndarray) -> BMFResult:
    """Solve Boolean Matrix Factorization using GreConDPlus algorithm."""

    self._validate_input(A)
    start_time = time.time()
    m, n = A.shape
    
    factor_concepts = set()
    universe = {(i, j) for i in range(m) for j in range(n) if A[i, j] == 1}
    
    factor_id: int = 0

    while universe:
      best_gain = 0
      D = set()
      
      while True:
        gain = 0
        D_j = set()
        
        for j in range(n):
          if j not in D:
            D_candidate = D | {j}
            C_candidate = self.attributes_to_objects(A, D_candidate)
            D_candidate = self.objects_to_attributes(A, C_candidate)
            
            concept_pairs = {(i, jj) for i in C_candidate for jj in D_candidate}
            result = concept_pairs & universe
            
            if len(result) > gain:
              gain = len(result)
              D_j = D_candidate
        
        if gain > best_gain:
          best_gain = gain
          D = D_j
        else:
          break
      
      C = self.attributes_to_objects(A, D)
      E, F = self.expansion(A, (C, D), self.w, universe)
      
      object_factor = C | E
      attribute_factor = D | F
      
      factor_concepts.add((frozenset(object_factor), frozenset(attribute_factor)))
      print(f"[GreConDPlus] Found factor {factor_id}: |C|={len(object_factor)}, |D|={len(attribute_factor)}, gain={best_gain}")
      factor_id += 1
      
      # Update U := U − (C ∪ E) × (D ∪ F)
      to_remove = {(i, j) for i in object_factor for j in attribute_factor}
      universe -= to_remove
      
      # Redundancy removal and pruning procedure
      factors_to_remove = []
      factors_to_update = {}
      
      for A_set, B_set in factor_concepts:
        # Check if entire factor (A, B) is redundant
        factor_redundant = True
        A_cross_B = {(i, j) for i in A_set for j in B_set}
        
        # For each (i,j) ∈ A × B with I_ij = 1
        for (i, j) in A_cross_B:
          if A[i, j] == 1:
            # Check if there exists (G, H) ∈ F - {A, B} with (i, j) ∈ G × H
            covered_by_other = False
            for (G, H) in factor_concepts:
              if (G, H) != (A_set, B_set):
                if i in G and j in H:
                  covered_by_other = True
                  break
            
            if not covered_by_other:
              factor_redundant = False
              break
        
        if factor_redundant:
          # Remove entire factor (A, B) from F
          factors_to_remove.append((A_set, B_set))
          print(f"[GreConDPlus] Removing redundant factor: |A|={len(A_set)}, |B|={len(B_set)}")
        else:
          # Factor is not entirely redundant, check individual attributes
          nucleus_B = self.objects_to_attributes(A, A_set)  # nucleus(B) = A'
          attributes_to_remove = set()
          
          # For each j ∈ B - nucleus(B)
          for j in B_set - nucleus_B:
            attribute_redundant = True
            
            # For each (i, j) ∈ A × B with I_ij = 1
            for i in A_set:
              if A[i, j] == 1:
                # Check if there exists (G, H) ∈ F - {A, B} with (i, j) ∈ G × H
                covered_by_other = False
                for (G, H) in factor_concepts:
                  if (G, H) != (A_set, B_set):
                    if i in G and j in H:
                      covered_by_other = True
                      break
                
                if not covered_by_other:
                  attribute_redundant = False
                  break
            
            if attribute_redundant:
              # Remove j from B
              attributes_to_remove.add(j)
          
          if attributes_to_remove:
            new_B = B_set - attributes_to_remove
            factors_to_update[(A_set, B_set)] = (A_set, new_B)
            print(f"[GreConDPlus] Pruning {len(attributes_to_remove)} attributes from factor: {attributes_to_remove}")
      
      # Apply removals and updates
      for factor in factors_to_remove:
        factor_concepts.discard(factor)
      
      for old_factor, new_factor in factors_to_update.items():
        factor_concepts.discard(old_factor)
        if len(new_factor[1]) > 0:  # Only add if B is not empty
          factor_concepts.add(new_factor)

    # Construct factor matrices
    k = len(factor_concepts)
    B = np.zeros((m, k), dtype=int)
    C = np.zeros((k, n), dtype=int)
    
    for idx, (C_k, D_k) in enumerate(factor_concepts):
      for i in C_k:
        B[i, idx] = 1
      for j in D_k:
        C[idx, j] = 1
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"[GreConDPlus] factorize runtime: {runtime:.6f} seconds")
    
    metadata = {
      'w': self.w,
    }
    
    return BMFResult(A=A, B=B, C=C, time_taken=runtime, metadata=metadata)
