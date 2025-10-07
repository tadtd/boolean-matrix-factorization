import numpy as np
import time
from .base import BMFAlgorithm
from ..utils import BMFResult
from typing import List, Tuple


class Asso(BMFAlgorithm):
  """
  Association-based Boolean Matrix Factorization algorithm.
  
  This algorithm uses association rules to find basis vectors and greedily
  selects factors based on coverage scoring.
  """
  
  def __init__(self, rank: int = 5, tau: float = 0.8, wp: float = 1.0, wn: float = 1.0):
    """
    Initialize Asso algorithm.
    
    Args:
        rank (int): Number of factors/components
        tau (float): Confidence threshold for association rules
        wp (float): Weight for positive matches (true positives)
        wn (float): Weight for negative penalty (false positives)
    """
    super().__init__()
    self.rank = rank
    self.tau = tau
    self.wp = wp
    self.wn = wn
  
  @property
  def name(self) -> str:
    return "Asso"
  
  def _compute_confidence(self, A: np.ndarray, i: int, j: int) -> float:
    """
    Compute confidence: conf(i → j) = |c_i ∧ c_j| / |c_i|
    
    Args:
        A: Input matrix
        i, j: Column indices
        
    Returns:
        Confidence value
    """
    # Count support and intersection
    supp = np.sum(A[:, i])
    if supp == 0:
      return 0.0
        
    inter = np.sum(A[:, i] & A[:, j])
    return float(inter) / float(supp)
  
  def _build_association_matrix(self, A: np.ndarray) -> np.ndarray:
    """
    Build association matrix: entry (i,j) = 1 if conf(i→j) >= tau
    
    Args:
        A: Input matrix
        
    Returns:
        Association matrix
    """
    _, n = A.shape
    assoc = np.zeros((n, n), dtype=int)
    
    for i in range(n):
      for j in range(n):
        if self._compute_confidence(A, i, j) >= self.tau:
          assoc[i, j] = 1
    
    return assoc
  
  def _generate_optimal_s(self, A: np.ndarray, covered: np.ndarray, 
                          basis_vector: np.ndarray) -> np.ndarray:
    """
    Generate s: select rows where candidate improves score
    
    Args:
      A: Input matrix
      covered: Already covered entries
      basis_vector: Current basis vector
        
    Returns:
      Column vector s indicating selected rows
    """
    m, n = A.shape
    s = np.zeros((m, 1), dtype=int)
    
    for i in range(m):
      gain = 0
      for j in range(n):
        if basis_vector[j]:
          if A[i, j]:
            if not covered[i, j]:
              gain += int(self.wp)
          else:
            if not covered[i, j]:
              gain -= int(self.wn)
      
      if gain > 0:
        s[i, 0] = 1
    
    return s
  
  def _compute_cover_score(self, B: np.ndarray, C: np.ndarray, A: np.ndarray) -> float:
    """
    Compute cover score = w+ * TP - w- * FP
    
    Args:
      B, C: Factor matrices
      A: Original matrix
        
    Returns:
      Cover score
    """
    approx = self._boolean_multiply(B, C)
    return self._positive_score(A, approx) - self._negative_penalty(A, approx)

  def _positive_score(self, A: np.ndarray, approx: np.ndarray) -> float:
    """Compute positive score (true positives)"""
    tp = np.sum(A & approx)
    return self.wp * float(tp)
  
  def _negative_penalty(self, A: np.ndarray, approx: np.ndarray) -> float:
    """Compute negative penalty (false positives)"""
    fp = np.sum((~A) & approx)
    return self.wn * float(fp)
  
  def _boolean_multiply(self, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Boolean matrix multiplication: (B * C) > 0"""
    return (np.dot(B, C) > 0).astype(int)
  
  def solve(self, A: np.ndarray) -> BMFResult:
    """
    Factorize boolean matrix A using association-based approach.
    
    Args:
      A (np.ndarray): Input boolean matrix
        
    Returns:
      BMFResult: Factorization result containing B, C matrices and metadata
    """
    self._validate_input(A)
    start_time = time.time()
    
    m, n = A.shape
    k = self.rank
    
    # Initialize B (m × k) and C (k × n)
    B = np.zeros((m, k), dtype=int)
    C = np.zeros((k, n), dtype=int)
    
    # Build association matrix (n × n)
    association_mat = self._build_association_matrix(A)
    
    # Track covered entries
    covered = np.zeros((m, n), dtype=bool)
    
    factor_id = 0

    # Greedy selection of basis vectors
    for l in range(k):
      best_score = float('-inf')
      best_index = -1
      best_basis_vec = None
      best_s = None
      
      # Try each column of association matrix as candidate
      for cand in range(n):
        candidate_vec = association_mat[cand, :].astype(bool)
        
        # Generate s (indicator of rows explained by this basis)
        s = self._generate_optimal_s(A, covered, candidate_vec)
        
        # Build temporary C with row l = candidate_vec
        C_tmp = C.copy()
        C_tmp[l, :] = candidate_vec.astype(int)
        
        # Build temporary B with column l from s
        B_tmp = B.copy()
        B_tmp[:, l] = s.flatten()
        
        # Score this candidate
        score = self._compute_cover_score(B_tmp, C_tmp, A)
        
        if score > best_score:
          best_score = score
          best_index = cand
          best_basis_vec = candidate_vec
          best_s = s
      
      print(f"[Asso] Selected basis vector #{factor_id} (assoc col {best_index}) with score {best_score}")
      factor_id += 1

      # Commit best candidate into B and C
      if best_index >= 0 and best_basis_vec is not None and best_s is not None:
        C[l, :] = best_basis_vec.astype(int)
        B[:, l] = best_s.flatten()
        
        # Update covered entries: covered = covered OR (s * basis_vec^T)
        s_bool = best_s.flatten().astype(bool)
        for i in range(m):
          if s_bool[i]:
            for j in range(n):
              if best_basis_vec[j]:
                covered[i, j] = True
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"[Asso] factorize runtime: {runtime:.6f} seconds")    
   
    metadata = {
      'tau': self.tau,
      'wp': self.wp,
      'wn': self.wn,
    }

    return BMFResult(A=A, B=B, C=C, time_taken=runtime, metadata=metadata)