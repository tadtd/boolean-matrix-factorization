import numpy as np
from .base import BMFAlgorithm
from ..utils import utils, BMFResult
import time
from typing import Tuple

class Asso(BMFAlgorithm):
  '''
  Implementation of the ASSO algorithm for Boolean matrix factorization
  '''

  def __init__(self, 
               rank: int,
               tau: float,
               wp: float = 1.0, 
               wn: float = 1.0
  ):
    '''
    Initialize ASSO algorithm
      Args:
      k: Target rank for factorization
      tau: Similarity threshold for candidate selection
      wp: Weight for positive matches
      wn: Weight for negative matches (penalty for false positives)
    '''
    super().__init__()
    self.rank = rank
    self.tau = tau
    self.wp = wp
    self.wn = wn

  @property
  def name(self) -> str:
    return 'Asso'

  def _compute_cover_score(self, B: np.ndarray, S: np.ndarray, C: np.ndarray) -> float:
    '''
    Compute the cover score for current factorization
    Args:
      B, S: Factor matrices
      C: Original matrix
    Returns:
      Cover score
    '''
    appr = utils.boolean_product(S, B)
    positive_score = self.wp * np.sum((C == 1) & (appr == 1))
    negative_penalty = self.wn * np.sum((C == 0) & (appr == 1))
    return positive_score - negative_penalty
  
  def _build_candidate_matrix(self, C: np.ndarray) -> np.ndarray:
    '''
    Build candidate matrix A based on column similarity.
    Args:
      C: Input matrix
    Returns:
      Candidate matrix A (same shape as C)
    '''
    n, m = C.shape
    A = np.zeros((n, m), dtype=int)

    for j in range(m):
      for k in range(m):
        similarity = utils.cosine_similarity(C[:, j], C[:, k])
        if similarity >= self.tau:
          A[j, k] = 1
    
    return A
  
  def solve(self, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Fit ASSO algorithm to the input matrix
    Args:
      C: Input matrix (n x m)
    Returns:
      BMFResult object containing the factorization C = S \circ B
    '''

    k = self.rank

    # Validate input
    self._validate_input(C)

    n, m = C.shape

    # Initialize result matrices - S is n x k, B is k x m
    S = np.zeros((n, k), dtype=int)
    B = np.zeros((k, m), dtype=int)

    # Build candidate matrix
    A = self._build_candidate_matrix(C)
    
    # Track which columns have been used
    used_columns = set()
    
    for i in range(k):
      # Select the best candidate column based on cover score
      best_col = -1
      best_score = -np.inf
      
      for j in range(m):
        if j in used_columns or A[:, j].sum() == 0:
          continue
          
        # Create temporary matrices to test this candidate
        S_temp = S.copy()
        B_temp = B.copy()
        S_temp[:, i] = A[:, j]
        B_temp[i, :] = C[:, j]  # Use original column, not candidate
        
        score = self._compute_cover_score(B_temp, S_temp, C)
        if score > best_score:
          best_score = score
          best_col = j
      
      if best_col != -1:
        S[:, i] = A[:, best_col]
        B[i, :] = C[:, best_col]  # Use original column
        used_columns.add(best_col)
    
    # Create result object
    result = BMFResult(B=S, C=B,
                       metadata={'tau': self.tau,
                                 'wp': self.wp,
                                 'wn': self.wn})
    return result