import numpy as np
from base import BMFAlgorithm
from BMF.utils import utils
import time

class AssoAlgorithm(BMFAlgorithm):
  '''
  Implementation of the ASSO algorithm for Boolean matrix factorization
  '''

  def __init__(self, k: int,
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
    super().__init__(k, tau=tau, wp=wp, wn=wn)
    self.tau = tau
    self.wp = wp
    self.wn = wn
  
  @property
  def name(self) -> str:
    return 'ASSO'

  def _compute_cover_score(self, B: np.ndarray, S: np.ndarray, C: np.ndarray) -> float:
    '''
    Compute the cover score for current factorization
    Args:
      B, S: Factor matrix
      C: Original matrix
    Returns:
      Cover score
    '''
    appr = utils.boolean_product(S, B)
    positve_score = self.wp * np.sum((C==1)&(appr==1))
    negative_penalty = self.wp * np.sum((C==0)&(appr==1))
    return positve_score - negative_penalty
  
  def _build_candidate_matrix(self, C:np.ndarray) -> np.ndarray:
    '''
    Build candidate matrix A based on column similarity.
    Args:
      C: Input matrix
    Returns:
      Candidate matrix A
    '''
    n, m = C.shape
    A = np.zeros((n, m), dtype=int)

    for i in range(n):
      for j in range(m):
        similarity = utils.cosine_similarity(C[:, i], C[:, j])
        if similarity >= self.tau:
          A[i, j] = 1
    
    return A
  
  def fit(self, C: np.ndarray) -> np.ndarray:
    '''
    Fit Asso algorithm to the input matrix
    Args:
      C: Input matrix (n [by] m)
    Returns:

    '''
    start_time = time.time()

    # Validate input
    self._validate_input(C)

    n, m = C.shape

    # Initialize result matrices
    B = np.zeros((self.rank, m), dtype=int)
    S = np.zeros((n, self.rank), dtype=int)

    # Build candidate matrix
    A = self._build_candidate_matrix(C)
    