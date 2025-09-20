import numpy as np
from .base import BMFAlgorithm
from ..utils import utils, BMFResult
import time

class Greedy(BMFAlgorithm):
  '''
  Greedy BMF algorithm implementation.
  '''
  def __init__(self, rank: int):
    '''
    Initialize the GreedyBMF model.
    Args:
      rank: The rank (number of latent factors) for the BMF.
    '''
    super().__init__()
    self.rank = rank
    self.B = None
    self.C = None
  
  @property
  def name(self) -> str:
    return 'Greedy'
  
  def solve(self, X: np.ndarray) -> BMFResult:
    '''
    Fit the model to the data using greedy approach.
    Args:
      X: Input data matrix.
    Returns:
      BMFResult: Fitted model results.
    '''
    start_time = time.time()
    
    # Validate input
    self._validate_input(X)
    
    n, m = X.shape
    
    # Initialize factor matrices
    self.B = np.zeros((n, self.rank), dtype=int)
    self.C = np.zeros((self.rank, m), dtype=int)
    
    # Greedy selection
    remaining = X.copy()
    
    for k in range(self.rank):
      best_error = float('inf')
      best_b = None
      best_c = None
      
      # Try all possible rank-1 factorizations
      for i in range(n):
        for j in range(m):
          if remaining[i, j] == 1:
            # Create rank-1 factors
            b_candidate = np.zeros(n, dtype=int)
            c_candidate = np.zeros(m, dtype=int)
            b_candidate[i] = 1
            c_candidate[j] = 1
            
            # Extend to cover more 1s
            for ii in range(n):
              for jj in range(m):
                if remaining[ii, jj] == 1 and (ii == i or jj == j):
                  b_candidate[ii] = 1
                  c_candidate[jj] = 1
            
            # Compute reconstruction error for this rank-1 factor
            reconstruction = np.outer(b_candidate, c_candidate)
            error = np.sum(np.abs(remaining - reconstruction))
            
            if error < best_error:
              best_error = error
              best_b = b_candidate
              best_c = c_candidate
      
      if best_b is not None:
        self.B[:, k] = best_b
        self.C[k, :] = best_c
        # Update remaining matrix
        rank1_approx = np.outer(best_b, best_c)
        remaining = np.logical_and(remaining, ~rank1_approx).astype(int)
    
    # Compute final reconstruction and error
    reconstruction = utils.boolean_product(self.B, self.C)
    error = np.sum(np.abs(X - reconstruction))
    convergence_time = time.time() - start_time
    converged = error == 0
    
    # Create result object
    result = BMFResult(B=self.B, C=self.C, original_matrix=X, 
                       error=error, iterations=self.rank,
                       convergence_time=convergence_time, 
                       converged=converged)
    return result