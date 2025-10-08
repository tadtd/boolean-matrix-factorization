from .base import BMFAlgorithm
from ..utils import BMFResult, utils
import numpy as np
from typing import Optional, List
import time

class Panda(BMFAlgorithm):
  def __init__(self, k: Optional[int] = None, tol: float = 0, w_model: float = 1, 
               w_fp: float = 1, w_fn: float = 1, init_method: str = 'frequency', 
               exact_decomp: bool = False):
    super().__init__()
    self.k = k
    self.tol = tol
    self.w_model = w_model
    self.w_fp = w_fp
    self.w_fn = w_fn
    self.init_method = init_method
    self.exact_decomp = exact_decomp
    
    # Validate parameters
    if self.init_method not in ['frequency', 'couples-frequency', 'correlation']:
      raise ValueError("init_method must be 'frequency', 'couples-frequency', or 'correlation'")
    
    if self.exact_decomp:
      self.w_model = 0
      self.init_method = 'frequency'
  
  @property
  def name(self) -> str:
    return "Panda"
  
  def description_length(self, A: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
    """
    Calculate description length for pattern evaluation.
    
    DL = w_model * (|U| + |V|) + w_fp * FP + w_fn * FN
    """
    # Get prediction and residual
    prediction = utils.boolean_product(U, V)
    
    # Count false positives and false negatives
    fp = np.sum((prediction == 1) & (A == 0))
    fn = np.sum((prediction == 0) & (A == 1))
    
    # Model complexity (number of 1s in factors)
    model_cost = np.sum(U) + np.sum(V)
    
    return self.w_model * model_cost + self.w_fp * fp + self.w_fn * fn
  
  def get_residual(self, A: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Calculate residual matrix."""
    prediction = utils.boolean_product(U, V)
    return A - prediction
  
  def sort_items(self, residual: np.ndarray, extension_list: List[int], 
                 T: np.ndarray, method: str) -> List[int]:
    """
    Sort items in extension list by descending scores.
    
    Parameters:
    -----------
    residual : np.ndarray
      Current residual matrix
    extension_list : List[int]
      List of item indices to sort
    T : np.ndarray
      Current transaction vector
    method : str
      Sorting method
    """
    if not extension_list:
      return extension_list
    
    scores = np.zeros(len(extension_list))
    
    if method == 'frequency':
      # Sort by frequency in residual
      for i, item_idx in enumerate(extension_list):
        scores[i] = np.sum(residual[:, item_idx])
    
    elif method == 'couples-frequency':
      # Sort by frequency of item-pairs
      for i, item_idx in enumerate(extension_list):
        T_item = residual[:, item_idx]
        trans_indices = np.where(T_item > 0)[0]
        if len(trans_indices) > 0:
          scores[i] = np.sum(residual[trans_indices, :]) - np.sum(T_item)
        else:
          scores[i] = 0
    
    elif method == 'correlation':
      # Sort by correlation with current transactions
      trans_indices = np.where(T > 0)[0]
      if len(trans_indices) > 0:
        for i, item_idx in enumerate(extension_list):
          scores[i] = np.sum(residual[trans_indices, item_idx])
      else:
        scores = np.zeros(len(extension_list))
    
    # Sort in descending order
    sorted_indices = np.argsort(scores)[::-1]
    return [extension_list[i] for i in sorted_indices]
  
  def find_core(self, residual: np.ndarray) -> tuple:
    """
    Find a dense core pattern (T, I) and extension list E.
    
    Returns:
    --------
    T : np.ndarray
      Transaction (row) vector
    I : np.ndarray  
      Item (column) vector
    extension_list : List[int]
      Remaining items for extension
    """
    m, n = residual.shape
    
    # Initialize extension list with all items
    extension_list = list(range(n))
    I = np.zeros(n, dtype=int)
    T = np.zeros(m, dtype=int)
    
    # Sort items by chosen method
    extension_list = self.sort_items(residual, extension_list, T, self.init_method)
    
    if not extension_list:
      return T, I, extension_list
    
    # Add first item to I and update T
    first_item = extension_list.pop(0)
    I[first_item] = 1
    T = residual[:, first_item].copy()
    
    # Extend core greedily
    i = 0
    while i < len(extension_list):
      I_new = I.copy()
      
      if self.init_method == 'correlation':
        # Re-sort and use highest correlated item
        extension_list = self.sort_items(residual, extension_list, T, 'correlation')
        if extension_list:
          next_item = extension_list[0]
          I_new[next_item] = 1
          T_new = T * residual[:, next_item]
        else:
          break
      else:
        next_item = extension_list[i]
        I_new[next_item] = 1
        T_new = T * residual[:, next_item]
      
      # Calculate cost difference
      w0, h0 = np.sum(I), np.sum(T)
      w1, h1 = np.sum(I_new), np.sum(T_new)
      d_cost = self.w_model * ((w1 + h1) - (w0 + h0)) - self.w_fn * ((w1 * h1) - (w0 * h0))
      
      if d_cost <= 0:  # Cost improvement
        I = I_new
        T = T_new
        if self.init_method == 'correlation':
          extension_list.pop(0)
        else:
          extension_list.pop(i)
      else:
        i += 1
    
    return T, I, extension_list
  
  def extend_core(self, A: np.ndarray, residual: np.ndarray, T: np.ndarray, 
                  I: np.ndarray, extension_list: List[int]) -> tuple:
    """
    Extend core pattern with items from extension list.
    
    Returns:
    --------
    T : np.ndarray
      Updated transaction vector
    I : np.ndarray
      Updated item vector
    """
    T = T.copy()
    I = I.copy()
    
    # Try to add each item in extension list
    for item_idx in extension_list:
      I_new = I.copy()
      I_new[item_idx] = 1
      
      # Calculate cost for adding this item
      T_indices = np.where(T > 0)[0]
      if len(T_indices) > 0:
        partial_fn = -np.sum(residual[T_indices, item_idx])
        prediction_subset = A[T_indices, item_idx]  # Simplified prediction
        partial_fp = np.sum(T) - np.sum(prediction_subset) + partial_fn
        d_cost_item = self.w_model * 1 + self.w_fp * partial_fp + self.w_fn * partial_fn
      else:
        d_cost_item = self.w_model * 1
      
      if d_cost_item <= 0:  # Item improves cost
        I = I_new
        
        # Try to add transactions
        I_indices = np.where(I > 0)[0]
        T_candidates = np.where(T == 0)[0]  # Transactions not in T
        
        for trans_idx in T_candidates:
          d_fn = -np.sum(residual[trans_idx, I_indices])
          d_fp = np.sum(I) - np.sum(A[trans_idx, I_indices]) + d_fn
          d_cost_trans = self.w_model * 1 + self.w_fn * d_fn + self.w_fp * d_fp
          
          if d_cost_trans <= 0:  # Transaction improves cost
            T[trans_idx] = 1
    
    return T, I
  
  def solve(self, A: np.ndarray) -> BMFResult:
    """
    Solve Boolean Matrix Factorization using PaNDa algorithm.
    """
    self._validate_input(A)
    start_time = time.time()
    m, n = A.shape
    
    # Initialize factors
    U = np.zeros((m, 0), dtype=int)
    V = np.zeros((0, n), dtype=int)
    
    # Initialize residual
    residual = A.copy()
    
    factor_count = 0
    cost_old = self.w_fn * np.sum(residual)
    
    print(f"[Panda] Starting factorization, initial cost: {cost_old:.6f}")
    
    while True:
      # Find core pattern
      T, I, extension_list = self.find_core(residual)
      
      # Extend core if not in exact decomposition mode
      if not self.exact_decomp and extension_list:
        T, I = self.extend_core(A, residual, T, I, extension_list)
      
      # Check if pattern is valid
      if np.sum(T) == 0 or np.sum(I) == 0:
        print(f"[Panda] No valid pattern found at iteration {factor_count}")
        break
      
      # Add factor to matrices
      U_new = np.column_stack([U, T.reshape(-1, 1)]) if U.size > 0 else T.reshape(-1, 1)
      V_new = np.vstack([V, I.reshape(1, -1)]) if V.size > 0 else I.reshape(1, -1)
      
      # Calculate new cost
      cost_new = self.description_length(A, U_new, V_new)
      
      print(f"[Panda] Factor {factor_count}: cost {cost_old:.6f} -> {cost_new:.6f}, "
            f"pattern size: {np.sum(T)} x {np.sum(I)}")
      
      # Check stopping criteria
      if cost_new > cost_old:
        print(f"[Panda] Cost increased, stopping")
        break
      
      if self.k is not None and factor_count >= self.k:
        print(f"[Panda] Reached target rank {self.k}")
        break
      
      if abs(cost_new - cost_old) < self.tol:
        print(f"[Panda] Converged within tolerance {self.tol}")
        break
      
      # Update factors and residual
      U = U_new
      V = V_new
      residual = self.get_residual(A, U, V)
      cost_old = cost_new
      factor_count += 1
      
      # Additional stopping criterion - no significant residual
      if np.sum(residual > 0) == 0:
        print(f"[Panda] Perfect factorization achieved")
        break
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"[Panda] factorize runtime: {runtime:.6f} seconds")
    print(f"[Panda] Final factors: {factor_count}, final cost: {cost_old:.6f}")
    
    # Ensure we have valid factors
    if U.size == 0:
      U = np.zeros((m, 1), dtype=int)
      V = np.zeros((1, n), dtype=int)
    
    metadata = {
      'k': factor_count,
      'w_model': self.w_model,
      'w_fp': self.w_fp,
      'w_fn': self.w_fn,
      'init_method': self.init_method,
      'exact_decomp': self.exact_decomp,
      'final_cost': cost_old
    }
    
    return BMFResult(A=A, B=U, C=V, time_taken=runtime, metadata=metadata)
