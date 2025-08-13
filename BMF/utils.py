import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import time

class utils:
  '''
  Utility class for Boolean matrix operations.
  '''

  @staticmethod
  def boolean_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Compute boolean matrix product.
    Args:
      A: First matrix
      B: Second matrix
    Returns:
      Boolean product of A and B.
    '''
    return (A @ B > 0).astype(int)
  
  @staticmethod
  def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    '''
    Compute cosine similarity between two vectors.
    Args:
      v1: First vector
      v2: Second vector
    Returns:
      Consine similarity value.
    '''
    denom = np.dot(v1, v1)
    if denom < 1e-10:
      return 0.0
    return np.dot(v2, v1) / denom
  
class BMFResult:
  '''
  Container class for Boolean Matrix Factorization results.
  
  Attributes:
    B: Left factor matrix (n x k)
    C: Right factor matrix (k x m)
    reconstruction: Reconstructed matrix B ⊙ C
    original_shape: Shape of the original matrix
    rank: Factorization rank (k)
    error: Reconstruction error
    iterations: Number of iterations performed
    convergence_time: Time taken to converge
    converged: Whether the algorithm converged
    metadata: Additional algorithm-specific information
  '''
  
  def __init__(self, B: np.ndarray, C: np.ndarray, 
               original_matrix: Optional[np.ndarray] = None,
               error: Optional[float] = None,
               iterations: int = 0,
               convergence_time: float = 0.0,
               converged: bool = False,
               metadata: Optional[Dict[str, Any]] = None):
    '''
    Initialize BMF result.
    
    Args:
      B: Left factor matrix
      C: Right factor matrix
      original_matrix: Original matrix for error computation
      error: Reconstruction error
      iterations: Number of iterations performed
      convergence_time: Time taken to converge
      converged: Whether the algorithm converged
      metadata: Additional algorithm-specific information
    '''
    self.B = B
    self.C = C
    self.rank = B.shape[1]
    self.original_shape = (B.shape[0], C.shape[1])
    self.error = error
    self.iterations = iterations
    self.convergence_time = convergence_time
    self.converged = converged
    self.metadata = metadata or {}
    
    # Compute reconstruction
    self.reconstruction = utils.boolean_product(self.B, self.C)

    # Compute error if original matrix is provided and error not given
    if original_matrix is not None and error is None:
      self.error = self.compute_error(original_matrix)
  
  def compute_error(self, original_matrix: np.ndarray) -> float:
    '''
    Compute reconstruction error (Frobenius norm of difference).
    
    Args:
      original_matrix: Original matrix to compare against
      
    Returns:
      Reconstruction error
    '''
    diff = original_matrix - self.reconstruction
    return np.linalg.norm(diff, 'fro')
  
  def relative_error(self, original_matrix: np.ndarray) -> float:
    '''
    Compute relative reconstruction error.
    
    Args:
      original_matrix: Original matrix to compare against
      
    Returns:
      Relative reconstruction error
    '''
    orig_norm = np.linalg.norm(original_matrix, 'fro')
    if orig_norm < 1e-10:
      return 0.0
    return self.compute_error(original_matrix) / orig_norm
  
  def sparsity_ratio(self) -> Tuple[float, float]:
    '''
    Compute sparsity ratios of factor matrices.
    
    Returns:
      Tuple of (B_sparsity, C_sparsity) where sparsity is ratio of zeros
    '''
    b_sparsity = 1.0 - np.count_nonzero(self.B) / self.B.size
    c_sparsity = 1.0 - np.count_nonzero(self.C) / self.C.size
    return b_sparsity, c_sparsity
  
  def compression_ratio(self) -> float:
    '''
    Compute compression ratio achieved by factorization.
    
    Returns:
      Compression ratio (original_size / factorized_size)
    '''
    original_size = self.original_shape[0] * self.original_shape[1]
    factorized_size = self.B.size + self.C.size
    return original_size / factorized_size
  
  def summary(self) -> Dict[str, Any]:
    '''
    Get summary statistics of the factorization result.
    
    Returns:
      Dictionary containing summary information
    '''
    b_sparsity, c_sparsity = self.sparsity_ratio()
    
    return {
      'original_shape': self.original_shape,
      'rank': self.rank,
      'error': self.error,
      'iterations': self.iterations,
      'convergence_time': self.convergence_time,
      'converged': self.converged,
      'compression_ratio': self.compression_ratio(),
      'B_sparsity': b_sparsity,
      'C_sparsity': c_sparsity,
      'B_shape': self.B.shape,
      'C_shape': self.C.shape,
      'metadata': self.metadata
    }
  
  def __str__(self) -> str:
    '''String representation of BMF result.'''
    summary = self.summary()
    return (f"BMFResult(shape={summary['original_shape']}, "
            f"rank={summary['rank']}, error={summary['error']:.4f}, "
            f"iterations={summary['iterations']}, "
            f"converged={summary['converged']})")
  
  def __repr__(self) -> str:
    '''Detailed string representation of BMF result.'''
    return self.__str__() 