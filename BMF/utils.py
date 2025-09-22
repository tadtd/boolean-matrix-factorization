import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

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
  def __init__(self,
               A: np.ndarray,
               B: np.ndarray,
               C: np.ndarray,
               metadata: Dict[str, Any] = {}):
    self.A = A
    self.B = B
    self.C = C
    self.rank = B.shape[1]
    self.reconstruction = utils.boolean_product(B, C)
    self.error = np.linalg.norm(np.bitwise_xor(self.A, self.reconstruction), ord='fro')
    self.metadata = metadata

  def summary(self) -> Dict[str, Any]:
    return {
      'rank': self.rank,
      'error': self.error,
      'metadata': self.metadata
    }

  def __str__(self) -> str:
    summary = self.summary()
    return f"Rank={summary['rank']}, error={summary['error']}"
  
  def __repr__(self) -> str:
    return self.__str__()