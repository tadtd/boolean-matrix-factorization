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
  pass