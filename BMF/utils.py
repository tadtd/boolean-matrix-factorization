import numpy as np
from typing import Dict, Any
import os

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
  
  @staticmethod
  def is_boolean_matrix(A: np.ndarray) -> None:
    if not np.all((A==0) | (A==1)):
      raise ValueError('Input must be a boolean matrix')
  
  @staticmethod
  def random_boolean_matrix(m: int, n: int, density: float = 0.5, random_state: int = None) -> np.ndarray:
    '''
    Create a random boolean matrix.
    Args:
      m: Number of rows
      n: Number of columns
      density: Density of 1s in the matrix
      random_state: Random seed
    Returns:
      Random boolean matrix of shape (m, n).
    '''
    if random_state is not None:
      np.random.seed(random_state)
    return (np.random.rand(m, n) < density).astype(int)
    
class BMFResult:
  def __init__(self,
               A: np.ndarray,
               B: np.ndarray,
               C: np.ndarray,
               time_taken: float = 0.0,
               metadata: Dict[str, Any] = {}):
    self.A = A
    self.B = B
    self.C = C
    self.rank = B.shape[1]
    self.reconstruction = utils.boolean_product(B, C)
    self.error = np.linalg.norm(np.bitwise_xor(self.A, self.reconstruction), ord='fro') ** 2
    self.coverage = np.sum(self.reconstruction) / np.sum(self.A)  
    self.metadata = metadata
    self.time_taken = time_taken

  def summary(self) -> Dict[str, Any]:
    return {
      'rank': self.rank,
      'error': self.error,
      'coverage': self.coverage,
      'metadata': self.metadata
    }

  def show_factors(self) -> None:
    '''
    Pretty-print the factor matrices B and C 
    '''
    print('Factor matrix B (objects x rank):')
    for row in self.B:
      print("[" + ", ".join(str(int(x)) for x in row) + "]")

    print('Factor matrix C (rank x attributes):')
    for row in self.C:
      print("[" + ", ".join(str(int(x)) for x in row) + "]")

  def save_factors(self, filename_prefix: str, path: str = '.', filetype: str = 'csv') -> None:
    if filetype not in ('csv', 'txt'):
      raise ValueError("File type must be 'csv' or 'txt'")
    
    os.makedirs(path, exist_ok=True)

    B_file = os.path.join(path, f"{filename_prefix}_B.{filetype}")
    C_file = os.path.join(path, f"{filename_prefix}_C.{filetype}")
    A_hat_file = os.path.join(path, f"{filename_prefix}_A_hat.{filetype}")

    if filetype == 'csv':
      np.savetxt(B_file, self.B, fmt='%d', delimiter=',')
      np.savetxt(C_file, self.C, fmt='%d', delimiter=',')
      np.savetxt(A_hat_file, self.reconstruction, fmt='%d', delimiter=',')
    elif filetype == 'txt':
      np.savetxt(B_file, self.B, fmt='%d', delimiter=' ')
      np.savetxt(C_file, self.C, fmt='%d', delimiter=' ')
      np.savetxt(A_hat_file, self.reconstruction, fmt='%d', delimiter=' ')

  def __str__(self) -> str:
    summary = self.summary()
    return f"Rank={summary['rank']}, error={summary['error']}, coverage={summary['coverage']}"
  
  def __repr__(self) -> str:
    return self.__str__()