from abc import ABC, abstractmethod
import numpy as np
from ..utils import BMFResult, utils

class BMFAlgorithm(ABC):
  '''
  Base class for all BMF algorithms. 
  '''

  def __init__(self, rank: int, **kwargs):
    '''
    Initialize the BMF algorithm with given parameters.
    Args:
      **kwargs: Algorithm-specific parameters.
    '''
    self.B = None
    self.C = None
    self.params = kwargs
  
  @abstractmethod
  def solve(self, C: np.ndarray) -> BMFResult:
    '''
    Fit the model to the data.
    Args:
      C: Input data matrix.
    Returns:
      self: Fitted model instance.
    '''
    pass
  
  @abstractmethod
  def name(self) -> str:
    ''' 
    Return the name of the algorithm.
    '''
    pass

  def reconstruct(self, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    '''
    Reconstruct the original data matrix.
    Returns:
      Reconstructed data matrix.
    ''' 
    if self.B is None or self.C is None:
      raise ValueError('Model must be fitted before reconstruction')
    return utils.boolean_product(self.B, self.C)

  def _validate_input(self, A: np.ndarray):
    '''
    Validate input matrix
    '''
    if not isinstance(A, np.ndarray):
      raise TypeError('Input must be a numpy array')
    
    if A.ndim != 2:
      raise ValueError('Input must be a 2D matrix')
    
    if not np.all((A==0) | (A==1)):
      raise ValueError('Input must be a boolean matrix')
