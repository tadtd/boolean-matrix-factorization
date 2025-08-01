from abc import ABC, abstractmethod
import numpy as np

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
    self.rank = rank
    self.params = kwargs
  
  @abstractmethod
  def fit(self, C):
    '''
    Fit the model to the data.
    Args:
      C: Input data matrix.
    Returns:
      self: Fitted model instance.
    '''
    pass
  
  @property
  @abstractmethod
  def name(self) -> str:
    ''' 
    Return the name of the algorithm.
    '''
    pass

  @abstractmethod
  def reconstruct(self):
    '''
    Reconstruct the original data matrix.
    Returns:
      Reconstructed data matrix.
    ''' 
    pass

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

    if self.rank <= 0 or self.rank > min(A.shape):
      raise ValueError(f'Invalid rank k = {self.rank} for matrix shape {A.shape}')
