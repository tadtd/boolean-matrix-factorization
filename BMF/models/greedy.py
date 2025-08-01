from base import BMFAlgorithm
import numpy as np

class GreedyBMF(BMFAlgorithm):
  '''
  Greedy BMF algorithm implementation.
  '''
  def __init__(self, rank):
    '''
    Initialize the GreedyBMF model.
    Args:
      rank: The rank (number of latent factors) for the BMF.
    '''
    super().__init__(rank=rank)
    self.rank = rank
    self.B = None
    self.S = None
  
  def fit(self, C: np.ndarray):
    '''
    Fit the model to the data.
    Args:
      C: Input data matrix.
    Returns:
      self: Fitted model instance.
    '''
    pass