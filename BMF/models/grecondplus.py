from .base import BMFAlgorithm
from ..utils import BMFResult
import numpy as np

class GreConDPlus(BMFAlgorithm):
  def __init__(self):
    super().__init__()
  
  @property
  def name(self) -> str:
    return 'GreConD+'

  def solve(self, A: np.ndarray) -> BMFResult:
    self._validate_input(A)
