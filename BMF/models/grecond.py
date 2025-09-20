import numpy as np
from .base import BMFAlgorithm
from ..utils import utils, BMFResult
import time

class GreConD(BMFAlgorithm):
  def __init__(self):
    super().__init__()

  @property
  def name(self) -> str:
    return 'GreConD'

  def solve(self, A: np.ndarray):
    self._validate_input(A)
    start = time.time()