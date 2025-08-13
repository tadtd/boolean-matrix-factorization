import numpy as np

class Simplex:
  def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray):
    self.c = c
    self.A = A
    self.b = b
    self.m, self.n = A.shape
  
  def solve(self):
    ...
    
  