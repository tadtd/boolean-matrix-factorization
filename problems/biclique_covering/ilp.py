# Integer Linear Programming model for the biclique covering problem
from .base import BicliqueCoverSolver, Biclique
from typing import List
import numpy as np

class ILP(BicliqueCoverSolver):
  def __init__(self, left_nodes, right_nodes, edges):
    super().__init__(self, left_nodes, right_nodes, edges)
    
    self.left_nodes = list(self.left_nodes)
    self.right_nodes = list(self.right_nodes)

    left_idx = {node: i for i, node in enumerate(self.left_nodes)}
    right_idx = {node: j for j, node in enumerate(self.right_nodes)}
    
    m, n = len(self.left_nodes), len(self.right_nodes)
    self.A = np.zeros((m, n))
    for (u, v) in self.edges:
      self.A[left_idx[u], right_idx[v]] = 1
  
  def solve(self) -> List[Biclique]:
    pass
