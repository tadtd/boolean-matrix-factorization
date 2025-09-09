from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Any
import numpy as np

Biclique = Tuple[Set[Any], Set[Any]]

class BicliqueCoverSolver(ABC):
  '''
  Abstract base class for biclique covering algorithms.
  Subclasses must implement `solve()`
  '''
  def __init__(self, 
               left_nodes: List[Any] = [], 
               right_nodes: List[Any] = [], 
               edges: List[Tuple[Any, Any]] = []):
    '''
    Params:
      - left_nodes: List of nodes on the left side 
      - right_nodes: List of nodes on the right side 
      - edges: List of edges 
    '''
    self.validate_bipartite(left_nodes, right_nodes, edges)
    
    self.left_nodes = set(left_nodes)
    self.right_nodes = set(right_nodes)
    self.edges = set(edges)
  
    self.neighbors_left = {u: set() for u in self.left_nodes}
    self.neighbors_right = {v: set() for v in self.right_nodes}

    for u, v in self.edges:
      self.neighbors_left[u].add(v)
      self.neighbors_right[v].add(u)

  @classmethod
  def from_matrix(cls, D: np.ndarray) -> 'BicliqueCoverSolver':
    m, n = D.shape

    left_nodes = [f'T{i}' for i in range(m)]
    right_nodes = [f'I{i}' for i in range(n)]

    edges = []
    for i in range(m):
      for j in range(n):
        if D[i, j] == 1:
          edges.append(left_nodes[i], right_nodes[j])
    
    return cls(left_nodes, right_nodes)
  
  @abstractmethod
  def solve(self) -> List[Biclique]:
    pass

  def validate_biclique(self, A: Set[Any], B: Set[Any]) -> bool:
    '''
    Check if (A, B) is a valid biclique.
    '''
    return all(b in self.neighbors_left[a] for a in A for b in B)
  
  def validate_bipartite(self,
                         left_nodes: List[Any],
                         right_nodes: List[Any],
                         edges: List[Tuple[Any, Any]]):
    '''
    Check if the given bipartite graph is valid.
    '''
    left_set = set(left_nodes)
    right_set = set(right_nodes)
    for u, v in edges:
      if u not in left_set or v not in right_set:
        raise ValueError(f'Edge ({u}, {v}) is not valid.')
