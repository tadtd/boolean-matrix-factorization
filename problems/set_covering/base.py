from abc import ABC, abstractmethod
from typing import List, Set, Any

class SetCoverSolver(ABC):
  def __init__(self,
               universe: List[Any],
               subsets: List[Set[Any]]):
    self.universe = set(universe)
    self.subsets = subsets
  
  @abstractmethod
  def solve(self) -> List[Set[Any]]:
    pass