from .base import SetCoverSolver
from typing import List, Set, Any

class Greedy(SetCoverSolver):
  def solve(self) -> List[Set[Any]]:
    selected_subsets: List[Set[Any]] = []
    available_subsets = list(self.subsets)
    uncovered = set(self.universe)

    while uncovered:
      best_subset = max(
        available_subsets,
        key=lambda s: len(s.intersection(uncovered))
      )

      if not best_subset.intersection(uncovered):
        raise ValueError('Cannot cover the universe with the given subsets.')
      selected_subsets.append(best_subset)
      uncovered -= best_subset
    return selected_subsets