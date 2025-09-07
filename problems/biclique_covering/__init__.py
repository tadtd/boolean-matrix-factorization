from typing import Dict, Type

from .base import BicliqueCoverSolver
from .greedy import Greedy

_SOLVERS: Dict[str, Type[BicliqueCoverSolver]] = {
  'greedy': Greedy,
}

def BicliqueCover(solver_name: str, **kwargs) -> BicliqueCoverSolver:
  '''
  Factory function to create a biclique cover solver instance.

  This is the single, clean entry point for the user.

  Params:
    - solver_name: The name of the solver to use (e.g., 'greedy').
    - **kwargs: Arguments for the graph (left_nodes, right_nodes, edges).
  '''
  solver_class = _SOLVERS.get(solver_name)
  
  if solver_class is None:
    available = list(_SOLVERS.keys())
    raise ValueError(f"Unknown solver: '{solver_name}'. Available solvers: {available}")
  
  # Create an instance of the chosen solver class and return it
  return solver_class(**kwargs)