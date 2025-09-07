from typing import Dict, Type

from .base import SetCoverSolver
from .greedy import Greedy

_SOLVERS: Dict[str, Type[SetCoverSolver]] = {
  'greedy': Greedy,
}

def SetCover(solver_name: str, **kwargs) -> SetCoverSolver:
  solver_class = _SOLVERS.get(solver_name)  

  if solver_class is None:
    available = list(_SOLVERS.keys())
    raise ValueError(f"Unknown solver: '{solver_name}'. Available solvers: {available}")
  
  return solver_class(**kwargs)
