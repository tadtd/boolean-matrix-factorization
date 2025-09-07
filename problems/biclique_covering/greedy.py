from typing import List, Tuple, Set, Any
from .base import BicliqueCoverSolver, Biclique

class Greedy(BicliqueCoverSolver):
  def solve(self) -> List[Biclique]:
    uncover = set(self.edges)
    cover: List[Biclique] = []

    while uncover:
      start_node, _ = next(iter(uncover))
      neighbors = {
        neighbor for neighbor in self.neighbors_left[start_node]
        if (start_node, neighbor) in uncover
      }

      if not neighbors:
        uncover.discard((start_node, _))
        continue
      
      left_nodes = {
        node for node in self.left_nodes
        if neighbors.issubset(self.neighbors_left.get(node, set()))
      }

      biclique: Biclique = (left_nodes, neighbors)
      cover.append(biclique)

      edges_in_biclique = {
        (u, v) for u in left_nodes for v in neighbors if (u, v) in self.edges
      }
      uncover -= edges_in_biclique

    return cover