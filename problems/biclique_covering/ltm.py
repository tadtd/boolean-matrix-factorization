from .base import BicliqueCoverSolver, Biclique
from typing import List, Any, Tuple, Set, Dict

Item = int
TidList = Set[int]

class TidListDB:
  def __init__(self, item_to_tids: Dict[Item, TidList]):
    self.db = item_to_tids
    self.items = sorted(item_to_tids.keys())
  
  def cover(self, item: Item) -> TidList:
    return self.db.get(item, set())

class LTM(BicliqueCoverSolver):
  def __init__(self,
               left_nodes: List[Any] = [], 
               right_nodes: List[Any] = [], 
               edges:List[Tuple[Any, Any]] = [], 
               threshold: int = 5):
    super.__init__(left_nodes, right_nodes, edges)
    self.threshold = threshold

    self.tids = sorted(list(self.left_nodes))
    self.items = sorted(list(self.right_nodes))

    self.tid_to_idx = {node: i for i, node in enumerate(self.tids)}
    self.item_to_idx = {node: j for j, node in enumerate(self.items)}

    self.idx_to_tid = {i: node for i, node in enumerate(self.tids)}
    self.idx_to_item = {j: node for j, node in enumerate(self.items)}

    initial_db_dict: Dict[Item, TidList] = {
      i: set for i in self.idx_to_item
    }

    for u, v in self.edges:
      tid_idx = self.tid_to_idx[u]
      item_idx = self.item_to_idx[v]
      initial_db_dict[item_idx].add(tid_idx)
    
    self.initial_db = TidListDB(initial_db_dict)
    self.found_tiles: List[Tuple[TidList, Set[Item]]] = []
  
  def solve(self) -> List[Biclique]:
    self.found_tiles = []
    self._ltm_recursive(prefix_I=set(), db = self.initial_db)

    bicliques: List[Biclique] = []
    for tid_indices, item_indices in self.found_tiles:
      left_side_nodes = {self.idx_to_tid[tid] for tid in tid_indices}
      right_side_nodes = {self.idx_to_item[item] for item in item_indices}
      bicliques.append((left_side_nodes, right_side_nodes))
    
    return bicliques

  def _prune(self, db: TidList, prefix_I: Set[Item]) -> TidListDB:
    pruned_db_dict = db.db.copy()
    while True:
      changed = False
      tid_sizes: Dict[int, int] = {}
      all_tids_in_db = set.union(*pruned_db_dict.values()) if pruned_db_dict else set()

      for tid in all_tids_in_db:
        size = 0
        for item in pruned_db_dict:
          if tid in pruned_db_dict[item]:
            size += 1
        tid_sizes[tid] = size
      items_to_iterate = sorted(pruned_db_dict.keys())

      for item_i in items_to_iterate:
        if item_i not in pruned_db_dict:
          continue

        cover_i = pruned_db_dict[item_i]
        
        # pruning 1: based on UPPER BOUND (UB)
        max_area_ub = 0
        relevant_tid_sizes = sorted(
          [tid_sizes[tid] for tid in cover_i if tid in tid_sizes],
          reverse=True)
        
        max_l = len(relevant_tid_sizes)
        for l in range(1, max_l + 1):
          supp_ge_l = l
          potential_area = (len(prefix_I) + relevant_tid_sizes[l-1]) * supp_ge_l
          if potential_area > max_area_ub:
            max_area_ub = potential_area

        if max_area_ub < self.threshold:
          del pruned_db_dict[item_i]
          changed = True
          continue

        # pruning 2: based on MINIMUM LENGTH (ML)
        min_len_ml = float('inf')
        for l in range(1, max_l + 1):
          supp_ge_l = l
          if (len(prefix_I) + relevant_tid_sizes[l-1]) * supp_ge_l >= self.threshold:
            min_len_ml = relevant_tid_sizes[l-1]
            break
        
        tids_to_remove_from_item = set()
        for tid in cover_i:
          if tid_sizes.get(tid, 0) < min_len_ml:
            tids_to_remove_from_item.add(tid)
            changed = True
        
        if tids_to_remove_from_item:
          pruned_db_dict[item_i].difference_update(tids_to_remove_from_item)
          if not pruned_db_dict[item_i]:
            del pruned_db_dict[item_i]
            
      # if nothing changed in the full loop, exit
      if not changed:
        break
    return TidListDB(pruned_db_dict)

  def _ltm_recursive(self, prefix_I: Set[Item], db: TidListDB):
    current_db = self._prune(db=db, prefix_I=prefix_I)
    items_to_check = current_db.items

    for i, item_i in enumerate(items_to_check):
      cover_i = current_db.cover(item_i)
      
      if len(cover_i) * (len(prefix_I) + 1) >= self.threshold:
        new_tile_itemset = prefix_I.union({item_i})
        self.found_tiles.append((cover_i, new_tile_itemset))
      
      conditional_db_dict: Dict[Item, TidList] = {}
      for j in range(i + 1, len(items_to_check)):
        item_j = items_to_check[j]
        cover_j = current_db.cover(item_j)
        intersection_tids = cover_i.intersection(cover_j)

        if intersection_tids:
          conditional_db_dict[item_j] = intersection_tids
        
        if conditional_db_dict:
          new_prefix = prefix_I.union({item_i})
          conditional_db = TidListDB(conditional_db_dict)
          self._ltm_recursive(prefix_I=new_prefix, db=conditional_db)
