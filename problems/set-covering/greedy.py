
# Greedy approach to the Set Covering Problem
def set_covering_greedy(universe: set, subsets: list[set]) -> list[set]:
  '''
  Greedily selects subsets to cover the universe.
    :param universe: A set representing the universe to be covered.
    :param subsets: A list of sets, each representing a subset of the universe.
  :return: A list of selected subsets that cover the universe.
  '''
  covered = set()
  selected_subsets = []
  while covered != universe:
    best_subset = None
    num_of_elements = 0
    for subset in subsets:
      temp_set = subset - covered 
      if temp_set != set() and num_of_elements < len(temp_set):
        best_subset = subset
        num_of_elements = len(temp_set)
    
    if best_subset is None:
      raise ValueError('Cannot cover the universe with the given subsets.')

    covered.update(best_subset)
    selected_subsets.append(best_subset)
    subsets.remove(best_subset)
  return selected_subsets

U = {1, 2, 3, 4, 5, 6}
S = [{1, 2, 3, 4, 5}, {1, 2, 3, 4}, {6}]

result = set_covering_greedy(U, S)
for i, s in enumerate(result):
  print(f'Subset {i+1}: {s}')