from typing import Set, List
from set_covering import SetCover

def main():
  ...

if __name__ == '__main__':
  universe: Set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  subsets: List[Set[int]] = [
    {1, 2, 3, 8},
    {1, 2, 3, 4, 5},
    {6, 7},
    {2, 3, 4, 8, 9},
    {4, 5, 10},
    {6, 7, 9}
  ]
  print("--- Set Cover Problem ---")
  print(f"Universe to cover: {universe}")
  print(f"Available subsets: {subsets}\n")

  solver = SetCover(
    solver_name='greedy',
    universe=universe,
    subsets=subsets,
  )

  try:
    solution = solver.solve()
      
    # print the result.
    print("--- Greedy Solution ---")
    print(f"Found a cover with {len(solution)} subsets:")
    for i, subset in enumerate(solution):
      print(f"  Subset {i + 1}: {subset}")
    print("-----------------------")

  except ValueError as e:
    print(f"Error: {e}")


# This ensures the main function runs when the script is executed.
if __name__ == "__main__":
  main()