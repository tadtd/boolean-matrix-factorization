from biclique_covering import BicliqueCover

def main():
  """
  Main function to run the solver using the clean, simplified import.
  """
  # 2. Define your graph data
  users = ['Alice', 'Bob', 'Charlie']
  permissions = ['Read', 'Write', 'Execute']
  edges = [
    ('Alice', 'Read'), ('Alice', 'Write'),
    ('Bob', 'Read'),
    ('Charlie', 'Write'), ('Charlie', 'Execute')
  ]

  # 3. Use the factory to get the correct solver instance
  #    The user only needs to know the name 'greedy'.
  solver = BicliqueCover(
    solver_name='greedy',
    left_nodes=users,
    right_nodes=permissions,
    edges=edges
  )

  # 4. Run the solver and print the solution
  solution = solver.solve()
  
  print("\n--- Biclique Cover Solution ---")
  for i, (left, right) in enumerate(solution):
    print(f"Biclique {i + 1}: Left={left}, Right={right}")
  print("-----------------------------")

if __name__ == "__main__":
  main()