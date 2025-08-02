from BMF import GreedyBMF as greedy, AssoAlgorithm as asso
import numpy as np

X = np.array([
  [1, 1, 0],
  [1, 1, 1],
  [0, 1, 1]
])
for tau in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
  algo = asso(rank=2, tau=tau)
  result = algo.fit(X)
  print(f"\ntau = {tau}")
  print(f"Error: {result.error}")
  print("B:\n", result.B)
  print("C:\n", result.C)
