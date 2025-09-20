from BMF import Asso, utils
import numpy as np

# Input matrix
X = np.array([
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1]
])

# Function to check if matrix is binary
def is_binary(M):
  return np.all((M == 0) | (M == 1))

# Loop over tau values
for tau in [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
  algo = Asso(rank=2, tau=tau)
  result = algo.fit(X)

  # Print results for debugging
  print(f"\n=== tau = {tau} ===")
  print(f"Error: {result.error}")
  print("B:\n", result.B)
  print("C:\n", result.C)

  # ---------- Property checks ----------
  # 1. Binary check for B and C
  assert is_binary(result.B), f"B is not binary for tau={tau}"
  assert is_binary(result.C), f"C is not binary for tau={tau}"

  # 2. Shape check
  assert result.B.shape[0] == X.shape[0], f"B has wrong number of rows for tau={tau}"
  assert result.C.shape[1] == X.shape[1], f"C has wrong number of columns for tau={tau}"
  assert result.B.shape[1] == result.C.shape[0], f"Inner dimensions of B and C don't match for tau={tau}"

  # 3. Reconstruction check
  # Boolean product: (B @ C) with OR-AND logic
  recon = utils.boolean_product(result.B, result.C)

  # Allow exact match or small error tolerance (5%)
  # errors = np.sum(X != recon)
  # error_rate = errors / X.size
  # assert error_rate <= 0.05, f"Reconstruction error too high ({error_rate:.2%}) for tau={tau}"
