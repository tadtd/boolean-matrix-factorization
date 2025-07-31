import numpy as np

def boolean_product(A, B):
  return (A @ B > 0).astype(int)

def cover(B, S, C, wp=1, wn=1):
  appr = boolean_product(S, B)
  return wp*(np.sum((C == 1) & (appr == 1))) - wn*(np.sum((C == 0) & (appr == 1)))

# Asso algorithm
def asso(C, k, tau=.5, wp=1, wn=1):
  n, m = C.shape
  A = np.zeros((n, m), dtype=int)
  B = np.zeros((k, m), dtype=int)
  S = np.zeros((n, k), dtype=int)

  for i in range(n):
    denom = np.dot(C[:, i], C[:, i])
    for j in range(m):
      if np.dot(C[:, j], C[:, i]) / max(denom, 1e-10) >= tau:
        A[i][j] = 1

  for l in range(k):
    best_score = -np.inf
    best_i = -1

    for i in range(n):
      B[l] = A[i]
      S[:,l] = boolean_product(C, A[i])
      score = cover(B, S, C, wp, wn)

      if score > best_score:
        best_score = score
        best_i = i
    
    B[l] = A[best_i]
    S[:,l] = boolean_product(C, A[best_i])

  return B, S

C = np.array([
    [1, 0, 1, 0, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 0]
])

B, S = asso(C, k=2)
print(f'first:\n {B}')
print(f'second:\n {S}')
print(f'original:\n {C}')
print(f'Appr:\n {boolean_product(S, B)}')