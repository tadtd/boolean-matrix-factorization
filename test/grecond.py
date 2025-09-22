from BMF import GreConD
import numpy as np

model = GreConD()
X = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 1],
])

res = model.solve(X)
print(res)
res.show_factors()
# print(res.reconstruction)