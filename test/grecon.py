from BMF import GreCon
import numpy as np

X = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 1],
])

model = GreCon()
res = model.solve(X)
print(res)
res.show_factors()
