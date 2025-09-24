from BMF import GreCon, utils
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder

def main():
  model = GreCon()
  mushroom = fetch_ucirepo(id=73) 

  X = mushroom.data.features 

  encoder = OneHotEncoder(sparse_output=False, dtype=int)
  X_bin = encoder.fit_transform(X)
  X_bin = X_bin.astype(int)

  Y = utils.random_boolean_matrix(100, 100, density=0.1, random_state=42)

  res = model.solve(Y)
  # res.show_factors()
  print(res)

if __name__ == "__main__":
  main()
