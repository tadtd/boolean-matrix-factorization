from BMF import GreConD, utils
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder

def main():
  model = GreConD()
  mushroom = fetch_ucirepo(id=73) 

    # data (as pandas dataframes) 
  X = mushroom.data.features 
  
  encoder = OneHotEncoder(sparse_output=False, dtype=int)
  X_bin = encoder.fit_transform(X)
  X_bin = X_bin.astype(int)

  Y = utils.random_boolean_matrix(80, 50, density=0.1, random_state=42)
  res = model.solve(Y)
  print(res)
  res.show_factors()
  # print(res.reconstruction)

if __name__ == "__main__":
  main()