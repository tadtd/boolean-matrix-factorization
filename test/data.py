from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder
from BMF import utils

# fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
# data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
# print(mushroom.metadata) 
  
# variable information 
# print(mushroom.variables) 

encoder = OneHotEncoder(sparse_output=False, dtype=int)
X_bin = encoder.fit_transform(X)
X_bin = X_bin.astype(int)

print(X_bin)
print(X_bin.dtype)
utils.is_boolean_matrix(X_bin)

print("Shape of binary matrix:", X_bin.shape)