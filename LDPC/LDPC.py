import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message

n = 15         # Must be divisible by d_c
d_v = 4        # # of ones per column
d_c = 5        # # of ones per row
systematic = True
sparse = True  # Use sparse matrix representation
snr = 20        # Signal-to-noise ratio

H, G = make_ldpc(n, d_v, d_c, systematic=systematic, sparse=sparse)
k = G.shape[1] 
v = np.random.randint(2, size=k)
y = encode(G, v, snr=snr)
d = decode(H, y, snr=snr)

x = get_message(G, d)

print("H:\n", H)
print("G:\n", G)
print('original message:\n', v)
print('encoded message:\n', y)
print('decoded message:\n', d)
print('recovered message:\n', x) 