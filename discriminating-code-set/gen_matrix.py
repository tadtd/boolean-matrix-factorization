# rs63_k3_gf64_ks_generate.py
# Generates coordinate-format (row_idx, col_idx, data) for RS(63,3) over GF(2^6)
# using Kautz–Singleton binary expansion and saves to rs63_k3_gf64_ks.npz

import numpy as np
import time

# Parameters
m_deg = 6
q = 1 << m_deg  # 64
n = 63
k = 3
cols = q ** k   # 262144
rows = n * q    # 4032
total_ones = n * cols
prim_poly = 0x43  # x^6 + x + 1

def build_gf_tables(prim_poly, m_deg):
  q = 1 << m_deg
  mask = q - 1
  exp = [0] * (q - 1)
  log = [-1] * q
  val = 1
  for i in range(q - 1):
    exp[i] = val
    log[val] = i
    val <<= 1
    if val & (1 << m_deg):
      val ^= prim_poly
    val &= mask
  return np.array(exp, dtype=np.int32), np.array(log, dtype=np.int32)

exp_table, log_table = build_gf_tables(prim_poly, m_deg)
eval_pts = exp_table  # 63 nonzero elements

# Precompute logs of evaluation points
x_vals = eval_pts                       # shape (63,)
x_logs = log_table[x_vals]              # shape (63,)
exp = exp_table
log = log_table

# Precompute mul tables:
# mul_table[a, pos] = a * x_vals[pos] in GF(64)
# mul_table_x2[a, pos] = a * (x_vals[pos]^2) in GF(64)
mul_table = np.zeros((q, n), dtype=np.int16)
mul_table_x2 = np.zeros((q, n), dtype=np.int16)

for a in range(1, q):
  la = log[a]
  # for a*x: exponent indices = la + x_logs (mod q-1)
  idxs = (la + x_logs) % (q - 1)
  mul_table[a, :] = exp[idxs]
  # for a*x^2: exponent indices = la + 2*x_logs
  idxs2 = (la + (2 * x_logs)) % (q - 1)
  mul_table_x2[a, :] = exp[idxs2]
# a==0 rows remain zeros

# Allocate arrays for coordinate format
row_idx = np.empty(total_ones, dtype=np.int32)
col_idx = np.empty(total_ones, dtype=np.int32)
data = np.ones(total_ones, dtype=np.uint8)

ptr = 0
start = time.time()
pos_offsets = (np.arange(n, dtype=np.int32) * q)  # pos*q offsets used repeatedly

for a2 in range(q):
  t2 = mul_table_x2[a2]      # length 63
  for a1 in range(q):
    t1 = mul_table[a1]     # length 63
    combined = np.bitwise_xor(t1, t2).astype(np.int32)  # t1 + t2 (GF add = XOR)
    base_col_high = (a2 << (m_deg*2)) | (a1 << m_deg)
    for a0 in range(q):
      col = base_col_high | a0
      y = np.bitwise_xor(combined, a0).astype(np.int32)  # length 63
      rows_vec = pos_offsets + y  # length 63
      row_idx[ptr:ptr+n] = rows_vec
      col_idx[ptr:ptr+n] = col
      ptr += n
  # optional progress print
  if (a2 + 1) % 8 == 0:
    elapsed = time.time() - start
    print(f"a2 {a2+1}/{q} elapsed {elapsed:.1f}s")

end = time.time()
print(f"Done: ptr={ptr}, expected={total_ones}, time {end - start:.1f}s")

# Sanity check
assert ptr == total_ones

# Save compressed npz file
out_path = "rs63_k3_gf64_ks.npz"
np.savez_compressed(out_path,
                    row_idx=row_idx, col_idx=col_idx, data=data,
                    rows=rows, cols=cols, n=n, k=k, q=q)
print("Saved to", out_path)
