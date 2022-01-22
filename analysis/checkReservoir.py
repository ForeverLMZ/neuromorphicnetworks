import numpy as np
import os
from bct import reference

aarr = ("John", "Charles", "Mike")
barr = ("Jenny", "Christy", "Monica")
result = [(a, b) for a, b in zip(aarr, barr)]
result = result[:, :, 0]
print(result)
# conn = np.load('./data/macaque_modha_conn.npy', allow_pickle=True)
# rewired =[]
# test = []
# for i in range(10):
#     temp,num = reference.randmio_dir(conn, 10)
#     rewired.append(temp)

# for i in range(10):
#     for j in range(10):
#         if (i != j):
#             print(np.array_equal(rewired[i],rewired[j]))
# for i in range(80):
#     for j in range(80):
#         a = np.load("./raw_results/sim_results/reliability/reservoir_states_25.npy", allow_pickle=True)
#         b = np.load("./raw_results/sim_results/reliability/reservoir_states_39.npy", allow_pickle=True)
#         # rtol = 0
#         # atol = 1e-09
        
#         # print(np.allclose(a, b, rtol, atol))']
#         if (np.array_equal(a,b) is False):
#             print("the reservoir states are NOT the same for: ", i," and ",j)
