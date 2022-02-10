import numpy as np

dist = np.load(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_dist.npy'))
dist = np.array(dist)
dist = np.delete(dist, np.s_[29:], 0)
np.save(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_dist.npy'),dist)


# conn = np.load(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_conn.npy'))
# conn = np.array(conn)

# conn = np.delete(conn, np.s_[29:], 0)
# np.save(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_conn.npy'),conn)

# conn = np.load(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_conn.npy'))
# conn = np.array(conn)
# print(conn.shape)
#np.savetxt(("/home/mingzeli/neuro/neuromorphicnetworks/data/marmoset_conn.csv"), label, delimiter=",")
#np.savetxt("/home/mingzeli/neuro/neuromorphicnetworks/data/marmoset_conn.csv", data[0], delimiter=",")




# dist1 = np.load(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_dist.npy'))

# conn = np.zeros((29,29))
# dist = np.zeros((29,29))
# for i in range(29):
#     conn[i] = conn1[i]
#     dist[i] = dist1[i]
# print(np.shape(conn))
# print(np.shape(dist))

# np.save(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_conn.npy'),conn)
# np.save(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_dist.npy'),dist)

# conn1 = np.load(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_conn.npy'))
# dist1 = np.load(('/home/mingzeli/neuro/neuromorphicnetworks/data/macaque_markov_dist.npy'))
# print(np.shape(conn1))