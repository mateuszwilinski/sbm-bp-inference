# -*- coding: utf-8 -*-

# Libraries, classes and functions

import numpy as np
import igraph as ig

from sbm_bp import DirectedMessages
from sbm_bp import get_real_overlap, directed_affinity

# Parameters

N = 1000  # number of nodes
q = 2  # number of blocks
c = 3  # average degree
eps = 0.1  # order parameter
eps_ = 0.1  # starting epsilon

n = np.ones(q) / q
p = directed_affinity(q, c, n, eps, eps_, N)

# Graph

G = ig.Graph.SBM(N, p.tolist(), (N * n).tolist(), directed=True)
groups = np.zeros(N, dtype='int')
for i in range(q - 1):
    groups[int(N * np.sum(n[:(i + 1)])):] = i + 1
G.vs['group'] = groups

# Initial parameters

q_ = q
c_ = np.mean(G.degree()) / 2.0

n_ = np.ones(q_) / q_
p_ = directed_affinity(q_, c_, n_, eps, eps_, N)

# simulation tresholds

crit_1 = 0.01
crit_2 = 0.01
t_max = 50

# Algorithm

conv_1 = crit_1 + 10.0
while conv_1 > crit_1:
    msg = DirectedMessages(q, n, N * np.array(p), G)
    msg.update_marginals()
    msg.update_field()
    
    conv_2 = crit_2 + 10.0
    t = 0
    while conv_2 > crit_2 and t < t_max:
        conv_2 = msg.update_messages()
        t += 1
    temp_n = n_.copy()
    temp_p = p_.copy()
    n_, p_ = msg.get_new_parameters()
    conv_1 = (np.abs(n_ - temp_n).sum() + N * np.abs(p_ - temp_p).sum()) / (temp_n.sum() + temp_p.sum())
    print(conv_1)

overlap = get_real_overlap(msg.get_marginals(), G, n)  # real overlap
overlap_ = msg.get_overlap()  # theoretical overlap

print("example;" + str(N) + ";" + str(q) + ";" + str(c) + ";" + str(eps) + ";" + str(eps_) + ";"
      + str(overlap) + ";" + str(overlap_))
