# -*- coding: utf-8 -*-

# Libraries, classes and functions

import numpy as np
import igraph as ig

from sbm_bp import Messages
from sbm_bp import affinity

# Graph

f = "file.txt"  # here should be a file with an edgelist (or you can import it differently)
G = ig.Graph.Read_Edgelist(f, directed=False)
N = G.vcount()
c = np.mean(G.degree())

# Initial parameters

q = 2  # number of blocks
eps = 0.1  # starting epsilon
n = np.ones(q) / q
p = affinity(q, c, n, eps, N)

# simulation tresholds

crit_1 = 0.01
crit_2 = 0.01
t_max = 50

# Algorithm

conv_1 = crit_1 + 10.0
while conv_1 > crit_1:
    msg = Messages(q, n, N * np.array(p), G)
    msg.update_marginals()
    msg.update_field()
    
    conv_2 = crit_2 + 10.0
    t = 0
    while conv_2 > crit_2 and t < t_max:
        conv_2 = msg.update_messages()
        t += 1
    temp_n = n.copy()
    temp_p = p.copy()
    n, p = msg.get_new_parameters()
    conv_1 = (np.abs(n - temp_n).sum() + N * np.abs(p - temp_p).sum()) / (temp_n.sum() + temp_p.sum())
    print(conv_1)  # you can comment this line

overlap = msg.get_overlap()  # theoretical overlap
