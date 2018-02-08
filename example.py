# -*- coding: utf-8 -*-

# Libraries, classes and functions

import numpy as np
import igraph as ig
import itertools as it

from messages import Messages
from functionsSBM import SBM, getRealOverlap, affinity

# Parameters

N = 300 # number of nodes
q = 2 # number of blocks
c = 3 # average degree
eps = 0.1 # order parameter
eps_ = 0.1 # starting epsilon

n = np.ones(q)/q
p = affinity(q,c,n,eps,N)

# Graph

G = SBM(n,p,N)

# Initial parameters

q_ = q
c_ = np.mean(G.degree())/2.0

n_ = np.ones(q_)/q_
p_ = affinity(q_,c_,n_,eps_,N)

# simulation tresholds

crit_1 = 0.1*(1 + c_*q_*q_)
crit_2 = 0.1
t_max = 50

# Time check

import time # lets check how fast the code is

start = time.time()

# Algorithm

conv_1 = crit_1 + 10.0
while(conv_1 > crit_1):
    msg = Messages(q_,n_,N*np.array(p_),G)
    msg.updateMarginals()
    msg.updateField()
    
    conv_2 = crit_2 + 10.0
    t = 0
    while(conv_2 > crit_2 and t < t_max):
        conv_2 = msg.updateMessages()
        t += 1
    temp_n = n_.copy()
    temp_p = p_.copy()
    n_,p_ = msg.getNewParameters()
    conv_1 = np.abs(n_ - temp_n).sum() + N*np.abs(p_ - temp_p).sum()

overlap = getRealOverlap(msg.getMarginals(),G,n_) # real overlap
overlap_ = msg.getOverlap() # theoretical overlap

end = time.time()

print("example;" + str(N) + ";" + str(q) + ";" + str(c) + ";" + str(eps) + ";" + str(eps_) + ";" + str(overlap) + ";" + str(overlap_) + ";" + str(end-start))
