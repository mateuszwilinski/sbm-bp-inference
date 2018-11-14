# -*- coding: utf-8 -*-
import numpy as np
import itertools as it

# functions


def get_real_overlap(marginals, G, n):
    N = marginals.shape[0]
    q = marginals.shape[1]
    perms = np.array(list(it.permutations(np.arange(q))))
    perms = perms[:, np.argmax(marginals, axis=1)]
    sum_v = np.sum(perms == G.vs['group'], axis=1, dtype='float').max() / N
    max_n = np.max(n)
    return (sum_v - max_n) / (1.0 - max_n)


def affinity(q, c, n, eps, N):
    p_in = c / (N * (eps + (1.0 - eps) * np.sum(n**2)))
    p_out = eps * p_in
    p = p_out * np.ones(q) + (p_in - p_out) * np.eye(q)
    return p


def directed_affinity(q, c, n, eps, eps_, N):
    p_in = c / (2.0 * N * (eps + (1.0 - eps) * np.sum(n**2)))
    p_out = 2.0 * eps * p_in / (eps_ + 1.0)
    p_out_ = eps_ * p_out
    p = p_in * np.eye(q)
    for i in range(1, q):
        off_diag = np.diag(np.ones(q - i), i)
        if i % 2 == 0:
            p = p + off_diag * p_out + off_diag.T * p_out_
        else:
            p = p + off_diag.T * p_out + off_diag * p_out_
    return p


def random_marginals(N, q, n):
    groups = np.zeros((N, q))
    groups[np.arange(N), np.random.choice(len(n), N, p=n)] = 1.0
    return groups
