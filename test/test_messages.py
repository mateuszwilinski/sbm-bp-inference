import unittest as ut
import numpy as np
import igraph as ig
from sbm_bp import Messages, get_real_overlap, affinity


class TestMessages(ut.TestCase):
    def test_overlap(self):
        N = 500  # number of nodes
        q = 2  # number of blocks
        c = 3  # average degree
        eps = 0.7  # order parameter

        crit = 0.01
        t_max = 50

        n = np.ones(q) / q
        p = affinity(q, c, n, eps, N)
        G = ig.Graph.SBM(N, p.tolist(), (N * n).tolist(), directed=False)

        groups = np.zeros(N, dtype='int')
        for i in range(q - 1):
            groups[int(N * np.sum(n[:(i + 1)])):] = i + 1
        G.vs['group'] = groups

        msg = Messages(q, n, N * np.array(p), G)
        msg.update_marginals()
        msg.update_field()

        conv = crit + 10.0
        t = 0
        while conv > crit and t < t_max:
            conv = msg.update_messages()
            t += 1

        self.assertAlmostEqual(msg.get_overlap(), 0.9, places=1)
