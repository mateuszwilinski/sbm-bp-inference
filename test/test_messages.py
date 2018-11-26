import unittest as ut
import numpy as np
import igraph as ig
from sbm_bp import Messages, DirectedMessages, PoissonMessages, affinity, directed_affinity


class TestMessages(ut.TestCase):
    N = 1000  # number of nodes
    q = 2  # number of blocks
    c = 3  # average degree

    crit = 0.01
    t_max = 50

    def test_undetectable_undirected(self):
        eps = 0.7  # order parameter

        n = np.ones(self.q) / self.q
        p = affinity(self.q, self.c, n, eps, self.N)
        G = ig.Graph.SBM(self.N, p.tolist(), (self.N * n).tolist(), directed=False)

        groups = np.zeros(self.N, dtype='int')
        for i in range(self.q - 1):
            groups[int(self.N * np.sum(n[:(i + 1)])):] = i + 1
        G.vs['group'] = groups

        msg = Messages(self.q, n, self.N * np.array(p), G)
        msg.update_marginals()
        msg.update_field()

        conv = self.crit + 10.0
        t = 0
        while conv > self.crit and t < self.t_max:
            conv = msg.update_messages()
            t += 1

        self.assertAlmostEqual(msg.get_overlap(), 0.0, places=1)

    def test_detectable_undirected(self):
        eps = 0.1  # order parameter

        n = np.ones(self.q) / self.q
        p = affinity(self.q, self.c, n, eps, self.N)
        G = ig.Graph.SBM(self.N, p.tolist(), (self.N * n).tolist(), directed=False)

        groups = np.zeros(self.N, dtype='int')
        for i in range(self.q - 1):
            groups[int(self.N * np.sum(n[:(i + 1)])):] = i + 1
        G.vs['group'] = groups

        msg = Messages(self.q, n, self.N * np.array(p), G)
        msg.update_marginals()
        msg.update_field()

        conv = self.crit + 10.0
        t = 0
        while conv > self.crit and t < self.t_max:
            conv = msg.update_messages()
            t += 1

        self.assertAlmostEqual(msg.get_overlap(), 0.8, places=1)

    def test_undetectable_directed(self):
        eps = 0.7  # order parameter
        gamma = 0.8  # symmetry order parameter

        n = np.ones(self.q) / self.q
        p = directed_affinity(self.q, self.c, n, eps, gamma, self.N)
        G = ig.Graph.SBM(self.N, p.tolist(), (self.N * n).tolist(), directed=True)

        groups = np.zeros(self.N, dtype='int')
        for i in range(self.q - 1):
            groups[int(self.N * np.sum(n[:(i + 1)])):] = i + 1
        G.vs['group'] = groups

        msg = DirectedMessages(self.q, n, self.N * np.array(p), G)
        msg.update_marginals()
        msg.update_field()

        conv = self.crit + 10.0
        t = 0
        while conv > self.crit and t < self.t_max:
            conv = msg.update_messages()
            t += 1

        self.assertAlmostEqual(msg.get_overlap(), 0.1, places=1)

    def test_detectable_directed(self):
        eps = 0.1  # order parameter
        gamma = 0.8  # symmetry order parameter

        n = np.ones(self.q) / self.q
        p = directed_affinity(self.q, self.c, n, eps, gamma, self.N)
        G = ig.Graph.SBM(self.N, p.tolist(), (self.N * n).tolist(), directed=True)

        groups = np.zeros(self.N, dtype='int')
        for i in range(self.q - 1):
            groups[int(self.N * np.sum(n[:(i + 1)])):] = i + 1
        G.vs['group'] = groups

        msg = DirectedMessages(self.q, n, self.N * np.array(p), G)
        msg.update_marginals()
        msg.update_field()

        conv = self.crit + 10.0
        t = 0
        while conv > self.crit and t < self.t_max:
            conv = msg.update_messages()
            t += 1

        self.assertAlmostEqual(msg.get_overlap(), 0.8, places=1)

    def test_undetectable_poisson(self):
        eps = 0.7  # order parameter

        n = np.ones(self.q) / self.q
        p = affinity(self.q, self.c, n, eps, self.N)
        G = ig.Graph.SBM(self.N, p.tolist(), (self.N * n).tolist(), directed=False)

        groups = np.zeros(self.N, dtype='int')
        for i in range(self.q - 1):
            groups[int(self.N * np.sum(n[:(i + 1)])):] = i + 1
        G.vs['group'] = groups

        msg = PoissonMessages(self.q, n, self.N * np.array(p), G)
        msg.update_marginals()
        msg.update_field()

        conv = self.crit + 10.0
        t = 0
        while conv > self.crit and t < self.t_max:
            conv = msg.update_messages()
            t += 1

        self.assertAlmostEqual(msg.get_overlap(), 0.0, places=1)

    def test_detectable_poisson(self):
        eps = 0.1  # order parameter

        n = np.ones(self.q) / self.q
        p = affinity(self.q, self.c, n, eps, self.N)
        G = ig.Graph.SBM(self.N, p.tolist(), (self.N * n).tolist(), directed=False)

        groups = np.zeros(self.N, dtype='int')
        for i in range(self.q - 1):
            groups[int(self.N * np.sum(n[:(i + 1)])):] = i + 1
        G.vs['group'] = groups

        msg = PoissonMessages(self.q, n, self.N * np.array(p), G)
        msg.update_marginals()
        msg.update_field()

        conv = self.crit + 10.0
        t = 0
        while conv > self.crit and t < self.t_max:
            conv = msg.update_messages()
            t += 1

        self.assertAlmostEqual(msg.get_overlap(), 0.8, places=1)
