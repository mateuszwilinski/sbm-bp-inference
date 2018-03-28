# -*- coding: utf-8 -*-
import numpy as np

class Messages(object):
    def __init__(self, q, n, p, g):
        self.q = q  # number of blocks
        self.n = np.array(n)  # vector of block probabilities
        self.p = np.array(p)  # matrix of intra-inter block edges probabilities times N
        self.G = g.copy()  # known network (igraph object)
        self.init_graph()  # change G into the network of messages
        self.N = self.G.vcount()  # number of nodes
        self.M = self.G.ecount()  # number of messages
        self.init_messages()  # message functions
        self.init_marginals()  # marginal probabilities
        self.e_h = np.ones(q)  # q-component auxiliary field
        self.Z_v = np.ones(self.N)  # defined in eq. 31 [undir]
        self.Z_e = np.ones(self.M)  # defined in eq. 29 [undir]

    def init_graph(self):
        self.G.to_directed()

    def init_marginals(self):
        self.G.vs['mar'] = np.ones((self.N, self.q))

    def init_messages(self):
        msgs = np.random.rand(self.M, self.q)  # message functions
        msgs = msgs / msgs.sum(1)[:, None]
        self.G.es['msg'] = msgs

    def update_marginal(self, v):
        e_jk = np.array([self.G.es[self.G.get_eid(x.index, v.index)]['msg']  # connected messages
                         for x in self.G.vs[v.index].neighbors('in')])
        v['mar'] = self.n * self.e_h
        if e_jk.size:
            temp_v = np.log(np.dot(e_jk, self.p)).sum(0)
            if not np.isinf(temp_v.max()):  # if all are not equal to -inf
                temp_v -= temp_v.max()
                v['mar'] *= np.exp(temp_v)  # eq. 28 [undir]
        v['mar'] = v['mar'] / np.sum(v['mar'])  # normalization

    def update_marginals(self):
        for v in self.G.vs:
            self.update_marginal(v)

    def update_field(self):
        h = -np.dot(np.array(self.G.vs['mar']), self.p).sum(0) / self.N
        self.e_h = np.exp(h)

    def update_messages(self):
        conv = 0.0  # conversion
        conv_denominator = np.array(self.G.es['msg']).sum()
        for e in self.G.es:
            # we start with updating the message
            old_msg = e['msg'].copy()
            e_jk = np.array([self.G.es[self.G.get_eid(x.index, e.source)]['msg']
                             for x in self.G.vs[e.source].neighbors('in')
                             if x != self.G.vs[e.target]])
            e['msg'] = self.n * self.e_h
            if e_jk.size:
                temp_e = np.log(np.dot(e_jk, self.p)).sum(0)
                if not np.isinf(temp_e.max()):  # if all are not equal to -inf
                    temp_e -= temp_e.max()
                    e['msg'] *= np.exp(temp_e)  # eq. 26
            e['msg'] = e['msg'] / np.sum(e['msg'])  # normalization
            conv += np.abs(e['msg'] - old_msg).sum()  # updating convergence
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v['mar'].copy()
            self.update_marginal(v)
            # finally we update the auxiliary field
            self.e_h *= np.exp((np.dot(old_mar, self.p) - np.dot(v['mar'], self.p)) / self.N)
        conv /= conv_denominator
        return conv

    def update_zv(self):
        for v in self.G.vs:
            e_jk = np.array([self.G.es[self.G.get_eid(x.index, v.index)]['msg']  # connected messages
                             for x in self.G.vs[v.index].neighbors('in')])
            temp_prod = np.ones(self.q)
            if e_jk.size:
                temp_prod = np.dot(e_jk, self.p).prod(0)
            self.Z_v[v.index] = np.sum(temp_prod * self.n * self.e_h)  # eq. 31

    def update_ze(self):
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            self.Z_e[e.index] = e['msg'].dot(self.p).dot(e_['msg'])  # eq. 29 [undir]

    def get_messages(self):
        return np.array(self.G.es['msg'])

    def get_marginals(self):
        return np.array(self.G.vs['mar'])

    def get_free_energy(self):
        self.update_zv()
        self.update_ze()
        c = np.dot(self.n, self.p).dot(self.n)  # average degree from eq. 1 [undir]
        return (np.sum(np.log(self.Z_e)) - np.sum(np.log(self.Z_v))) / self.N - c / 2.0

    def get_assignments(self):
        return self.get_marginals().argmax(1) + 1

    def get_overlap(self):
        sum_v = self.get_marginals().max(1).sum() / self.N
        max_n = np.max(self.n)
        return (sum_v - max_n) / (1.0 - max_n)

    def get_new_parameters(self):
        new_n = self.get_marginals().sum(0) / self.N  # eq. 34 [undir]
        new_p = 0.0
        self.update_ze()  # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            new_p += np.dot(e['msg'][:, None], e_['msg'][None, :]) / self.Z_e[e.index]  # eq. 35 [undir]
        new_p *= self.p / (np.dot(new_n[:, None], new_n[None, :]) * self.N * self.N)
        return new_n, new_p

