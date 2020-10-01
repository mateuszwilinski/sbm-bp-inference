# -*- coding: utf-8 -*-
import numpy as np
import math


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
        self.h = np.zeros(q)  # q-component auxiliary field
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
        e_jk = np.array([self.G.es[self.G.get_eid(x.index, v.index)]['msg']  # incoming messages
                         for x in self.G.vs[v.index].neighbors('in')])
        v['mar'] = self.n.copy()
        if e_jk.size:
            temp_v = np.log(np.dot(e_jk, self.p)).sum(0) + self.h
            if np.isinf(temp_v.max()):  # if all are equal to -inf
                v['mar'] *= np.exp(self.h)
            else:
                temp_v -= temp_v.max()
                v['mar'] *= np.exp(temp_v)  # eq. 28 [undir]
        v['mar'] = v['mar'] / np.sum(v['mar'])  # normalization

    def update_marginals(self):
        for v in self.G.vs:
            self.update_marginal(v)

    def update_field(self):
        self.h = -np.dot(np.array(self.G.vs['mar']), self.p).sum(0) / self.N

    def update_messages(self):
        conv = 0.0  # conversion
        for e in self.G.es:
            # we start with updating the message
            old_msg = e['msg'].copy()
            e_jk = np.array([self.G.es[self.G.get_eid(x.index, e.source)]['msg']
                             for x in self.G.vs[e.source].neighbors('in')
                             if x != self.G.vs[e.target]])
            e['msg'] = self.n.copy()
            if e_jk.size:
                temp_e = np.log(np.dot(e_jk, self.p)).sum(0) + self.h
                if np.isinf(temp_e.max()):  # if all are equal to -inf
                    e['msg'] *= np.exp(self.h)
                else:
                    temp_e -= temp_e.max()
                    e['msg'] *= np.exp(temp_e)  # eq. 26
            e['msg'] = e['msg'] / np.sum(e['msg'])  # normalization
            conv += np.abs(e['msg'] - old_msg).sum()  # updating convergence
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v['mar'].copy()
            self.update_marginal(v)
            # finally we update the auxiliary field
            self.h += (np.dot(old_mar, self.p) - np.dot(v['mar'], self.p)) / self.N
        conv /= self.M
        return conv

    def update_zv(self):
        for v in self.G.vs:
            e_jk = np.array([self.G.es[self.G.get_eid(x.index, v.index)]['msg']  # incoming messages
                             for x in self.G.vs[v.index].neighbors('in')])
            temp_prod = np.zeros(self.q)
            if e_jk.size:  # TODO: dodac sprawdzanie czy temp_prod jest ok, na wzor update'u messages i marginals
                temp_prod = np.log(np.dot(e_jk, self.p)).sum(0)
            self.Z_v[v.index] = np.sum(np.exp(temp_prod + np.log(self.n) + self.h))  # eq. 31

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


class DirectedMessages(Messages):
    def __init__(self, q, n, p, g):
        super(DirectedMessages, self).__init__(q, n, p, g)

    def init_graph(self):
        self.G.es['direct'] = 0  # storing info about the origial graph connections
        for e in self.G.es:
            if e.source < e.target:
                e['direct'] = 1
            else:
                e['direct'] = 2
        self.G.to_undirected(combine_edges=dict(direct=sum))
        self.G.to_directed()

    def update_marginal(self, v):
        e_jk = [self.G.es[self.G.get_eid(x.index, v.index)]  # incoming messages
                for x in self.G.vs[v.index].neighbors('in')]
        temp_v = self.h.copy()
        if e_jk:
            for m in e_jk:
                if m['direct'] == 3:  # if there was an edge in both direction
                    temp_v += np.log(np.dot(m['msg'], self.p * self.p.T))
                elif 1.5 + np.sign(m.source - m.target) / 2.0 == m['direct']:  # if there was same direction edge
                    temp_v += np.log(np.dot(m['msg'], self.p))
                else:
                    temp_v += np.log(np.dot(m['msg'], self.p.T))
            if np.isinf(temp_v.max()):  # if all are equal to -inf
                temp_v = self.h.copy()
            else:
                temp_v -= temp_v.max()
        v['mar'] = np.exp(temp_v) * self.n  # eq. 28 [undir]
        v['mar'] = v['mar'] / np.sum(v['mar'])  # normalization

    def update_field(self):
        self.h = -np.dot(np.array(self.G.vs['mar']), self.p + self.p.T).sum(0) / self.N

    def update_messages(self):
        conv = 0.0  # conversion
        conv_denominator = np.array(self.G.es['msg']).sum()
        for e in self.G.es:
            # we start with updating the message
            old_msg = e['msg'].copy()
            e_jk = [self.G.es[self.G.get_eid(x.index, e.source)]
                    for x in self.G.vs[e.source].neighbors('in')
                    if x != self.G.vs[e.target]]
            temp_e = self.h.copy()
            if e_jk:
                for m in e_jk:
                    if m['direct'] == 3:  # if there was an edge in both direction
                        temp_e += np.log(np.dot(m['msg'], self.p * self.p.T))
                    elif 1.5 + np.sign(m.source - m.target) / 2.0 == m['direct']:  # if there was same direction edge
                        temp_e += np.log(np.dot(m['msg'], self.p))
                    else:
                        temp_e += np.log(np.dot(m['msg'], self.p.T))
                if np.isinf(temp_e.max()):  # if all are equal to -inf
                    temp_e = self.h.copy()
                else:
                    temp_e -= temp_e.max()
            e['msg'] = np.exp(temp_e) * self.n  # eq. 26
            e['msg'] = e['msg'] / np.sum(e['msg'])  # normalization
            conv += np.abs(e['msg'] - old_msg).sum()  # updating convergence
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v['mar'].copy()
            self.update_marginal(v)
            # finally we update the auxiliary field
            self.h += (np.dot(old_mar, self.p + self.p.T) - np.dot(v['mar'], self.p + self.p.T)) / self.N
        conv /= conv_denominator
        return conv

    def update_zv(self):
        for v in self.G.vs:
            e_jk = [self.G.es[self.G.get_eid(x.index, v.index)]  # incoming messages
                    for x in self.G.vs[v.index].neighbors('in')]
            temp_prod = np.zeros(self.q)
            if e_jk:  # TODO: dodac sprawdzanie czy temp_prod jest ok, na wzor update'u messages i marginals
                for m in e_jk:
                    if m['direct'] == 3:  # if there was an edge in both direction
                        temp_prod += np.log(np.dot(m['msg'], self.p * self.p.T))
                    elif 1.5 + np.sign(m.source - m.target) / 2.0 == m['direct']:  # if there was same direction edge
                        temp_prod += np.log(np.dot(m['msg'], self.p))
                    else:
                        temp_prod += np.log(np.dot(m['msg'], self.p.T))
            self.Z_v[v.index] = np.sum(np.exp(temp_prod + np.log(self.n) + self.h))  # eq. 31

    def update_ze(self):
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            if (e['direct'] == 3) or (1.5 + np.sign(e.source - e.target) / 2.0 == e['direct']):
                self.Z_e[e.index] = e['msg'].dot(self.p).dot(e_['msg'])

    def get_new_parameters(self):
        new_n = self.get_marginals().sum(0) / self.N  # eq. 34 [undir]
        new_p = 0.0
        self.update_ze()  # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            if (e['direct'] == 3) or (1.5 + np.sign(e.source - e.target) / 2.0 == e['direct']):
                new_p += self.p * np.dot(e['msg'][:, None], e_['msg'][None, :]) / self.Z_e[e.index]
        new_p /= np.dot(new_n[:, None], new_n[None, :]) * self.N * self.N
        return new_n, new_p


class DegMessages(Messages):
    def __init__(self, q, n, p, g):
        super(DegMessages, self).__init__(q, n, p, g)
        self.theta = np.array(g.degree())  # degree correction

    def update_marginal(self, v):
        e_jk = np.array([self.G.es[self.G.get_eid(x.index, v.index)]['msg'] *
                         self.theta[x.index] * self.theta[v.index]
                         for x in self.G.vs[v.index].neighbors('in')])  # connected messages
        v['mar'] = self.n.copy()
        if e_jk.size:
            temp_v = np.log(np.dot(e_jk, self.p)).sum(0) + self.h * self.theta[v.index]
            if np.isinf(temp_v.max()):  # if all are equal to -inf
                v['mar'] *= np.exp(self.h * self.theta[v.index])
            else:
                temp_v -= temp_v.max()
                v['mar'] *= np.exp(temp_v)  # eq. 28 [undir]
        v['mar'] = v['mar'] / np.sum(v['mar'])  # normalization

    def update_field(self):
        self.h = -np.dot(np.array(self.G.vs['mar']), self.p).T.dot(self.theta) / self.N

    def update_messages(self):
        conv = 0.0  # conversion
        for e in self.G.es:
            # we start with updating the message
            old_msg = e['msg'].copy()
            e_jk = np.array([self.G.es[self.G.get_eid(x.index, e.source)]['msg'] *
                             self.theta[x.index] * self.theta[e.source]
                             for x in self.G.vs[e.source].neighbors('in')
                             if x != self.G.vs[e.target]])
            e['msg'] = self.n.copy()
            if e_jk.size:
                temp_e = np.log(np.dot(e_jk, self.p)).sum(0) + self.h * self.theta[e.source]
                if np.isinf(temp_e.max()):  # if all are equal to -inf
                    e['msg'] *= np.exp(self.h * self.theta[e.source])
                else:
                    temp_e -= temp_e.max()
                    e['msg'] *= np.exp(temp_e)  # eq. 26
            e['msg'] = e['msg'] / np.sum(e['msg'])  # normalization
            conv += np.abs(e['msg'] - old_msg).sum()  # updating convergence
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v['mar'].copy()
            self.update_marginal(v)
            # finally we update the auxiliary field
            self.h += (np.dot(old_mar, self.p) - np.dot(v['mar'], self.p)) * self.theta[v.index] / self.N
        conv /= self.M
        return conv

    def update_zv(self):
        for v in self.G.vs:
            e_jk = np.array([self.G.es[self.G.get_eid(x.index, v.index)]['msg'] *
                             self.theta[x.index] * self.theta[v.index]
                             for x in self.G.vs[v.index].neighbors('in')])  # connected messages
            temp_prod = np.ones(self.q)
            if e_jk.size:
                temp_prod = np.dot(e_jk, self.p).prod(0)
            self.Z_v[v.index] = np.sum(temp_prod * self.n * np.exp(self.h * self.theta[v.index]))

    def update_ze(self):
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            self.Z_e[e.index] = e['msg'].dot(self.p).dot(e_['msg']) * self.theta[e.source] * self.theta[e.target]

    def get_new_parameters(self):
        new_n = self.get_marginals().sum(0) / self.N  # eq. 34 [undir]
        new_n_deg = self.theta.dot(self.get_marginals()) / self.N
        new_p = 0.0
        self.update_ze()  # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            new_p += (np.dot(e['msg'][:, None], e_['msg'][None, :]) *
                      self.theta[e.source] * self.theta[e.target]) / self.Z_e[e.index]
        new_p *= self.p / (np.dot(new_n_deg[:, None], new_n_deg[None, :]) * self.N * self.N)
        return new_n, new_p


class DegDirectedMessages(DirectedMessages):
    def __init__(self, q, n, p, g):
        self.theta = np.array([g.outdegree(), g.indegree()])  # degree correction
        self.h_ = np.zeros(q)
        super(DegDirectedMessages, self).__init__(q, n, p, g)

    def update_marginal(self, v):
        e_jk = [self.G.es[self.G.get_eid(x.index, v.index)]  # connected messages
                for x in self.G.vs[v.index].neighbors('in')]
        temp_v = self.h * self.theta[1, v.index] + self.h_ * self.theta[0, v.index]
        if e_jk:
            for m in e_jk:
                if m['direct'] == 3:  # if there was an edge in both direction
                    temp_v += np.log(np.dot(m['msg'], self.p * self.p.T))
                    temp_v += np.log(self.theta[0, m.source] * self.theta[1, m.target] *
                                     self.theta[0, m.target] * self.theta[1, m.source])
                elif 1.5 + np.sign(m.source - m.target) / 2.0 == m['direct']:  # if there was same direction edge
                    temp_v += np.log(np.dot(m['msg'], self.p))
                    temp_v += np.log(self.theta[0, m.source] * self.theta[1, m.target])
                else:
                    temp_v += np.log(np.dot(m['msg'], self.p.T))
                    temp_v += np.log(self.theta[0, m.target] * self.theta[1, m.source])
        if np.isinf(temp_v.max()):  # if all are equal to -inf
            temp_v = np.zeros(self.q)
        else:
            temp_v -= temp_v.max()
        v['mar'] = np.exp(temp_v) * self.n
        v['mar'] = v['mar'] / np.sum(v['mar'])  # normalization

    def update_field(self):
        self.h = -np.dot(np.array(self.G.vs['mar']), self.p).T.dot(self.theta[0, :]) / self.N
        self.h_ = -np.dot(np.array(self.G.vs['mar']), self.p.T).T.dot(self.theta[1, :]) / self.N

    def update_messages(self):
        conv = 0.0  # conversion
        conv_denominator = np.array(self.G.es['msg']).sum()
        for e in self.G.es:
            # we start with updating the message
            old_msg = e['msg'].copy()
            e_jk = [self.G.es[self.G.get_eid(x.index, e.source)]
                    for x in self.G.vs[e.source].neighbors('in')
                    if x != self.G.vs[e.target]]
            temp_e = (self.h * self.theta[1, e.source] + self.h_ * self.theta[0, e.source])
            if e_jk:
                for m in e_jk:
                    if m['direct'] == 3:  # if there was an edge in both direction
                        temp_e += np.log(np.dot(m['msg'], self.p * self.p.T))
                        temp_e += np.log(self.theta[0, m.source] * self.theta[1, m.target] *
                                         self.theta[0, m.target] * self.theta[1, m.source])
                    elif 1.5 + np.sign(m.source - m.target) / 2.0 == m['direct']:  # if there was same direction edge
                        temp_e += np.log(np.dot(m['msg'], self.p))
                        temp_e += np.log(self.theta[0, m.source] * self.theta[1, m.target])
                    else:
                        temp_e += np.log(np.dot(m['msg'], self.p.T))
                        temp_e += np.log(self.theta[0, m.target] * self.theta[1, m.source])
            if np.isinf(temp_e.max()):  # if all are equal to -inf
                temp_e = np.zeros(self.q)  # zeros because it is a logarithm
            else:
                temp_e -= temp_e.max()
            e['msg'] = np.exp(temp_e) * self.n
            e['msg'] = e['msg'] / np.sum(e['msg'])  # normalization
            conv += np.abs(e['msg'] - old_msg).sum()  # updating convergence
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v['mar'].copy()
            self.update_marginal(v)
            # finally we update the auxiliary field
            self.h += (np.dot(old_mar, self.p) - np.dot(v['mar'], self.p)) * self.theta[0, v.index] / self.N
            self.h_ += (np.dot(old_mar, self.p.T) - np.dot(v['mar'], self.p.T)) * self.theta[1, v.index] / self.N
        conv /= conv_denominator
        return conv

    def update_zv(self):  # TODO: tu trzeba dodac jakies logarytmy, poniewaz to sie wykrzaczy dla duzych stopni
        for v in self.G.vs:
            e_jk = [self.G.es[self.G.get_eid(x.index, v.index)]  # connected messages
                    for x in self.G.vs[v.index].neighbors('in')]
            temp_prod = np.ones(self.q)
            if e_jk:
                for m in e_jk:
                    if m['direct'] == 3:  # if there was an edge in both direction
                        temp_prod *= np.dot(m['msg'], self.p * self.p.T)
                        temp_prod *= (self.theta[0, m.source] * self.theta[1, m.target] *
                                      self.theta[0, m.target] * self.theta[1, m.source])
                    elif 1.5 + np.sign(m.source - m.target) / 2.0 == m['direct']:  # if there was same direction edge
                        temp_prod *= np.dot(m['msg'], self.p)
                        temp_prod *= self.theta[0, m.source] * self.theta[1, m.target]
                    else:
                        temp_prod *= np.dot(m['msg'], self.p.T)
                        temp_prod *= self.theta[0, m.target] * self.theta[1, m.source]
            self.Z_v[v.index] = np.sum(temp_prod * self.n * np.exp(self.h * self.theta[1, v.index] +
                                                                   self.h_ * self.theta[0, v.index]))

    def update_ze(self):
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            if (e['direct'] == 3) or (1.5 + np.sign(e.source - e.target) / 2.0 == e['direct']):
                self.Z_e[e.index] = e['msg'].dot(self.p).dot(e_['msg'])
                self.Z_e[e.index] *= self.theta[0, e.source] * self.theta[1, e.target]

    def get_new_parameters(self):
        new_n = self.get_marginals().sum(0) / self.N  # eq. 34 [undir]
        new_n_out = self.theta[0, :].dot(self.get_marginals()) / self.N
        new_n_in = self.theta[1, :].dot(self.get_marginals()) / self.N
        new_p = 0.0
        self.update_ze()  # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            if (e['direct'] == 3) or (1.5 + np.sign(e.source - e.target) / 2.0 == e['direct']):
                new_p += (self.p * np.dot(e['msg'][:, None], e_['msg'][None, :]) *
                          self.theta[0, e.source] * self.theta[1, e.target]) / self.Z_e[e.index]
        new_p /= np.dot(new_n_out[:, None], new_n_in[None, :]) * self.N * self.N
        return new_n, new_p


class PoissonMessages(Messages):
    def __init__(self, q, n, p, g):
        super(PoissonMessages, self).__init__(q, n, p, g)
        self.p = np.array(p) / self.N

    def init_graph(self):
        self.G.to_directed()
        if np.logical_not(self.G.is_weighted()):
            self.G.es['weight'] = np.ones(self.G.ecount())

    def update_marginal(self, v):
        e_jk = []  # incoming messages
        v_j = []  # neighboring nodes
        for x in self.G.vs[v.index].neighbors('in'):
            e_jk.append(self.G.es[self.G.get_eid(x.index, v.index)])
            v_j.append(x['mar'])
        temp_v = self.h.copy()
        if e_jk:
            for m in e_jk:
                temp_v += (np.log(np.dot(m['msg'], self.p ** m['weight'] * np.exp(-self.p))) -
                           math.lgamma(m['weight']))
            temp_v -= np.log(np.dot(np.array(v_j), np.exp(-self.p))).sum(0)
            if np.isinf(temp_v.max()):  # if all are equal to -inf
                temp_v = self.h.copy()
            else:
                temp_v -= temp_v.max()
        v['mar'] = np.exp(temp_v) * self.n
        v['mar'] = v['mar'] / np.sum(v['mar'])  # normalization

    def update_field(self):
        self.h = np.log(np.dot(np.array(self.G.vs['mar']), np.exp(-self.p))).sum(0)

    def update_messages(self):
        conv = 0.0  # conversion
        conv_denominator = np.array(self.G.es['msg']).sum()
        for e in self.G.es:
            # we start with updating the message
            old_msg = e['msg'].copy()
            e_jk = []  # incoming messages
            v_j = []  # neighboring nodes
            for x in self.G.vs[e.source].neighbors('in'):
                if x != self.G.vs[e.target]:
                    e_jk.append(self.G.es[self.G.get_eid(x.index, e.source)])
                    v_j.append(x['mar'])
            temp_e = self.h.copy()
            if e_jk:
                for m in e_jk:
                    temp_e += (np.log(np.dot(m['msg'], self.p ** m['weight'] * np.exp(-self.p))) -
                               math.lgamma(m['weight']))
                temp_e -= np.log(np.dot(np.array(v_j), np.exp(-self.p))).sum(0)
                if np.isinf(temp_e.max()):  # if all are equal to -inf
                    temp_e = self.h.copy()
                else:
                    temp_e -= temp_e.max()
            e['msg'] = np.exp(temp_e) * self.n  # eq. 26
            e['msg'] = e['msg'] / np.sum(e['msg'])  # normalization
            conv += np.abs(e['msg'] - old_msg).sum()  # updating convergence
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v['mar'].copy()
            self.update_marginal(v)
            # finally we update the auxiliary field
            self.h += np.log(np.dot(v['mar'], np.exp(-self.p))) - np.log(np.dot(old_mar, np.exp(-self.p)))
        conv /= conv_denominator
        return conv

    def update_zv(self):
        for v in self.G.vs:
            e_jk = []  # incoming messages
            v_j = []  # neighboring nodes
            for x in self.G.vs[v.index].neighbors('in'):
                e_jk.append(self.G.es[self.G.get_eid(x.index, v.index)])
                v_j.append(x['mar'])
            temp_prod = np.zeros(self.q)
            if e_jk:  # TODO: dodac sprawdzanie czy temp_prod jest ok, na wzor update'u messages i marginals
                for m in e_jk:
                    temp_prod += (np.log(np.dot(m['msg'], self.p ** m['weight'] * np.exp(-self.p))) -
                                  math.lgamma(m['weight']))
                temp_prod += -np.log(np.dot(np.array(v_j), np.exp(-self.p)))
            self.Z_v[v.index] = np.sum(np.exp(temp_prod + np.log(self.n) + self.h))

    def update_ze(self):
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            self.Z_e[e.index] = (e['msg'].dot(self.p ** e['weight'] * np.exp(-self.p)).dot(e_['msg']) /
                                 math.gamma(e['weight']))

    def get_new_parameters(self):
        new_n = self.get_marginals().sum(0) / self.N  # eq. 34 [undir]
        new_p = 0.0
        self.update_ze()  # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            new_p += (e['weight'] * self.p ** e['weight'] * np.dot(e['msg'][:, None], e_['msg'][None, :]) /
                      self.Z_e[e.index]) / math.gamma(e['weight'])
        new_p *= np.exp(-self.p) / (np.dot(new_n[:, None], new_n[None, :]) * self.N * self.N)
        return new_n, new_p


class DegPoissMessages(PoissonMessages):
    def __init__(self, q, n, p, g):
        super(DegPoissMessages, self).__init__(q, n, p, g)
        self.theta = np.zeros((self.N, 2), dtype='int')  # degree correction
        self.theta[:, 0] = self.G.degree()
        self.degrees, self.theta[:, 1] = np.unique(self.theta[:, 0], return_inverse=True)
        self.h = np.zeros((self.degrees.shape[0], q))  # q-component auxiliary field

    def update_marginal(self, v):
        e_jk = []  # incoming messages
        v_j = [v]  # neighboring nodes
        for x in self.G.vs[v.index].neighbors('in'):
            e_jk.append(self.G.es[self.G.get_eid(x.index, v.index)])
            v_j.append(x)
        temp_v = self.h[self.theta[v.index, 1]].copy()
        # temp_v = self.h * self.theta[v.index, 0]
        if e_jk:
            for m in e_jk:
                temp_v += np.log(np.dot(m['msg'], self.p ** m['weight'] *
                                        np.exp(-self.theta[m.source, 0] * self.theta[m.target, 0] * self.p)))
                temp_v += m['weight'] * (np.log(self.theta[m.source, 0]) + np.log(self.theta[m.target, 0]))
                temp_v -= math.lgamma(m['weight'])
                # temp_v -= np.log(np.dot(m['msg'], np.exp(-self.theta[m.source, 0] *
                #                                          self.theta[m.target, 0] * self.p)))
            for x in v_j:  # TODO: czy ponizsza linijka jest poprawna?
                temp_v -= np.log(np.dot(x['mar'], np.exp(-self.theta[x.index, 0] * self.theta[v.index, 0] * self.p)))
            if np.isinf(temp_v.max()):  # if all are equal to -inf
                temp_v = self.h[self.theta[v.index, 1]].copy()
                # temp_v = self.h * self.theta[v.index, 0]
            else:
                temp_v -= temp_v.max()
        v['mar'] = np.exp(temp_v) * self.n
        v['mar'] = v['mar'] / np.sum(v['mar'])  # normalization

    # def update_field(self):
    #     self.h = -np.dot(np.array(self.G.vs['mar']), self.p).T.dot(self.theta[:, 0])

    def update_field(self):
        for v in self.G.vs:
            self.h += np.log(np.einsum('...i,...ij', v['mar'],
                                       np.exp(-np.multiply.outer(self.degrees, self.theta[v.index, 0] * self.p))))

    def update_field_by_marginal(self, v, old_mar):
        self.h -= np.log(np.einsum('...i,...ij', old_mar,
                                   np.exp(-np.multiply.outer(self.degrees, self.theta[v.index, 0] * self.p))))
        self.h += np.log(np.einsum('...i,...ij', v['mar'],
                                   np.exp(-np.multiply.outer(self.degrees, self.theta[v.index, 0] * self.p))))

    def update_messages(self):
        conv = 0.0  # conversion
        conv_denominator = np.array(self.G.es['msg']).sum()
        for e in self.G.es:
            # we start with updating the message
            old_msg = e['msg'].copy()
            e_jk = []  # incoming messages
            v_j = [self.G.vs[e.source]]  # neighboring nodes
            for x in self.G.vs[e.source].neighbors('in'):
                v_j.append(x)
                if x != self.G.vs[e.target]:
                    e_jk.append(self.G.es[self.G.get_eid(x.index, e.source)])
            temp_e = self.h[self.theta[e.source, 1]].copy()
            # temp_e = self.h * self.theta[e.source, 0]
            if e_jk:
                for m in e_jk:
                    temp_e += np.log(np.dot(m['msg'], self.p ** m['weight'] *
                                            np.exp(-self.theta[m.source, 0] * self.theta[m.target, 0] * self.p)))
                    temp_e += m['weight'] * (np.log(self.theta[m.source, 0]) + np.log(self.theta[m.target, 0]))
                    temp_e -= math.lgamma(m['weight'])
                    # temp_e -= np.log(np.dot(m['msg'], np.exp(-self.theta[m.source, 0] *
                    #                                          self.theta[m.target, 0] * self.p)))
                for x in v_j:  # TODO: czy ponizsza linijka jest poprawna?
                    temp_e -= np.log(np.dot(x['mar'], np.exp(-self.theta[x.index, 0] *
                                                             self.theta[e.source, 0] * self.p)))
                if np.isinf(temp_e.max()):  # if all are equal to -inf
                    temp_e = self.h[self.theta[e.source, 1]].copy()
                    # temp_e = self.h * self.theta[e.source, 0]
                else:
                    temp_e -= temp_e.max()
            e['msg'] = np.exp(temp_e) * self.n  # eq. 26
            e['msg'] = e['msg'] / np.sum(e['msg'])  # normalization
            conv += np.abs(e['msg'] - old_msg).sum()  # updating convergence
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v['mar'].copy()
            self.update_marginal(v)
            # finally we update the auxiliary field
            self.update_field_by_marginal(v, old_mar)
            # self.h += (np.dot(old_mar, self.p) - np.dot(v['mar'], self.p)) * self.theta[v.index, 0]
        conv /= conv_denominator
        return conv

    def update_zv(self):  # TODO: Brakuje dzielenia! Patrz doktorat Decelle.
        for v in self.G.vs:
            e_jk = []  # incoming messages
            v_j = []  # neighboring nodes
            for x in self.G.vs[v.index].neighbors('in'):
                e_jk.append(self.G.es[self.G.get_eid(x.index, v.index)])
                v_j.append(x['mar'])
            temp_prod = np.zeros(self.q)
            if e_jk:  # TODO: dodac sprawdzanie czy temp_prod jest ok, na wzor update'u messages i marginals
                for m in e_jk:
                    temp_prod += np.log(np.dot(m['msg'], self.p ** m['weight'] *
                                               np.exp(-self.theta[m.source, 0] * self.theta[m.target, 0] * self.p)))
                    temp_prod += m['weight'] * (np.log(self.theta[m.source, 0]) + np.log(self.theta[m.target, 0]))
                    temp_prod -= math.lgamma(m['weight'])
                for x in v_j:  # TODO: czy ponizsza linijka jest poprawna?
                    temp_prod -= np.log(np.dot(x['mar'], np.exp(-self.theta[x.index, 0] *
                                                                self.theta[v.index, 0] * self.p)))
            self.Z_v[v.index] = np.sum(np.exp(temp_prod + np.log(self.n) + self.h[self.theta[v.index, 1]]))

    def update_ze(self):
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            self.Z_e[e.index] = e['msg'].dot(self.p ** e['weight'] *
                                             np.exp(-self.theta[e.source, 0] * self.theta[e.target, 0] *
                                                    self.p)).dot(e_['msg'])
            self.Z_e[e.index] *= ((self.theta[e.source, 0] * self.theta[e.target, 0]) ** e['weight'] /
                                  math.gamma(e['weight']))
            # self.Z_e[e.index] /= e['msg'].dot(np.exp(-self.theta[e.source, 0] * self.theta[e.target, 0] * self.p)
            #                                   ).dot(e_['msg'])

    def get_new_parameters(self):
        new_n = self.get_marginals().sum(0) / self.N  # eq. 34 [undir]
        new_n_deg = self.theta[:, 0].dot(self.get_marginals())
        new_p = 0.0
        self.update_ze()  # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            new_p += ((self.theta[e.source, 0] * self.theta[e.target, 0] * self.p) ** e['weight'] *
                      np.exp(-self.theta[e.source, 0] * self.theta[e.target, 0] * self.p) *
                      np.dot(e['msg'][:, None], e_['msg'][None, :]) * e['weight'] /
                      self.Z_e[e.index] / math.gamma(e['weight']))
        # new_p /= (np.dot(new_n[:, None], new_n[None, :]) * self.N * self.N)
        new_p /= np.dot(new_n_deg[:, None], new_n_deg[None, :])
        return new_n, new_p
