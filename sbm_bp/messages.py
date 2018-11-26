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

    def update_marginal_by_message(self, v, msg, old_msg):
        temp_v = np.log(v['mar'])
        temp_v += np.log(np.dot(msg, self.p)) - np.log(np.dot(old_msg, self.p))
        if np.isinf(temp_v.max()):  # if all are equal to -inf
            v['mar'] = np.exp(self.h) * self.n
        else:
            temp_v -= temp_v.max()
            v['mar'] = np.exp(temp_v)  # eq. 28 [undir]
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
            self.update_marginal_by_message(v, e['msg'], old_msg)
            # finally we update the auxiliary field
            self.h += (np.dot(old_mar, self.p) - np.dot(v['mar'], self.p)) / self.N
        conv /= self.M
        return conv

    def update_zv(self):
        for v in self.G.vs:
            e_jk = np.array([self.G.es[self.G.get_eid(x.index, v.index)]['msg']  # incoming messages
                             for x in self.G.vs[v.index].neighbors('in')])
            temp_prod = np.zeros(self.q)
            if e_jk.size:
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

    def update_marginal_by_message(self, v, msg, old_msg):
        temp_v = np.log(v['mar'])
        if msg['direct'] == 3:  # if there was an edge in both direction
            temp_v += (np.log(np.dot(msg['msg'], self.p * self.p.T)) -
                       np.log(np.dot(old_msg, self.p * self.p.T)))
        elif 1.5 + np.sign(msg.source - msg.target) / 2.0 == msg['direct']:  # if there was same direction edge
            temp_v += (np.log(np.dot(msg['msg'], self.p)) -
                       np.log(np.dot(old_msg, self.p)))
        else:
            temp_v += (np.log(np.dot(msg['msg'], self.p.T)) -
                       np.log(np.dot(old_msg, self.p.T)))
        if np.isinf(temp_v.max()):  # if all are equal to -inf
            v['mar'] = np.exp(self.h) * self.n
        else:
            temp_v -= temp_v.max()
            v['mar'] = np.exp(temp_v)  # eq. 28 [undir]
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
            self.update_marginal_by_message(v, e, old_msg)
            # finally we update the auxiliary field
            self.h += (np.dot(old_mar, self.p + self.p.T) - np.dot(v['mar'], self.p + self.p.T)) / self.N
        conv /= conv_denominator
        return conv

    def update_zv(self):
        for v in self.G.vs:
            e_jk = [self.G.es[self.G.get_eid(x.index, v.index)]  # incoming messages
                    for x in self.G.vs[v.index].neighbors('in')]
            temp_prod = np.zeros(self.q)
            if e_jk:
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
            if e['direct'] == 3:
                self.Z_e[e.index] = e['msg'].dot(self.p * self.p.T).dot(e_['msg'])
            elif 1.5 + np.sign(e.source - e.target) / 2.0 == e['direct']:  # if there was same direction edge
                self.Z_e[e.index] = e['msg'].dot(self.p).dot(e_['msg'])

    def get_new_parameters(self):
        new_n = self.get_marginals().sum(0) / self.N  # eq. 34 [undir]
        new_p = 0.0
        self.update_ze()  # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es[self.G.get_eid(e.target, e.source)]  # the opposite message
            if e['direct'] == 3:
                new_p += self.p * self.p.T * np.dot(e['msg'][:, None], e_['msg'][None, :]) / self.Z_e[e.index]
            elif 1.5 + np.sign(e.source - e.target) / 2.0 == e['direct']:  # if there was same direction edge
                new_p += self.p * np.dot(e['msg'][:, None], e_['msg'][None, :]) / self.Z_e[e.index]
        new_p /= np.dot(new_n[:, None], new_n[None, :]) * self.N * self.N
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

    def update_marginal_by_message(self, v, msg, old_msg):
        temp_v = np.log(v['mar'])
        temp_v += (np.log(np.dot(msg['msg'], self.p ** msg['weight'] * np.exp(-self.p))) -
                   np.log(np.dot(old_msg, self.p ** msg['weight'] * np.exp(-self.p))))
        if np.isinf(temp_v.max()):  # if all are equal to -inf
            v['mar'] = np.exp(self.h) * self.n
        else:
            temp_v -= temp_v.max()
            v['mar'] = np.exp(temp_v)
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
            self.update_marginal_by_message(v, e, old_msg)
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
            if e_jk:
                for m in e_jk:
                    temp_prod += (np.log(np.dot(m['msg'], self.p ** m['weight'] * np.exp(-self.p))) -
                                  math.lgamma(m['weight']))
                temp_prod += -np.log(np.dot(np.array(v_j), np.exp(-self.p)))
            self.Z_v[v.index] = np.sum(np.exp(temp_prod + np.log(self.n) + self.h))  # eq. 31

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
            new_p += self.p ** e['weight'] * np.dot(e['msg'][:, None], e_['msg'][None, :]) / self.Z_e[e.index]
        new_p *= np.exp(-self.p) / (np.dot(new_n[:, None], new_n[None, :]) * self.N * self.N)
        return new_n, new_p
