# -*- coding: utf-8 -*-
import numpy as np
import igraph as ig

class Messages(object):
    def __init__(self,q,n,p,G):
        self.q = q # number of blocks
        self.n = np.array(n) # vector of block probabilities
        self.G = G.copy() # known network (igraph object)
        self.N = G.vcount() # number of nodes
        self.M = G.ecount() # number of edges
        self.p = np.array(p) # matrix of intra-inter block edges probabilities times N
        self.G.es["msg"] = np.random.rand(self.M,self.q) # message functions
        for e in self.G.es:
            e["msg"] = e["msg"]/np.sum(e["msg"]) # messages normalization
        self.G.vs["mar"] = np.zeros((self.N,self.q)) # marginal probabilities
        self.e_h = np.ones(q) # q-component auxiliary field
        self.Z_v = np.ones(self.N) # defined in eq. 31
        self.Z_e = np.ones(self.M) # defined in eq. 29
    
    def updateMarginal(self,v):
        e_jk = np.array(self.G.es.select(_target = v.index)["msg"]) # connected messages
        if(len(e_jk)):
            temp_v = np.log(np.dot(e_jk,self.p)).sum(0)
            if(np.isinf(temp_v.max())):
                temp_v = np.zeros(self.q) # zeros because it is a logarithm
            else:
                temp_v -= temp_v.max()
            v["mar"] = np.exp(temp_v)*self.n*self.e_h # eq. 28
        else: # this part is only for not connected nodes during initialization
            v["mar"] = self.n*self.e_h
        v["mar"] = v["mar"]/np.sum(v["mar"]) # normalization
    
    def updateMarginals(self):
        for v in self.G.vs:
            self.updateMarginal(v)
    
    def updateField(self):
        h = -np.dot(np.array(self.G.vs["mar"]),self.p).sum(0)/self.N
        self.e_h = np.exp(h)
    
    def updateMessages(self):
        conv = 0.0 # conversion
        for e in self.G.es:
            # we start with updating the message
            old_msg = e["msg"].copy()
            e_jk = np.array(self.G.es.select(_source_ne = e.target,_target = e.source)["msg"])
            if(len(e_jk)):
                temp_e = np.log(np.dot(e_jk,self.p)).sum(0)
                if(np.isinf(temp_e.max())):
                    temp_e = np.zeros(self.q) # zeros because it is a logarithm
                else:
                    temp_e -= temp_e.max()
                e["msg"] = np.exp(temp_e)*self.n*self.e_h # eq. 26
            else:
                e["msg"] = self.n*self.e_h
            e["msg"] = e["msg"]/np.sum(e["msg"]) # normalization
            conv += np.abs(e["msg"] - old_msg).sum() # updating conversion
            # then we need to update the target marginal
            v = self.G.vs[e.target]
            old_mar = v["mar"].copy()
            self.updateMarginal(v)
            if(np.sum(v["mar"])):
                v["mar"] = v["mar"]/np.sum(v["mar"]) # normalization
            # finally we update the auxiliary field
            self.e_h *= np.exp((np.dot(old_mar,self.p) - np.dot(v["mar"],self.p))/self.N)
        return(conv)
    
    def updateZv(self):
        for v in self.G.vs:
            e_jk = np.array(self.G.es.select(_target = v.index)["msg"]) # connected messages
            self.Z_v[v.index] = np.sum(np.dot(e_jk,self.p).prod(0)*self.n*self.e_h) # eq. 31
    
    def updateZe(self):
        for e in self.G.es:
            e_ = self.G.es.select(_source = e.target,_target = e.source)[0] # the opposite message
            self.Z_e[e.index] = e["msg"].dot(self.p).dot(e_["msg"]) # eq. 29
    
    def getMessages(self):
        return(np.array(self.G.es["msg"]))
    
    def getMarginals(self):
        return(np.array(self.G.vs["mar"]))
    
    def getFreeEnergy(self):
        self.updateZv()
        self.updateZe()
        c = np.dot(self.n,self.p).dot(self.n) # average degree from eq. 1
        return((np.sum(np.log(self.Z_e)) - np.sum(np.log(self.Z_v)))/self.N - c/2.0)
    
    def getAssignments(self):
        return(self.getMarginals().argmax(1) + 1)
    
    def getOverlap(self):
        sum_v = self.getMarginals().max(1).sum()/self.N
        max_n = self.n.max()
        return((sum_v - max_n)/(1.0 - max_n))
    
    def getNewParameters(self):
        newN = self.getMarginals().sum(0)/self.N # eq. 34
        newP = 0.0
        self.updateZe() # you may comment this line if you are sure that Z_e is actual
        for e in self.G.es:
            e_ = self.G.es.select(_source = e.target,_target = e.source)[0] # the opposite message
            newP += np.dot(e_["msg"][:,np.newaxis],e["msg"][np.newaxis,:])/self.Z_e[e.index] # eq. 35
        newP *= self.p/(np.dot(newN[:,np.newaxis],newN[np.newaxis,:])*self.N*self.N)
        return(newN,newP)
