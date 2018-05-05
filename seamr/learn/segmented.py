import seamr
from seamr import sem
from seamr.core import build
from itertools import groupby, product, chain
from collections import Counter
import networkx as nx
import numpy as np
from multiprocessing import Pool
from scipy import stats

class RollingStatistic(object):

    def __init__(self, sample, epsilon = 10.0 ** -3):
        self.N = len(sample)
        self.epsilon = epsilon
        self.average = np.mean(sample)
        self.variance = np.var(sample) + epsilon
        self.stddev = math.sqrt(self.variance)
        self.q = deque(sample)

    def update(self, new):
        old = self.q.popleft()
        self.q.append(new)

        oldavg = self.average
        newavg = oldavg + (new - old)/self.N
        self.average = newavg
        self.variance += (new-old) * ( new - newavg + old - oldavg) / (self.N-1)
        self.variance = abs(self.variance)
        self.stddev = math.sqrt(self.variance)
        
def makeSignal( args ):
    # http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/
    c, col, epsilon = args
    rStat = RollingStatistic( col[:120], epsilon = epsilon )
    for row in range(col.size):
        v = col[row]
        rStat.update(v)
        sig = (v - rStat.average) / (rStat.stddev + epsilon)

        col[row] = sig
    return (c, col)

def toProb(seq):
    return { c : n/len(seq) for c,n in Counter(seq).items() }

class SegmentedLearner:
    def __init__(self, store, dset, wrapped, weigh_by_entropy=False):
        self.model = build(wrapped)
        self.store = store
        self.dset = dset
        self.weigh_by_entropy = weigh_by_entropy
        cose = sem.COSE()

        g = nx.read_graphml( store.path(dset, "%s.graphml" % dset) )
        sg_edges = []

        cntOK = 0.0
        cntTotal = 0.0

        for n,d in g.nodes(data=True):
            if cose.is_a( "cose:"+d['kind'], "cose:FunctionalSpace" ):
                sg_edges.append( [d['start'], d['end']] )
                cntTotal += 1
                cntOK += int(d['start_label'] == d['end_label'])

        self.edges = sorted(sg_edges, key=lambda T: T[0])
        seamr.log.info("%s segmenting is %0.3f accurate" % ( dset, 100 * cntOK / cntTotal ))
        seamr.log.info("%s has %s segments" % (dset, len(self.edges)))
    
    @property
    def classes_(self):
        return self.model.classes_

    def __str__(self):
        return "gseg-%s" % str(self.model)

    def getRowGroups(self, times):
        g = [0] * len(times)
        k = 0
        t_iter = iter(enumerate(times))

        for i,(s,e) in enumerate( self.edges ):
            while k < len(times) and times[k] <= e:
                g[k] = i
                k += 1
                
        while k < len(times):
            g[k] = i
            k += 1
        
        return [ (i,list(items)) for (i,(gi, items)) in enumerate(groupby( range(len(times)), key=lambda i: g[i])) ]

    def resampleX(self, groups, X):
        newX = np.zeros( (len(groups), X.shape[1] ), dtype=X.dtype )
        for i,rows in groups:
            newX[i, :] = X[rows,:].max(axis=0)
            #newX[i, xi.shape[1]: ] = xi.std(axis=0)
        
        return newX

    def resampleY(self, groups, Y):
        newY = [""] * len(groups)
        
        for i,rows in groups:
            newY[i] = Counter( Y[x] for x in rows ).most_common(1)[0][0]
        
        return np.array(newY)
        
    def fit(self, Xtup, y):
        times, X = Xtup

        groups = self.getRowGroups(times)

        self.model.fit( self.resampleX(groups, X), 
                        self.resampleY(groups, y) )

        return self

    def predict(self, Xtup):
        mcl = self.classes_[ self.predict_proba(Xtup).argmax(axis=1) ]

        assert mcl.shape[0] == Xtup.shape[0], "Got wrong number of outputs for the inputs"

        return mcl

    def predict_proba(self, Xtup):
        times, X = Xtup
        
        groups = self.getRowGroups(times)

        ySample = self.model.predict_proba( self.resampleX(groups, X) )
        yPred = np.zeros( (X.shape[0], ySample.shape[1]) )

        for i,rows in groups:
            yPred[rows,:] = ySample[i,:]

        return yPred

class SegmentFeatureAdapter:
    def __init__(self, wrapped, store=None, sensors=None, dset=None):
        self.wrapped = build(wrapped, local_args = locals())

    def __str__(self):
        return "gseg-%s" % str(self.wrapped)

    def fit(self, labels, stream, times):
        if hasattr(self.wrapped,"fit"):
            return self.wrapped.fit(labels, stream, times)

    def make_dataset(self, event_stream, times):
        return times, self.wrapped.make_dataset(event_stream, times)
        
