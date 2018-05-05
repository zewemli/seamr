import numpy as np
from collections import defaultdict, Counter, deque
from multiprocessing import Pool

from seamr import core
from seamr.core import Features, ID_Index
from seamr.core import build
from seamr.spatial import Reasoner
from seamr import sem
import seamr

from seamr.expectations import CrossModelExp

from scipy import stats
from tqdm import tqdm

import os,sys

import math

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
        sig = (v - rStat.average) / rStat.stddev

        col[row] = math.tanh(sig)
    return (c, col)

class PatchFeatures(Features):
    _length = 0
    _days = []

    def __init__(self, store, dset, decay=15, with_exp=False, **kw):

        self.with_exp = with_exp
        self.expModels = None
        self.expIndices = None
        self.res = Reasoner( dset, store.get_sensors(dset) )
        self.mean = None
        self.std = None
        self.epsilon = 10.0 ** -3

        cose = sem.COSE()

        if with_exp:
            self.expModels = []
            self.expIndices = []

            em = set()
            for cls in sorted( set(sem.activity_type_map.values()) ):
                try:
                    model = CrossModelExp(store, cls)
                    if model:
                        em.add(model)
                except:
                    pass

            self.expModels = sorted(em, key=hash)

        if decay > 1:
           self.decay = 1.0 / decay
        else:
            self.decay = 0.5

        self.dataset = dset
        self.index = {}
        self.kind = []
        self.sensors = {}
        self._length = 0
        self._patchLevels = None

        self._store = store

        self.index = {}
        
        self.index.update( cose.time_order() )

        for p in sorted(self.res.getDefaultPatches(), key=self.key):
            if self.key(p) not in self.index:
                self.index[ self.key(p) ] = len(self.index)
        
        if self.expModels:
            for m in self.expModels:
                act = m.cls.split(":")[-1]
                inds = []
                for feat in ('level','smooth','diff'):
                    inds.append(len(self.index))
                    self.index[ "%s-%s" % (act, feat) ] = len(self.index)
                self.expIndices.append(inds)

        self.feat_names = sorted(self.index.keys(), key=lambda s: self.index[s])
        
        self.feat_names.extend(["sec_of_day", "hour_of_day"])

    """
    This provides features for each object/area within
    a specific environment.
    """

    def key(self, p):
        return str(p.kindKey)
    
    @staticmethod
    def entropy(vals, norm):
        s = sum(vals)
        if s == 0:
            return 0
        else:
            e = 0.0
            for v in vals:
                p = v/s
                e -= p * np.log2(p)
                
            return e / np.log2(norm)

    @staticmethod
    def toProb(d):
        s = sum(d.values())
        return { k : v/s for k,v in d.items() }

    @staticmethod
    def findCut( valDist, minMass = 0.05 ):
        allK = Counter()
        for mk in valDist.values():
            allK.update(mk)
        
        bestUtil = 0.0
        bestCut = 0.0

        norm = len(allK)
        maxMass = sum(allK.values())
        pMass = 1.0

        for kLevel, kDist in sorted(valDist.items()):
            allK.subtract( kDist )
            nz = [ v for v in allK.values() if v > 0 ]
            kMass = sum(allK.values()) / maxMass
            if kMass < minMass:
                break
            else:
                kEnt = PatchFeatures.entropy(nz, norm)
                kUtility = ((pMass - kMass) / (pMass + kMass)) / kEnt
                if kUtility > bestUtil:
                    bestUtil = kUtility
                    bestCut = kLevel
                pMass = kMass
        
        return bestCut

    def genPatches(self, events, times):

        epsilon = 0.05
        patchGen = self.res.genPatches(events, times, withSensors=False)

        n = len(times)
        for rowNum, (patches, ts) in enumerate(patchGen):
            n -= 1
            yield [ p for p in patches if p.value >= epsilon ], ts
            
        assert n == 0, ("Got wrong times for genPatches", n, len(times),)

    def make_dataset(self, events, times, patches=None):
        time_label_num = sem.COSE.time_label_num

        days = sorted(set([ e.day for e in events() ]))

        lenIndex = len(self.index)
        dStart = lenIndex
        dEnd = 2 * lenIndex

        fDecay = np.zeros( lenIndex )

        X = np.zeros( (len(times), lenIndex * (1 + int(self.decay > 0)) + 2), dtype=np.float32 )
        i = -1
        epsilon = 10.0 ** -3

        _msg_ = "building dataset with %s times and %s days for %s" % (len(times), len(days), self.dataset)

        with seamr.BlockWrapper(_msg_):

            if patches is None:
                patchGen = self.genPatches(events, times)
            else:
                patchGen = patches

            for rowNum, (patches, ts) in enumerate(patchGen):
                sec_of_day = ts % 86400                
                X[rowNum, time_label_num(sec_of_day) ] = 1.0
                
                for p in patches:
                    X[rowNum, self.index[ self.key(p) ] ] = p.value
                
                if self.expModels:
                    for mod, (m_val, m_smooth, m_diff) in zip(self.expModels, self.expIndices):
                        value = mod.matchValue( patches )
                        X[rowNum, m_val] = value
                        if rowNum > 0:
                            X[rowNum, m_smooth] = X[rowNum-1, m_smooth] * (1.0 - self.decay) +  self.decay * value
                            X[rowNum, m_diff] = X[rowNum, m_val] - X[rowNum, m_smooth]

                X[rowNum, -2] = sec_of_day / 86400
                X[rowNum, -1] = sec_of_day // 3600
                
        X[ ~np.isfinite(X) ] = 0.0

        return X
        
    def _adjust(self, X):
        adj = (X - self.mean) / self.std
        adj[ ~np.isfinite(adj) ] = 0.0
        return adj

class SingleEnvFeatures(PatchFeatures):
    def key(self, p):
        return str(p.objKey)

    def __str__(self):
        if self.with_exp:
            return "single-model-%s" % self.decay
        else:
            return "single"

class CrossEnvFeatures(PatchFeatures):

    def key(self, p):
        return str(p.kindKey)
    
    def intersect(self, feats, mat):
        cIndex = {c:i for i,c in enumerate(self.feat_names)}
        fCols = [ cIndex[x] for x in feats ]
        return mat[:, fCols]

    def __str__(self):
        if self.with_exp:
            return "cross-model-%s" % self.decay
        else:
            return "cross"
