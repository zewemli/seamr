from seamr import sem
from seamr import core
from seamr import spatial

from seamr.features.semantic import CrossEnvFeatures
from seamr.features.semantic import SingleEnvFeatures

import numpy as np
import math
from itertools import chain
from collections import Counter

from pycrfsuite import ItemSequence

class CRFFeatures( core.Features ):
    def __init__(self, store, dset, crossEnv=True, with_exp = True, decay= 8.0):
        self.crossEnv = crossEnv
        self.with_exp = with_exp

        if crossEnv:
            self.semFeats = CrossEnvFeatures(store, dset, decay=decay, with_exp = with_exp)
        else:
            self.semFeats = SingleEnvFeatures(store, dset, decay=decay, with_exp = with_exp)
        
        self.objSpace = { b.name : b.space for b in self.semFeats.res.getObjBoxes() if b.space }
        self.feat_names = sorted(sem.COSE.time_order().keys()) \
                          + sorted( set(self.semFeats.feat_names) \
                          | set(self.objSpace.values()) )

    def __str__(self):

        if self.crossEnv:
            name = "cross-crf"
        else:
            name = "env-crf"
        
        if self.with_exp:
            return "%s-model" % name
        else:
            return name

    def intersect(self, commonFeats, mat):
        if set(self.feat_names) == set(commonFeats):
            return mat
        else:
            commonFeats = set(commonFeats)
            isOK = lambda o: o in commonFeats
            return [ list(filter(isOK, row)) for row in mat ]

    def fit(self, labels, events, times):
        self.semFeats.fit(labels, events, times)

    def genPatches(self, events, times):
        return self.semFeats.genPatches(events, times)

    def make_dataset(self, events, times, patches = None):
        time_label = sem.COSE.time_label
        key = self.semFeats.key

        dataset = []

        if patches is None:
            patches = self.semFeats.genPatches(events, times)

        for batch,t in patches:
            row = {}

            for p in batch:
                row[ key(p) ] = p.value

                pSpace = self.objSpace.get(p.obj, None)
                if pSpace:
                    row[pSpace] = max(p.value, row.get(pSpace, 0.0))

            row[ time_label(t % 86400) ] = 1.0
            dataset.append(row)

        return dataset
    
    # Semantic weighting functions
    #--------------------------------------------------------------------------------------------

    def getModelMasked(self, classModel, X, y_pred, featSet = None):

        if classModel is None:
            return y_pred
        else:
            epsilon = 10.0 ** -5
            if featSet is None:
                featSet = set(self.feat_names)
            
            featSet = featSet & set(self.feat_names) # Just making sure

            clsConcepts = { c.split(":")[-1] for c in classModel.concepts() }
            featCols = [ c for c in self.feat_names if c in featSet and c.split("-")[0] in clsConcepts ]
            
            example_weight = np.ones_like( y_pred )
            for i in range(y_pred.shape[0]):
                row = X[i]
                rowSum = sum( row.values() )
                rowW = sum( row.get(k,0) for k in featCols )
                
                example_weight[i] = rowW / (rowSum + epsilon)

            masked = y_pred * example_weight

            return masked / max(1.0 / masked.size, masked.max())

    def getExpectedMask(self, expectations, X, y_pred, featSet = None):
        if expectations is None:
            return y_pred
        else:
            if featSet is None:
                featSet = set(self.feat_names)

            epsilon = 10.0 ** -5
            if featSet is None:
                featSet = set(self.feat_names)
            
            featSet = featSet & set(self.feat_names) # Just making sure

            clsConcepts = { c.split(":")[-1] for c in classModel.concepts() }
            featCols = [ c for c in self.feat_names if c in featSet and c.split("-")[0] in clsConcepts ]
            
            example_weight = np.ones_like( y_pred )
            for i in range(y_pred.shape[0]):
                row = X[i]
                rowSum = sum( row.values() )
                rowW = sum( expectations.get(c,0.0) * v for c,v in row.items() )
                
                example_weight[i] = rowW / (rowSum + epsilon)

            masked = y_pred * example_weight

            return masked / max(1.0 / masked.size, masked.max())


    #--------------------------------------------------------------------------------------------

    def getExpectations(self, X, y):
        ret = []
        nRows = len(X)
        for i in range(y.shape[1]):
            iVals = Counter()
            
            for r in range(y.shape[0]):
                if y[r] > 0:
                    iVals.update( X[r] )
            
            ret.append({ k : v/nRows for k,v in iVals.items() })

        return ret

class CRFWrapped( core.Features ):
    def __init__(self, sensors = None,
                       dset = None,
                       store = None,
                       wrapped = {}):
                       
        self.feats = core.build(wrapped, locals())
        self.dset = dset
        self.sensors = sensors
        assert hasattr(self.feats, "feat_names")
        
    def __str__(self):
        return 'crf-%s' % str(self.feats)

    def make_dataset(self, event_stream, times):
        
        X = self.feats.make_dataset(event_stream, times)
        assert X.shape[1] == len(self.feats.feat_names)

        return [
            { k:v for k,v in zip(self.feats.feat_names, X[i,:]) if v != 0 }    
            for i in range(X.shape[0])
        ]