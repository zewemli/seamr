import os

import seamr
from seamr import sem

import numpy as np
from pycrfsuite import Trainer, Tagger

class CRFSegmenter:
    def __init__(self,
                    store,
                    dset,
                    model_dir,
                    gap=3,
                    window=15,
                    step=7,
                    minSegLen=10,
                    maxSegLen = 200,
                    max_iterations = 150):

        self.store = store
        self.dset = dset
        
        self.gap = gap
        self.window = window
        self.step = step
        self.minSegLen = minSegLen
        self.maxSegLen = maxSegLen

        self.splitAt = 1.0

        self.max_iterations = max_iterations
        self.model_path = os.path.join(model_dir, "%s.segmenter.crf" % dset)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def seg_to_crf(self, fields, row, gap, mat ):
        feats = {}
        for i in range(max(0, (row+1) - gap), row + 1):
            for j,col in enumerate(fields):
                if mat[i,j] != 0.0:
                    feats["%s:%02d" % (col,i)] = mat[i,j]
        return feats

    def fit(self, features, labelIndex, events, times):
    
        X = features.make_dataset(events, times)
    
        crfDataset = [ self.seg_to_crf( features.feat_names, i, self.gap, X ) for i in range(X.shape[0]) ]

        trainer = Trainer(params = { 'max_iterations': self.max_iterations })
        
        for res in labelIndex.make_dataset(times, prefer_mat = True):
            rSeq = [ sem.activity_type_map[x] for x in res ]
            segMarker = ['f'] * len(rSeq)
            for i in range( 1, len(rSeq) ):
                segMarker[i] = 't' if rSeq[i] != rSeq[i-1] else 'f'
            
            for i in range(self.window, len(rSeq), self.step):
                trainer.append( crfDataset[i - self.window : i], segMarker[i - self.window : i] )

        with seamr.BlockWrapper("Fitting: %s" % self.model_path):
            trainer.train( self.model_path )
        
        self.setSplitLevel( features, X )
        seamr.log.info("Splitting at %s" % self.splitAt)

        return self
    
    def setSplitLevel(self, features, X):

        crfDataset = [ self.seg_to_crf(features.feat_names, i, self.gap, X) for i in range( X.shape[0] ) ]

        tagger = Tagger()
        tagger.open(self.model_path)

        probs = []

        try:
            tagger.set(crfDataset)
            for i in range( len(crfDataset) ):
                probs.append( tagger.marginal('t', i) )

        finally:
            tagger.close()
        
        for i in range(100, 1, -1):
            iprob = 1.0 / (i+1)
            longest = max(map(lambda T: T[1] - T[0], self.labelsToSegs([ 't' if p > iprob else 'f' for p in probs ])))
            print(longest, 1.0 / i)
            self.splitAt = 1.0 / i
            if longest > self.maxSegLen:
                return self.splitAt

    def tag(self, features, events, times):

        X = features.make_dataset(events, times)
        
        crfDataset = [ self.seg_to_crf( features.feat_names, i, self.gap, X ) for i in range( X.shape[0] ) ]

        tagger = Tagger()
        tagger.open(self.model_path)

        tags = []

        try:
            tagger.set(crfDataset)
            lastT = 0
            for j in range(len(crfDataset)):
                jdiff = j - lastT
                if jdiff > self.minSegLen and (tagger.marginal('t', j) >= self.splitAt or jdiff > self.maxSegLen):
                    tags.append('t')
                    lastT = j
                else:
                    tags.append('f')
                    
        finally:
            tagger.close()
        
        return tags
    
    def labelsToSegs(self, tags):
        segs = []

        s = 0
        ends = [ i for i in range(1, len(tags) - self.minSegLen) if tags[i] == 't' and tags[i-1] == 'f' ]

        for e in ends:
            if (e-s) >= self.minSegLen:
                segs.append((s,e))
                s = e
        
        segs.append( (s, len(tags)) )

        return segs
