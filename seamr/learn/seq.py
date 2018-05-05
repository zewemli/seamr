from collections import Counter, defaultdict
import os
import uuid

from tqdm import tqdm

import numpy as np
import tempfile

import seamr
from seamr.core import build

from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy import stats
from seamr.learn import ClassifierBase

from sklearn import cluster
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import MiniBatchKMeans

from seqlearn.hmm import MultinomialHMM

from pycrfsuite import Trainer, Tagger, ItemSequence

class HMMClassifier(ClassifierBase):
    
    def __init__(self, key, clusterer, seqlen = 30, seqstep=5):
        self.key = key
        self.clusterer = build(clusterer)
        self.nGroups = 0
        self.getGroups = None
        self.seqlen = seqlen
        self.seqstep = seqstep
        
    def __str__(self):
        return self.key

    def predict(self, X):
        groups = self.getGroups(X)
        csrMat = csr_matrix(([1] * len(groups), (np.arange(len(groups)), groups)),
                            shape=(len(groups), self.nGroups))
        
        return self.hmm.predict( csrMat )

    def fit(self, X, y):
        self.setClasses(y)
        
        with seamr.BlockWrapper("HMM: Fitting clusterer"):
            self.clusterer.fit(X,y)

        if hasattr(self.clusterer, "apply"):
            self.getGroups = self.clusterer.apply
            self.nGroups = self.clusterer.get_params().get("max_leaf_nodes") * 2        
        else:
            self.getGroups = self.clusterer.predict
            self.nGroups = self.clusterer.cluster_centers_.shape[0]

        groups = self.getGroups(X)

        seqGroups = []
        seqLabels = []
        lengths = []
        for k in range(self.seqlen, groups.shape[0], self.seqstep):
            lengths.append(self.seqlen)
            seqGroups.extend( groups[k - self.seqlen : k] )
            seqLabels.extend( y[k - self.seqlen : k] )
        
        gLen = len(seqGroups)
        csrMat = csr_matrix(([1] * gLen, (np.arange(gLen), seqGroups)), shape=(gLen, self.nGroups))

        with seamr.BlockWrapper("HMM: Fitting hmm"):
            self.hmm = MultinomialHMM()
            self.hmm.fit( csrMat, seqLabels, lengths )
                    
        return self

class CRFClassifier(ClassifierBase):

    def __init__(self, key,
                    seqlen=20,
                    seqstep = 10,
                    crf_params = {"c1": 1.0,
                                  "c2": 0.001,
                                  "max_iterations": 150, 
                                  'feature.possible_transitions': True}):

        self.key = key
        self.seqlen = seqlen
        self.seqstep = seqstep
        self.crf_params = crf_params
        self.modelfile = os.path.join(tempfile.gettempdir(), str(uuid.uuid4())+".crfsuite")

    def __str__(self):
        return self.key

    def fit(self, X, y):
        y = list(map(str, y))
        self.setClasses( y )

        trainer = Trainer( verbose = True )

        for i in range(self.seqlen, len(X), self.seqstep):
            trainer.append(X[i - self.seqlen : i], y[i - self.seqlen : i])

        trainer.set_params(self.crf_params)

        trainer.train(self.modelfile)

        return self

    def predict_proba(self, X):

        tagger = Tagger()
        tagger.open( self.modelfile )
        tagger.set( X )

        probs = np.zeros( (len(X), self.classes_.size) )

        for col,cls in enumerate(self.classes_):
            try:
                for row in range(len(X)):
                    probs[row,col] = tagger.marginal(cls, row)
            except:
                pass

        tagger.close()

        return probs

    def predict(self, X):

        tagger = Tagger()
        tagger.open( self.modelfile )

        labels = tagger.tag( xseq = X )

        tagger.close()

        return labels