import numpy as np
from sklearn import cluster
from scipy import stats
from scipy.spatial import distance

import seamr
from seamr.core import build
from seamr.learn import smoothGroups
from seamr.learn import ClassifierBase

from multiprocessing import Pool

from collections import Counter, defaultdict, deque
from tqdm import tqdm

def external_fit(args):
    g, model, X, y = args
    model.fit(X,y)
    return {g : model}

def external_predict_proba(args):
    g, rows, model, X = args
    return (g, rows, model.predict_proba(X))

class ACEClassifier( ClassifierBase ):

    def __init__(self, key, clusterer, expert, smoothBy=None):
        self.key = key
        self.clusterer = build(clusterer)
        self.experts = {}
        self._expertDef = expert
        self._model_groups = None
        self.remap = None
        self.smoothBy = smoothBy

        self.otherIndex = 0

    def __str__(self):
        return self.key

    def predict_proba(self, X):

        gX, fX = self.wrt(X)

        probs = np.zeros( (gX.shape[0], self.classes_.size) )

        pred_args = []
        if isinstance(fX, list):
            for exp, expRows in self.getIndices(X):
                expSubMat = [ fX[r] for r in expRows ]
                pred_args.append( (exp, expRows, self.experts.get(exp, self.defaultExpert), expSubMat) )
        else:
            for exp, expRows in self.getIndices(X):
                pred_args.append( (exp, expRows, self.experts.get(exp, self.defaultExpert), fX[expRows,:]) )

        with Pool() as pool:
            for exp, rows, exProbs in pool.imap_unordered(external_predict_proba, pred_args):
                ex = self.experts[exp]
                for i,c in enumerate(ex.classes_):
                    probs[rows, self.classIndex[c]] = exProbs[:,i]
        
        return probs

    def getIndices(self, X):

        gX, fX = self.wrt(X)

        expertGroups = defaultdict(list)
        exampleGroup = self.remap[ self._model_groups( gX ) ]

        for i,g in enumerate(exampleGroup):
            expertGroups[g].append(i)

        return list(expertGroups.items())

    def predict(self, X):
        return self.classes_[ self.predict_proba(X).argmax(axis=1) ]

    def getExpert(self):
        return build(self._expertDef)

    def wrt(self, X):
        if isinstance(X, tuple) and len(X) == 2:
            return X
        else:
            return X, X

    def fit(self, X, y):
        gX, fX = self.wrt(X)

        self.setClasses(y)
        
        self.otherIndex = self.classIndex['other']

        with seamr.BlockWrapper("ACE: Fitting clusterer"):
            self.clusterer.fit(gX,y)

        if hasattr(self.clusterer, "apply"):
            self._model_groups = self.clusterer.apply
            self.nGroups = self.clusterer.get_params().get("max_leaf_nodes") * 2
        else:
            self._model_groups = self.clusterer.predict
            self.nGroups = self.clusterer.cluster_centers_.shape[0]

        indices = self._model_groups( gX )

        if self.smoothBy is None:
            self.remap = np.arange( self.nGroups )
        else:
            self.remap = smoothGroups( indices, self.nGroups, self.smoothBy )
        
        # Now remap the indices to the group they belong to
        expertGroups = defaultdict(list)
        for i,g in enumerate( self.remap[ indices ] ):
            expertGroups[g].append(i)
        
        y = np.array(y) # Just making sure

        fit_args = []

        if isinstance(fX, list):
            for g,gRows in expertGroups.items():
                fit_args.append( (g, self.getExpert(), [fX[r] for r in gRows], y[gRows]) )
        else:
        
            for g,gRows in expertGroups.items():
                fit_args.append( (g, self.getExpert(), fX[gRows, :], y[gRows]) )
        
        with Pool() as pool:
            modelStream = pool.imap_unordered(external_fit, fit_args)
            for fitted in tqdm(modelStream, desc="ACE: Fitting experts", total=len(fit_args)):
                self.experts.update( fitted )
        
        mainExp = max( expertGroups.keys(), key=lambda k: len(expertGroups[k]) )
        self.defaultExpert = self.experts[mainExp]

        return self