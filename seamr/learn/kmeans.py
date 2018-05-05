import numpy as np
from sklearn import cluster
from scipy import stats

from scipy.spatial import distance

from seamr.learn import ClassifierBase
from seamr.learn import smoothGroups

from collections import Counter, defaultdict

class KMeansClassifier( ClassifierBase ):

    def __init__(self, key,
                       n_clusters = 128,
                       center_and_scale = False,
                       epsilon = 1.0 / (10 ** 3),
                       xform = None,
                       smoothBy = None):

        self.key = key
        self.remap = None
        self.n_clusters = n_clusters
        self.smoothBy = smoothBy
        self.center_and_scale = center_and_scale
        self.xform = getattr(np, xform, None)
        self.mean = None
        self.std = None
        self.epsilon = epsilon

    def __str__(self):
        return self.key

    def predict(self, X):
        probs = self.predict_proba( X )
        return self.classes_[ probs.argmax(axis=1) ]

    def predict_proba(self, X):
        indices = self.remap[ self.model.predict(X) ]
        return self.clusterDist[ indices ]

    def fit(self, X, y):
        self.setClasses(y)

        if self.xform is not None:
            X = self.xform(X)
        
        if self.center_and_scale:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0) + self.epsilon

            X = (X - self.mean) / self.std
        
        n_clusters = self.n_clusters
        self.model = cluster.MiniBatchKMeans(n_clusters = self.n_clusters)
        
        indices = self.model.fit_predict(X).astype(np.int32)
        self.remap = smoothGroups(indices, self.n_clusters, self.smoothBy).astype(np.int32)
        indices = self.remap[ indices ]

        # Now get the class distribution
        self.clusterDist = np.zeros( (self.n_clusters, self.classes_.size) )

        for c,lbl in zip(indices, y):
            self.clusterDist[ c, self.classIndex[lbl] ] += 1

        self.clusterDist /= (self.epsilon + self.clusterDist.sum(axis=1, keepdims=True))

        return self