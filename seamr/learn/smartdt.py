import seamr
from seamr.core import build
from seamr.learn import ConstantClassifier
from seamr import log
from random import choice, sample
from collections import defaultdict, deque

from multiprocessing import Pool, cpu_count, current_process
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.naive_bayes import GaussianNB

from tqdm import tqdm

import numpy as np

def train_leaf(args):
    leaf, X, y, leafDef = args
    model = build(leafDef)
    try:
        model.fit(X,y)
    except:
        model = ConstantClassifier(y[0])
    return leaf, model
    
class SmartDT:
    def __init__(self, dt, leaf, sample_rate = 0, parallelize = False, key=None):
        self.sample_rate = sample_rate
        self.parallelize = parallelize
        self.dt = build(dt)
        self.leaves = {}
        self.leafDef = leaf
        self.name = key or 'smart-dt'

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)

    def fit(self,X,y):

        self.classes_ = np.array(sorted(set(y)))
        self.clsIndex = { c : i for i,c in enumerate(self.classes_) }

        rows = sample(list(range(X.shape[0])), X.shape[0] // self.sample_rate)

        self.dt.fit(X[rows,:], [ y[i] for i in rows ])

        # get leaves
        leaves = self.dt.apply(X)

        log.info("SmartDT has %s leaves" % np.unique(leaves).size)

        if self.parallelize and self.parallelize != 1 and not current_process().daemon:
            p = self.parallelize
            
            with Pool( p if p > 0 else cpu_count() ) as pool:
                args = []
                for r in np.unique(leaves):
                    XSub = X[ leaves == r, : ]
                    ySub = [ y[i] for i,l in enumerate(leaves) if l == r ]
                    args.append([ r, XSub, ySub, self.leafDef ])
                
                self.leaves = { r : svm for (r,svm) in tqdm(pool.imap_unordered(train_leaf, args), total=len(args), desc='fitting leaves') }

        else:
            r_unique = np.unique(leaves)
            for r in tqdm(r_unique, total=len(r_unique), desc='fitting leaves'):
                # Yes you could parallelize this
                # however we generally run a bunch of these
                # in parallel at a higher level. Feel free to add 
                # to this class.
                XSub = X[ leaves == r, : ]
                ySub = [ y[i] for i,l in enumerate(leaves) if l == r ]
                
                leaf,model = train_leaf([ r, XSub, ySub, self.leafDef ])
                self.leaves[leaf] = model

        return self

    def predict(self, X):
        return self.classes_[ self.predict_proba(X).argmax(axis=1) ]

    def predict_proba(self, X):
        
        preds = np.zeros( (X.shape[0], self.classes_.size) )

        leaves = self.dt.apply( X )
        for r in np.unique(leaves):

            if r not in self.leaves:
                # If we've never seen the leaf before, find the 'nearest' leaf
                r = min(self.leaves.keys(), key=lambda k: abs(k-r))

            svm = self.leaves[r]
            rows = leaves == r
            
            probs = svm.predict_proba( X[rows,:] )
            for j,c in enumerate(svm.classes_):
                preds[rows, self.clsIndex[c]] = probs[:, j]
        
        return preds
