import numpy as np
import random
from multiprocessing import Pool, cpu_count
from collections import Counter
from seamr.core import build
from tqdm import tqdm

def fitOne(args):
    model,X,y,kw = args
    model.fit(X,y,**kw)
    return model

def predictOne(args):
    model,X = args
    return model.classes_, model.predict_proba(X)

def genPreds(args, jobs):
    if jobs > 1:
        with Pool(jobs) as pool:
            yield from pool.imap_unordered(predictOne, args)
    else:
        yield from map(predictOne, args)

class ClassifierBase:
    def setClasses(self, y):
        self.classes_ = np.array( sorted(set(y)) )
        self.classIndex = {c:i for i,c in enumerate(self.classes_)}

class BoostedClassifier:
    def __init__(self, key, estimator, n_estimators, n_jobs = 1, max_samples=0):
        self.estimator_def = estimator
        self.n_estimators = n_estimators
        self.sample_rate = max_samples

        if n_jobs < 0:
            self.n_jobs = (1+cpu_count()) + n_jobs
        else:
            self.n_jobs = n_jobs

        self.name = key

    def fit(self, X, y, **kw):
        
        self.classes_ = np.unique(y)
        self.index = { c : i for i,c in enumerate(self.classes_) }

        n = X.shape[0]
        y = np.array(y)

        amount = 0
        if self.sample_rate > 0:
            if self.sample_rate <= 1:    
                amount = X.shape[0] * self.sample_rate
            else:
                amount = min(X.shape[0], self.sample_rate)
        
        if self.n_jobs > 1:
            with Pool(self.n_jobs) as pool:
                pool_args = []
                for k in range(self.n_estimators):
                    model = build(self.estimator_def)
                    rows = np.random.randint(X.shape[0], size=amount)
                    pool_args.append([ model, X[rows, :], y[rows], kw])
                
                self.estimators = list(tqdm(pool.imap_unordered(fitOne, pool_args), total=len(pool_args), desc='fitting estimators') )
        else:
            self.estimators = []
            for k in tqdm(range(self.n_estimators), total=self.n_estimators, desc='fitting estimators'):
                model = build(self.estimator_def)
                rows = np.random.randint(X.shape[0], size=amount)
                self.estimators.append( model.fit( X[rows, :], y[rows], **kw) )
        
        return self
    
    def predict(self, X):
        return self.classes_[ self.predict_proba(X).argmax(axis=1) ]
    
    def predict_proba(self, X):
        probs = np.zeros( (X.shape[0], self.classes_.size) )
        col_w = np.zeros( (1, self.classes_.size) )

        args = [ (e, X) for e in self.estimators ]

        for pCls, eProb in tqdm(genPreds(args, self.n_jobs), total=len(self.estimators), desc="predicting"):
            for j,c in enumerate(pCls):
                col = self.index[c]
                col_w[0, col] += 1
                probs[:, col] = eProb[:, j]
        
        col_w[ col_w == 0 ] = 1
        return probs / col_w

    def __str__(self):
        return self.name

def smoothGroups(indices, mapSize, smoothBy = None):

    mat = np.zeros( (mapSize, mapSize) )
    for i in range(1, len(indices)):
        if indices[i-1] != indices[i]:
            mat[ indices[i-1], indices[i] ] += 1
    
    mat = np.log1p(mat)
    rng = mat.max() - mat.min()
    mat = (mat - mat.min()) / rng
    for i in range(mat.shape[0]):
        mat[i,i] = 1.0

    if isinstance(smoothBy, dict):
        model = build(smoothBy)
    else:
        model = smoothBy

    return model.fit_predict(mat)

class ConstantClassifier:

    def __init__(self, cls):
        self.cls = cls
        self.classes_ = np.array([cls])
    
    def fit(self, *args):
        pass
    
    def predict(self, X):
        return np.array([self.cls] * X.shape[0])
    
    def predict_proba(self, X):
        preds = np.zeros( (X.shape[0], 2) )
        preds[:,1] = 1.0
        return preds

    def __str__(self):
        return "constant"