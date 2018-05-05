from random import choice, sample
from collections import defaultdict, deque
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier 
import numpy as np

class SVMSubset:
    def __init__(self, n_samples, svmOptions):
        self.n_samples = n_samples
        self.svmOptions = dict(svmOptions.items())
        self.svmOptions["probability"] = True
    
    def __str__(self):
        return "svm"
    
    def __repr__(self):
        return str(self)

    def fit(self,X,y):
        self.classes_ = np.array(sorted(set(y)))

        df = defaultdict(list)
        for i,l in enumerate(y):
            df[l].append(i)
        
        n = X.shape[0]
        order = deque(sorted(df.items(), key=lambda T: len(T[1])))

        nleft = self.n_samples

        choice_opts = []

        while len(order):
            nOrd = len(order)
            cls, indices = order.popleft()

            optSize = int((self.n_samples - len(choice_opts)) / nOrd)
            if len(indices) < optSize:
                choice_opts.extend( indices )
            else:
                choice_opts.extend( sample(indices, optSize) )
        
        self.model = SVC(**self.svmOptions)
        choice_opts = np.array(choice_opts, dtype=np.int32)
        
        self.model.fit( X[choice_opts,:], np.array(y)[choice_opts] )

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)