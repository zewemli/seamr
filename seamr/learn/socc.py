import numpy as np
import pomegranate as pm
from seamr.core import build, OptimalF1
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter, namedtuple

from sklearn import metrics

import random
from tqdm import tqdm
import seamr
from seamr import sem
from prg import prg

class CPFinder:

    @staticmethod
    def f1(tp, fn, fp, tn):
        beta = 1.0
        prec = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (1 + beta **2) * ((prec*recall) / ((beta**2 * prec) + recall))

    @staticmethod
    def opPoints(y, preds):
        srt = np.argsort(preds)
        preds = preds[srt]
        y = y[srt]

        n = y.size
        tn = 0.0
        fn = 0.0
        tp = y.sum()
        fp = y.size - tp

        for i in range(preds.size):
            if y[i] == 1:
                fn += 1
                tp -= 1
            else:
                tn += 1
                fp -= 1
            
            if i == 0 or preds[i] != preds[i-1]:
                yield preds[i], (tp, fn, fp, tn)
            

    @staticmethod
    def freqPoints(y, preds, min_freq, max_freq):
        srt = np.argsort(preds)
        preds = preds[srt]

        i_start = int(preds.size * (1 - min_freq))
        i_end = int(preds.size * (1 - max_freq))

        for i,(cp,(tp, fn, fp, tn)) in enumerate(CPFinder.opPoints(y, preds)):
            yield cp, (tp, fn, fp, tn)
            
    @staticmethod
    def getBest(y, preds, measure, min_freq, max_freq):
        cpoint = 1.0
        best_score = 0.0

        for cp, (tp, fn, fp, tn) in CPFinder.freqPoints(y, preds, min_freq, max_freq):
            
            val = measure(tp, fn, fp, tn)
            if val > best_score:
                best_score = val
                cpoint = cp
        
        return best_score, cpoint

class DensityProb:
    def __init__(self, classes, i, y, probs):
        self.label = classes[i]
        self.probs = probs
        self.y = y

        self.fitHist, self.bins = np.histogram(probs, bins='doane')
        h = []

        for j in range(self.fitHist.size):
            rows = (probs >= self.bins[j]) & (probs < self.bins[j+1])
            if rows.sum() > 0:
                h.append( y[rows].mean() )
            else:
                h.append( 0.0 )
            
        self.hist = np.array( h )
        
        ym = y.mean()

        step = int(y.size // 5)
        ys = np.std([ y[i-step:i].mean() for i in range(step, y.size, step)])

        self.lowFreq = max(0, ym - ys)
        self.highFreq = ym + ys

        print(self.label, ym, self.lowFreq, self.highFreq)

        tup = CPFinder.getBest( y, probs, CPFinder.f1, ym - ys, ym + ys )
        self.bestScore, self.bestCut = tup

    def checkPoint(self, checky, preds):
        preds = self(preds)
        pmax = 0.0
        pct = 0.0
        cp = 0.0

        _,b = np.histogram(np.hstack([preds, self.probs]), bins='doane')

        prob_digits = np.digitize(self.probs, self.bins)
        pdist = np.zeros( int(prob_digits.max()) + 1 )
        cn = np.zeros_like( pdist )
        for x,y in zip(prob_digits, self.y):
            pdist[x] += int(y)
            cn[x] += 1.0
        cn[ cn == 0 ] = 1.0
        pdist = pdist / pdist.sum() # cn

        yvec = checky == self.label

        assert preds.size == yvec.size

        pred_digits = np.digitize(preds, self.bins)
        pred_dist = np.zeros_like( pdist )
        zn = np.zeros_like( pred_dist )
        for x,y in zip(pred_digits, yvec):
            pred_dist[x] += int(y)
            zn[x] += 1.0
        zn[zn == 0] = 1.0

        pred_dist = pred_dist / pred_dist.sum() # zn

        print("### "+self.label)
        print(" ".join([ "%0.3f" % v for v in b ]))
        print("- fit -")
        print(" ".join([ "%0.3f" % v for v in pdist ]))
        print(" ".join([ "%0.3f" % v for v in ( cn / cn.sum() ) ]))
        print('------')
        print("- test -")
        print(" ".join([ "%0.3f" % v for v in pred_dist ]))
        print(" ".join([ "%0.3f" % v for v in ( zn / zn.sum() ) ]))
        print("\n")

    def pred(self, pred):
        return (pred >= self.bestCut).astype(np.float64)

    def __call__(self, pred):
        return self.hist[ np.minimum(np.digitize(pred, self.bins), self.hist.shape[0] - 1) ]

# -----------------------------------------

class SOCC:
    '''
    Self-Organizing Classifier Community
    '''

    def __init__(self, model= None, default_cut = 0.95, sample_rate=3, time_var = -2):
        self.modelDef = model
        self.sample_rate = sample_rate
        self.time_var = time_var
        self.time_mask = None
        self.default_cut = default_cut
        self.model = None
        self.points = []
                    
    def fit(self, X, y):
        
        self.model = build(self.modelDef)

        self.classes_ = np.array( sorted(set(y)) )
        cIndex = { c:i for i,c in enumerate(self.classes_) }
        self.cIndex = cIndex

        # Find the instances of activities
        classBlocks = defaultdict(list)
        pl = None
        lstart = 0
        yMat = np.zeros( (X.shape[0], len(cIndex)) )

        for i,l in enumerate(y):
            yMat[i, cIndex[l]] = 1.0

            if l != pl:
                if pl:
                    classBlocks[pl].append([ lstart, i ])
                pl = l
                lstart = i

        classBlocks[ y[-1] ].append([lstart, len(y)])

        if self.time_var is not None:
            self.time_mask = np.zeros( (len(sem.COSE.time_order()), len(cIndex)) )
            for i,(s,l) in enumerate( zip( X[:, self.time_var], y ) ):
                tm = sem.COSE.time_label_num( s )
                cls = cIndex[l]
                self.time_mask[ tm, cls ] = 1.0

        yVec = np.array(y)

        # Now get the samples
        rows = np.full( len(y), False, dtype=bool )

        for _, insts in classBlocks.items():
            if len(insts) == 1:
                s,e = insts[0]
                inst_rows = list(range( s, e ))
                rows[ random.sample(inst_rows, len(inst_rows) // self.sample_rate) ] = True
            else:
                nsamp = max(1, len(insts) // self.sample_rate)
                for _ in range(nsamp):
                    s,e = random.choice(insts)
                    rows[ s : e ] = True

        fit_rows = rows
        tune_rows = ~rows

        # Now get the subset for classifier training
        fitX = X[fit_rows,:]
        fitY = yVec[fit_rows]

        tuneX = X[tune_rows,:]
        tuneY = yVec[tune_rows]
        tuneYMat = yMat[tune_rows,:]

        self.model.fit(fitX, fitY)

        yPred = self.model.predict_proba(tuneX)

        self.points = [ DensityProb(self.classes_, i, tuneYMat[:,i], yPred[:,i]) for i in range(yPred.shape[1]) ]

        for p in self.points:
            print("%s | f1 %0.3f | cut %0.3f" % (p.label.ljust(20), p.bestScore, p.bestCut))

    def multi_pred(self, X):
        probs = self.model.predict_proba(X)
        for i,p in enumerate(self.points):
            probs[:,i] = p(probs[:,i])
        return probs

    def predict_proba(self, X):
        probs = self.model.predict_proba(X)
        for i,p in enumerate(self.points):
            probs[:,i] = p(probs[:,i])
        
        return probs

    def predict(self, X):
        return self.model.predict(X)
        