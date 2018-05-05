import gzip
import os, sys
import csv
import pickle
from itertools import groupby
from collections import Counter, defaultdict, namedtuple
from uuid import uuid4
import math
import json
import re
import abc
from tabulate import tabulate

from scipy.spatial import ConvexHull, distance
from hmmlearn.hmm import MultinomialHMM

import numpy as np

from prg import prg
from sklearn import metrics

from scipy.stats import rankdata
from hmmlearn import hmm
from seamr import sem
import seamr
from seamr import core

import pickle

shortName = {
        "CASASOneHotFeatures": "sparse-casas",
        "CASASFeatures": "casas",

        "CASASTimeMiFeatures": "casas-time",
        "CASASTimeFeatures": "casas-mi-time",
        "CASASMiFeatures": "casas-mi",

        "SingleEnvFeatures": "object",
        "CrossEnvFeatures": "concept",
        "StateFeatures": "state",
        "RuleGroupFeatures" : "rules",
        "CRFWrapped" : "crf-rules"
    }

class ConfPoint:
    rowHeader = [
        "name",
        "p",
        "n",
        "tp",
        "fp",
        "fn",
        "tn",
        "specificity",
        "precision",
        "recall",
        "npv",
        "fpr",
        "fdr",
        "f1"
    ]
    def __init__(self, name, p, n, tp, fp, fn, tn):
        self.name = name
        self.p = p
        self.n = n
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.tn = tn
    
    @property
    def specificity(_):
        return _.tn / (_.tn + _.fp)
    
    @property
    def precision(_):
        return _.tp / (_.tp + _.fp)
    
    @property
    def npv(_):
        return _.tn / (_.tn + _.fn)
    
    @property
    def fpr(_):
        return _.fp / (_.tn + _.fn)
    
    @property
    def recall(_):
        return _.tp / (_.tp + _.fn)
    
    @property
    def fdr(_):
        return _.fp / (_.tp + _fp)
    
    @property
    def f1(_):
        r = _.recall()
        p = _.precision()
        return 2 * (r * p) / (r + p)
    
    def writeRow(self, csvWriter):
        row = [ getattr(self, prop) for prop in self.rowHeader ]
        csvWriter.writerow(row)
        return self


class PerfBase:

    def __init__(self):
        self.name = None
        self.params = {}
        self.roc = None
        self.wEv = {}
        self.wOcc = {}
        self.confusion = {}
        self.nExamples = 0

    def __getitem__(self, key):
        return getattr(self, key, self.params.get(key, None))

    def __len__(self):
        return self.nExamples

    def getSchemes(self):
        return ["micro","occ","macro"]

    def weigh(self, base, scheme):

        if scheme is None:
            return base
        else:
            w = None
            if scheme == "micro":
                w = self.wEv
            elif scheme == "occ":
                w = self.wOcc
            elif scheme == "macro":
                w = dict([ (k, 1.0) for k,v in base.items() ])
            else:
                raise NotImplementedError("I don't know anything about the [%s] aggregation scheme."\
                                            +" Try, micro, macro, or occ" % scheme)

            n = sum(w.values()) or 1.0

            s = 0.0
            for k,v in base.items():
                s += v * (w.get(k, 0.0) / n)
            return s

    @staticmethod
    def rekey(root):
        if isinstance(root, dict):
            return dict([ (str(getattr(k, "__name__", k)), PerfBase.rekey(v),) for k,v in root.items() ])

        elif isinstance(root, list):
            return list(map(PerfBase.rekey, root))

        elif type(root) is abc.ABCMeta:
            return root.__name__
            
        else:
            return root

    def setParams(self, params, roc=None, **kw):
        self.params = {}
        self.params.update(params)
        self.params.update(kw)
        self.roc = roc
        return self
    
    def addParam(self, **kw):
        self.params.update(kw)
        return self
    
    def save(self, to_folder, **kw):

        if not os.path.exists(to_folder):
            os.makedirs(to_folder)

        as_dct = self.rekey(dict([ (n, getattr(self, n)) for n in self.storeVars ]))

        nicename = re.sub(r"[\{\}\]\[]+", "-", self.name)

        for k,v in sorted(kw.items()):
            nicename += "-{}[{}]".format(k,v)

        fname = ("%s.%s.json.gz" % (nicename, str(uuid4())))
        with gzip.open(os.path.join(to_folder, fname), "wb") as f:
            f.write( json.dumps(as_dct, sort_keys=True).encode("UTF-8") )
            
        return self


class Perf(PerfBase):
    storeVars  = ["name", "params", "wEv", "wOcc", "confusion"]
            
    def getMetric(self, f, scheme = None):
        p = {}
        for k in self.confusion.keys():
            
            tp = self.confusion.get(k,{}).get(k, 0.0)
            fp = sum([ d.get(k, 0.0) for alt,d in self.confusion.items() if alt != k ])
            fn = sum([ cnt for alt,cnt in self.confusion.get(k,{}).items() if alt != k ])

            try:
                p[k] = f(tp, fp, fn)
            except ZeroDivisionError:
                p[k] = 0.0

        for k in self.confusion.keys():
            if k not in p:
                p[k] = 0.0
        
        return self.weigh(p, scheme)

    def accuracy(self):
        n = 0
        tp = 0
        for a,d in self.confusion.items():
            n += sum(d.values())
            tp += d.get(a,0)
        return tp/n 

    def precision(self, agg = None):
        return self.getMetric(lambda tp,fp,fn: tp / (tp + fp), agg)

    def recall(self, agg = None):
        return self.getMetric(lambda tp,fp,fn: tp / (tp + fn), agg)

    def fbeta(self, beta=1.0, agg = None):
        def fb(tp, fp, fn):
            b2 = beta * beta
            top = (1+b2) * tp
            bottom = ((1+b2) * tp) + (b2 * fn) + fp
            return top / bottom

        return self.getMetric(fb, agg)

    def f1(self, agg = None):
        return self.fbeta(1.0, agg = agg)

    def setup(self, name, reals, preds, times):

        if isinstance(reals, np.ndarray):
            reals = reals.ravel()

        if isinstance(preds, np.ndarray):
            preds = preds.ravel()
        
        if isinstance(reals, np.ndarray) and isinstance(preds, np.ndarray):
            assert reals.shape == preds.shape, ("Reals and predictions don't match", reals.shape, preds.shape)    

        self.nExamples = len(reals)

        self.name = name
        wEv  = Counter(reals)
        wOcc = Counter()

        conf = dict([ (k, Counter()) for k in wEv ])
        for r,items in groupby(zip(reals, preds), key=lambda T: T[0]):
            wOcc[r] += 1
            cr = conf[r]
            cr.update([ p for _,p in items ])

        self.wEv = dict(wEv)
        self.wOcc = dict(wOcc)    
        self.confusion = dict([ (k, dict(d.items())) for k,d in conf.items() ])
        
        return self


    def load(self, from_file, load_full=False):
    
        with gzip.open(from_file, "rb") as f:
            for k,v in json.loads( f.read().decode("UTF-8") ).items():
                setattr(self, k, v)
            
        return self

    def show(self, with_confusion = True, **kw):
        ext = ""
        if len(kw):
            ext = "| %s" % "|".join([ "%s : %s" % T for T in sorted(kw.items()) ])

        real = Counter()
        pred = Counter()
        sm = 0
        for r,d in self.confusion.items():
            for p,n in d.items():
                real[r] += n
                pred[p] += n
                sm += n
        
        crossf1 = self.f1()

        rtbl = [ [k, "%0.3f%%" % (100*real.get(k, 0)/sm,),
                     "%0.3f%%" % (100*pred.get(k, 0)/sm),
                     real.get(k, 0),
                     pred.get(k, 0),
                     "%0.3f" % crossf1.get(k, 0) ]
                    
                 for k in sorted( set(real.keys()) | set(pred.keys()) ) ]

        print("\n\n"+("#"*50))
        print( tabulate(rtbl, headers=["Name","P(real)", "P(pred)", "N(real)", "N(pred)", "F1"]) )
        print("-"*50)

        for agg in ["micro","occ","macro"]:
            print("%s | %18s | Accuracy %0.2f%% | %s | F1 %0.3f | Prec %0.3f | Recall %0.3f | %s" % (
                    self.params.get("relation",""), 
                    self.name,
                    100 * self.accuracy(),
                    agg.rjust(6," "),
                    self.f1(agg = agg),
                    self.precision(agg = agg),
                    self.recall(agg = agg), 
                    ext))


        if with_confusion:

            f1 = self.f1()
            prec = self.precision()
            rec = self.recall()

            for r,d in sorted(self.confusion.items()):
                sm = sum(d.values())
                print("-----\nF1 %0.3f | Prec %0.3f | Recall %0.3f | %s" % 
                        (f1.get(r, 0.0), prec.get(r, 0.0), rec.get(r, 0.0), r))
                
                for p,n in sorted(d.items()):
                    print("\t%s : %0.3f%% : %s" % (p, 100*n/sm, n))

            print("*"*50)
            
        return self

class SequenceProb:
    def __init__(self, n_components=32):
        self.n_components = n_components
        self.name = None
        self.realSeqProb = 0.0
        self.predSeqProb = 0.0

    def getSegments(self, seq, sleepActs):
        i=0
        lens = []
        wasAsleep = False  
        for j in range(len(seq)):
            if wasAsleep and (seq[j] not in sleepActs) and (j - 1) > 1:
                lens.append(j - i)
                i=j
            wasAsleep = seq[j] in sleepActs

        lens.append( len(seq) - sum(lens) )
        assert sum(lens) == len(seq), (sum(lens), len(seq))
        return [ x for x in lens if x > 0 ]

    def to2D(self, seq):
        m = np.array(seq, dtype=np.int32)
        return np.expand_dims(m, 1)

    def getScores(self, model, seq, lens):
        scores = []
        s=0
        for e in lens:
            scores.append( model.score( self.to2D(seq[s:s+e]) ) )
            s += e
        return scores

    def setup(self, reals, preds):

        cls = sorted(set(reals))
        index = { k:i for i,k in enumerate(cls) }
        sleepActs = { k for k,v in sem.activity_type_map.items() if v == "activity:Sleeping" }
        sleepActs.add("activity:Sleeping")

        sleepActIDS = { index[k] for k in sleepActs if k in index }

        with seamr.BlockWrapper("Fitting HMM for estimating sequence probs"):
            model = hmm.MultinomialHMM(n_components = self.n_components)
    
        distReals = [index[g] for g,_ in groupby(reals)]
        distPreds = [index[g] for g,_ in groupby(preds) if g in index]

        realLens = self.getSegments(distReals, sleepActIDS)
        predLens = self.getSegments(distPreds, sleepActIDS)

        model.fit(self.to2D(distReals), realLens)

        self.realScores = self.getScores(model, distReals, realLens)
        self.predScores = self.getScores(model, distPreds, predLens)
        return self
    
    def show(self, **kw):

        print("\nProbs: real %0.3f +/- %0.3f || pred %0.3f +/- %0.3f\n" % (
            np.mean(self.realScores), np.std(self.realScores),
            np.mean(self.predScores), np.std(self.predScores)
        ))

        return self

    def toJSON(self):
        return {"predScores": self.predScores, "realScores": self.realScores }

class ROCAnalysis:

    def __init__(self, classes):
        self.classes_ = np.array(classes, copy=True)
        try:
            self.other = [ c.lower() == "other" for c in self.classes_ ].index(True)
        except ValueError:
            self.other = self.classes_.size

        self.roc_curves = {}
        self.pr_curves = {}
        self.pr_gain_curves = {}

        self.avg_prec = np.zeros( self.classes_.size )
        self.auc = np.zeros( self.classes_.size )
        self.auprg = np.zeros( self.classes_.size )

        self.occ_weight = np.zeros( self.classes_.size )
        self.micro_weight = np.zeros( self.classes_.size )

        self.params = {}
        self.name = None
        self.sampleSize = 100

    @staticmethod
    def digitize(predsRaw, nbins):
        step = 1.0 / nbins
        maxBin = nbins - 1
        preds = np.zeros_like( predsRaw )
        assert np.isfinite(nbins), "Bins not finite (%s)" % nbins
        
        for i in range(preds.shape[0]):
            preds[i] = min(int( predsRaw[i] / step ), maxBin)
        
        return preds

    def getClassPerf(self):
        perfs = []

        if getattr(self, "_scalar_params", None) is None:
            self._scalar_params =  { k:v for k,v in self.params.items() if type(v) not in [list,dict] }

            if isinstance(self.params.get("features", None), dict):
                cls,clsArgs = list(self.params["features"].items())[0]
                self._scalar_params['features'] = shortName[cls.__name__.split(".")[-1]]

            if isinstance(self.params.get("classifier", None), dict):
                cls,clsArgs = list(self.params["classifier"].items())[0]
                self._scalar_params["classifier"] = clsArgs.get("key",str(cls))

        for i,c in enumerate(self.classes_):
            iparam = {
                "activity": c,
                "roc": self.auc[i],
                "avg_prec": self.avg_prec[i],
                "auprg": self.auprg[i],
                "optimalF1": self.optimalF1[i],
                "f1Curve" : self.f1Curve[i],
                "wOCC": self.occ_weight[i],
                "wMicro": self.micro_weight[i]
            }
            iparam.update( self._scalar_params )
            perfs.append( iparam )
        
        return perfs

    @staticmethod
    def getCHull(a, b, c, optimal=(0.0, 0.0)):

        points = np.vstack([a,b]).T
        if c.shape[0] < points.shape[0]:
            points = points[0:c.shape[0],:]

        hull = ConvexHull( points )
        hPoints = hull.points[hull.vertices]

        return hPoints[:,0], hPoints[:,1], c[hull.vertices]
            
    def setup(self, name, reals, probs):
        self.name = name

        if isinstance(reals, list) or reals.ndim == 1:
            cidx = { c:i for i,c in enumerate(self.classes_) }

            xr = np.zeros_like(probs)
            for row,lbl in enumerate(reals):
                if lbl in cidx:
                    xr[row, cidx[lbl]] = 1.0
            reals = xr

        assert isinstance(reals, np.ndarray)
        assert isinstance(probs, np.ndarray)
        assert reals.shape == probs.shape

        # ---------------------------------
        self.micro_weight = reals.sum(axis=0)
        self.occ_weight = np.zeros( self.classes_.size )

        self.avg_prec = np.zeros( self.classes_.size )
        self.auc = np.zeros( self.classes_.size )
        self.auprg = np.zeros( self.classes_.size )
        self.optimalF1 = np.zeros( self.classes_.size )
        self.f1Curve = [ None for _ in range(self.classes_.size) ]

        try:
            self.lrap = metrics.label_ranking_average_precision_score( reals, probs )
        except:
            self.lrap = 0.0
        
        try:
            self.rankloss = metrics.label_ranking_loss( reals, probs )
        except:
            self.rankloss = float("inf")

        for c in range(reals.shape[1]):
            if c != self.other:
                self.occ_weight[c] = (reals[1:, c] != reals[:-1, c]).sum()

                rVec, pVec = reals[:, c], probs[:, c]

                className = self.classes_[c]

                try:
                    fpr,tpr,thresholds = self.getCHull(*metrics.roc_curve( rVec, pVec ))
                    self.roc_curves[c] = np.stack([fpr,tpr,thresholds])
                except:
                    pass

                try:
                    p,r,t = self.getCHull(*metrics.precision_recall_curve(rVec, pVec))
                    self.pr_curves[c] = np.stack([ p[:t.shape[0]], r[:t.shape[0]], t ])
                except:
                    pass
                
                self.auprg[c] = max(0, prg.calc_auprg( prg.create_prg_curve(rVec, pVec) ))

                cF1Curve = core.OptimalF1(rVec, pVec)
                self.optimalF1[c] = cF1Curve.bestF1
                self.f1Curve[c] = list(cF1Curve.curve)

                try:
                    self.auc[c] = metrics.roc_auc_score(rVec, pVec)
                    self.avg_prec[c] = metrics.average_precision_score(rVec, pVec)
                except (ValueError, FloatingPointError):
                    self.auc[c] = 0.0
                    self.avg_prec[c] = 0.0
    
        self.micro_weight /= self.micro_weight.sum()
        self.occ_weight   /= self.occ_weight.sum()

        return self
    
    def setParams(self, params = {}, **kw):
        p = {}
        p.update(params)
        p.update(kw)
        self.params = p
        return self

    def show(self, **kw):
        params = " ".join([ "%s:%s" % (k, kw[k]) for k in sorted(kw.keys()) ])
        
        print("-"*10, self.name, "-"*10)
        print("Params: %s" % params)
        for meas,vec in [("Avg Prec", self.avg_prec), ("AUC", self.auc), ("AUPRG", self.auprg)]:
            macro = vec.sum() / (vec.size - 1)
            micro = ( vec * self.micro_weight ).sum()
            occ = ( vec * self.occ_weight ).sum()
            print("\t%s | macro %0.3f | micro %0.3f | occ %0.3f" % (meas, macro, micro, occ))
        
        print("----------------------")
        for (c,av,au, optF1) in zip(self.classes_, self.avg_prec, self.auc, self.auprg):
            print("prec %0.4f | auc %0.4f | Opt F1 %0.4f | %s" % (av, au, optF1, c))

        print("-" * 30)

        return self

    def save(self, to_folder, **kw):
        if not os.path.exists(to_folder):
            os.makedirs(to_folder)

        store = os.path.join(to_folder, '%s-%s.roc.pkl' % ( self.name, str(uuid4()) ))

        with open(store, "wb") as f:
            pickle.dump(self, f)

        return self