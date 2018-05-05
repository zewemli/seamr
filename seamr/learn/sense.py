import numpy as np
import seamr
from seamr import core
from seamr import spatial
from tqdm import tqdm

from seamr.learn import ClassifierBase


class SensibleClassifier( ClassifierBase ):
    def __init__(self,
                 key, 
                 store,
                 dset,
                 estimator,
                 semthreshold=0.1,
                 weight_samples=False,
                 factors={"objects": "pclass"}):
        
        self.key = key
        self.sensors = store.get_sensors(dset)
        self.model = core.build(estimator)
        self.factorDef = factors
        self.factors = {}
        self.semthreshold = semthreshold
        self.weight_samples = weight_samples
        self.epsilon = 1.0 / (10 ** 6)

        self.res = spatial.Reasoner( self.sensors )

        self.objKinds = { o.name : {o.name, o.type, o.space} for o in self.res.getObjBoxes() }
        
        oIndex = set()
        for s in self.objKinds.values():
            oIndex |= s
        self.objects_ = np.array( sorted(filter(bool, oIndex)) )

        self.invObj = self.rev(self.objects_)
        
    def __str__(self):
        return self.key

    def rev(self, strings):
        return { c:i for i,c in enumerate( strings ) }

    def normBy(self, mat, n, method):
        """
        Mat is a count matrix of shape item x class
        n is the actual number of observations (each observation can have 0+ associations)
        """

        assert n > 0, "N is 0, check"

        if method == "pclass":
            return mat / (self.epsilon + mat.sum(axis=0, keepdims=True))
        
        elif method == "pitem":
            return mat / (self.epsilon + mat.sum(axis=1, keepdims=True))
        
        elif method == "npmi":
            eps = 1.0 / (10 ** 5)
            npmi = np.zeros_like( mat )

            pj = np.log1p(mat.sum(axis=0) / n)
            pi = np.log1p(mat.sum(axis=1) / n)

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    pco = np.log1p(mat[i,j] / n) + eps
                    pmi = pco - pi[i] - pj[j]
                    npmi[i,j] = pmi / -pco
            return npmi

        else:
            raise ValueError("Don't know how to norm by %s. Options are 'pclass', 'pitem', and 'npmi'" % method)

    def get_sensor_mat(self, events, times):
        sSensors = self.factors["sensor"][1]
        M = np.zeros( (len(times), sSensors.shape[0]) )

        for i,(sensors,t) in enumerate(core.gen_active_sensors(events, times)):
            if len(sensors) == 0:
                M[i, -1] = 1.0
            else:
                M[i, list(sensors)] = 1.0
        
        return M
        
    def get_semantic_mat(self, events, times):
        sObjects = self.factors["object"][1]
        M = np.zeros( (len(times), sObjects.shape[0]) )

        for i,(patches, t) in enumerate(self.res.genPatches(events, times)):
            M[i, self.patchIndices(patches)] = 1.0

        return M
            
    
    def get_time_mat(self, events, times):
        sTimes = self.factors["time"][1]
        M = np.zeros( (len(times), sTimes.shape[0]) )

        for i,t in enumerate(times):
            M[i, int((t % 86400) // 3600)] = 1.0

        return M

    def patchIndices(self, patches):
        tSet = set()
        for n,o,v in patches:
            if v > self.semthreshold:
                tSet |= self.objKinds[n]
        
        if len(tSet) == 0:
            indices = [-1]
        else:
            indices = [ self.invObj[k] for k in tSet if k ]
        return indices

    def predict_proba(self, X):
        events, times, mat = X
        probs = self.model.predict_proba(mat)

        factors = []

        for func, weight in self.factors.values():
            factors.append( np.dot(func(events, times), weight) )
            
        if len(factors):
            newProbs = probs * np.stack( factors ).mean( axis=0 )
        else:
            newProbs = probs

        pSums = newProbs.sum(axis=1)
        if (pSums == 0).sum() > 0:
            newProbs[ pSums == 0, self.classIndex['other'] ] = 1.0
        
        return newProbs / newProbs.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        return self.classes_[ self.predict_proba(X).argmax(axis=1) ]

    def fit(self, X, y):
        self.setClasses(y)
        
        events, times, mat = X
        sample_weight = None
        
        n = len(times)
        nc = self.classes_.size
        ns = len(self.sensors)
        ax = 0

        if "time" in self.factorDef:
            cTimes = np.zeros( (24, nc) )
            
            tmp = np.zeros_like(self.sTimes)

            for t,lbl in tqdm(zip(times, y), desc="Sensible: Fitting Times", total=n):
                hr = int((t % 86400) // 3600)
                ilabel = self.classIndex[ lbl ]
                tmp[hr, ilabel] += 1
            
            # Smooth over the previous and next hour
            for c in range(nc):
                if (c+1) == nc:
                    cols = [c-1, c, 0]
                else:
                    cols = [c-1, c, c+1]

                cTimes[:, c] = tmp[:, cols].mean(axis=1)

            self.factors['time'] = (self.get_time_mat, self.normBy(cTimes, n, self.factorDef['time']))

        if "sensor" in self.factorDef:
    
            cSensors = np.zeros((ns+1, nc))
            pStream = zip(core.gen_active_sensors(events, times), y)

            for (sensors, t), lbl in tqdm(pStream, desc="Sensible: Fitting Sensors", total=n):
                # sensors is a set of ints
                ilabel = self.classIndex[lbl]
                if len(sensors):
                    cSensors[list(sensors), ilabel] += 1
                else:
                    cSensors[-1, ilabel] += 1
            
            self.factors["sensor"] = (self.get_sensor_mat, self.normBy(cSensors, n, self.factorDef['sensor']))
    
        if "object" in self.factorDef:       
            cObjects = np.zeros( (self.objects_.size + 1, nc) )

            pStream = zip(self.res.genPatches(events, times), y)

            for (patches, t),lbl in tqdm(pStream, desc="Sensible: Fitting Semantics", total=n):
                indices = self.patchIndices(patches)
                
                ilabel = self.classIndex[lbl]
                cObjects[ indices, ilabel ] += 1
            
            self.factors["object"] = (self.get_semantic_mat, self.normBy(cObjects, n, self.factorDef['object']))

        if self.weight_samples and len(self.factors):
            factors = []
            for func, weight in self.factors.values():
                factors.append( np.dot(func(events, times), weight) )
            weights = np.stack(factors).mean(axis=0)
            indices = [ self.classIndex[l] for l in y ]
            sample_weight = weights[:, indices]

        with seamr.BlockWrapper("Sensible: Learning classifier"):
            try:
                self.model.fit(mat, y, sample_weight = sample_weight)
            except: # Maybe this models doesn't support sample_weight?
                self.model.fit(mat, y)

        return self