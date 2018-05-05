import seamr
from seamr import sem
from seamr import spatial
import math

class CrossKey:
    def key(self, p):
        return p.kindKey

class EnvKey:
    def key(self, p):
        return p.objKey

class ModelExp:
    def __init__(self, store, cls):
        self.cls = cls
        self.store = store
        self.model = sem.COSE().get_activity_models()[cls]
        self.concepts = { c.split(":")[-1] for c in self.model.concepts() }
    
    def __hash__(self):
        return hash(frozenset(self.concepts))

    def __call__(self, patchSource):

        for patches,ts in patchSource:
            yield self.matchValue(patches)
            
    def matchValue(self, patches):
        total = 0.0

        for p in patches:
            pKey = self.key(p).split("-")[0]
            if pKey in self.concepts:
                total += p.value
        
        return total
        
    def __bool__(self):
        return len(self.concepts) > 0

    def isMerged(self):
        return False

    def getErrors(self, patchSource, binaryLabels):
        errors = {}
        for i,(patches,ts) in enumerate(patchSource):

            if binaryLabels[i] == 1:
                ok = set()
                for p in patches:
                    pKey = self.key(p).split("-")[0]
                    ok.add( pKey )
                
                for k in ok - self.concepts:
                    errors[k] = errors.get(k,0) + 1
        return errors

class DataExp:
    def __init__(self, store, cls):
        self.store = store
        self.cls = cls
        self.dist_false = {}
        self.dist_true = {}
        self.n_true = 0.0
        self.n = 0
        self._merged = False
        
    def update(self, patchSource, binaryLabels):
        
        self.n_true = binaryLabels.sum()

        for i,(patches, ts) in enumerate(patchSource):
            self.n += 1
            isTrue = binaryLabels[i] == 1.0
            for p in patches:
                pKey = self.key(p)
                
                if isTrue:
                
                    try:
                        self.dist_true[ pKey ] += 1
                    except:
                        self.dist_true[ pKey ] = 1
                else:

                    try:
                        self.dist_false[ pKey ] += 1
                    except:
                        self.dist_false[ pKey ] = 1
        
        return self
    
    def toProb(self, d, s):
        return {k : v/s for k,v in d.items()}

    def merge(self, other):
        newOne = self.__class__( self.store, self.cls )
        newOne.dist_false.update( other.dist_false )
        newOne.dist_true.update( other.dist_true )

        for dest,src in [ (newOne.dist_false, self.dist_false), (newOne.dist_true, self.dist_true) ]:
            for k,v in src.items():
                dest[k] = dest.get(k, 0) + v

        newOne.n = self.n + other.n
        newOne.n_true = self.n_true + other.n_true

        newOne._merged = True

        return newOne

    def isMerged(self):
        return self._merged

    def __call__(self, patchSource):
        
        # Do some good turing smoothing of the conditional distribution
        # ---
        p_true = self.toProb(self.dist_true, self.n_true)
        p_false = self.toProb(self.dist_false, self.n - self.n_true)
        
        p_lbl = self.n_true / self.n

        for patches,ts in patchSource:

            if len(patches) == 0:
                yield 1.0 / self.n

            else:

                pTotal = 0.0
                for pt in patches:
                    pKey = self.key(pt)

                    p_obs_lbl = p_true.get(pKey, 0.0)

                    if p_obs_lbl > 0:
                        pTotal +=  p_obs_lbl / p_lbl

                yield pTotal

# --------------------------------------------------------

class CrossDataExp(DataExp, CrossKey):
    pass

class CrossModelExp(ModelExp, CrossKey):
    pass

class EnvDataExp(DataExp, EnvKey):
    pass
