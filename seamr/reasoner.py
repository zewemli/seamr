import os
import sys
from itertools import groupby
from collections import defaultdict, Counter, deque
from tqdm import tqdm

import seamr
from seamr import core
from seamr.sem import COSE, activity_type_map
from seamr import spatial
from seamr.features.semantic import CrossEnvFeatures

import numpy as np
from hmmlearn import hmm
import random

from dtaidistance.dtw import distance_fast as dtw_distance
from scipy.stats import entropy

from multiprocessing import Pool

epsilon = 1.0 / (10 ** 5)

def toProb(d, n=None):
    if n is None:
        n = sum(d.values())
    
    return { k : v/n for k,v in d.values() }

class SeqEstimator:
    def __init__(self, count, seq, times, randArgs, hmm_components, prefix=[]):
        self.prefix = prefix

        baseSeq = self.makeSeq(seq, times)
        
        i = 0
        l = random.randint( *randArgs )
        lens = []
        while (i + l) < baseSeq.shape[0]:
            i += l
            lens.append(l)
            l = random.randint( *randArgs )

        if sum(lens) < baseSeq.shape[0]:
            lens.append( baseSeq.shape[0] - sum(lens) )
        
        extSeq = np.zeros( (baseSeq.shape[0] + count, 1) )
        extSeq[ 0:baseSeq.shape[0], : ] = baseSeq
        for i in range(count):
            extSeq[ baseSeq.shape[0] + i , 0 ] = i
        lens.append(count)

        self.hmm = hmm.MultinomialHMM(n_components = hmm_components)
        self.hmm.fit( extSeq.astype(np.int32), lens )
    
    def makeSeq( self, seq, times ):
        seqTimes = [ (g, sorted([ ts for _,ts in gItems ])) for g, gItems in groupby(zip(seq, times), key=lambda T: T[0]) ]

        # Ensures that each symbol occurs at least once
        baseSeq = []
        for g, gTimes in seqTimes:
            minutes = (gTimes[-1] - gTimes[0]) / 60.0
            n = 1 + int(np.log1p(minutes))
            
            baseSeq.extend( [ g ] * n )
        
        baseSeq = np.array(baseSeq, dtype=np.int32)
        baseSeq = np.expand_dims(baseSeq, -1)
        
        return baseSeq

    def __call__(self, seq):
        return self.hmm.score( seq ) / len(seq)

class EnvModel:
    
    def __init__(self,
                 dset,
                 store,
                 days,
                 step_by,
                 max_steps,
                 hmm_components,
                 binaryClassifier,
                 featureArgs,
                 randArgs = (8,16) ):

        self.hmm_components = hmm_components
        self.cose = COSE()
        self.step_by = step_by
        self.max_steps = max_steps
        self.randArgs = randArgs
        self.dataset = dset
        self.env = dset.split(".")[0]
        self.store = store
        self.featArgs = featureArgs
        self._clsDef = binaryClassifier
        self._weight = 0

        if isinstance(days, int):
            days = list(range(days))

        # Get all labels
        events, times = self.getEventsAndTimes( days )
        
        self._weight = len(times)

        allLabels = store.get_labels(dset, days=days)
        self.classes_ = np.array( sorted(set( l.activity for l in allLabels.labels )) )
        self.classIndex = { c:i for i,c in enumerate(self.classes_) }
        self.classLevel = np.ones( self.classes_.size )

        resLabels = allLabels.make_dataset(times, prefer_mat = True)
        rSet = set()
        for res in resLabels:
            rSet |= set(res)

        self.models = { c : [] for c in rSet }

        features = CrossEnvFeatures( store, dset, **featureArgs )
        X = features.make_dataset( events, times )

        self.actDist = {}
        # ------------------------------------------------------
        # First: Fit each class model
        #
        for res in resLabels:
            # Map to portable labels
            classes = set(res)
            for cls in classes:
                if cls not in self.actDist:
                    self.actDist[ cls ] = np.zeros(24)

            for lbl,tm in zip(res, times):
                hr = int((tm % 86400) // 3600)
                self.actDist[lbl][hr] += 1
        
            for cls in classes:
                if cls != "other":
                    lbls = [ int(r==cls) for r in res ]
                    m = self.getFittedModel( X, lbls )
                    self.models[cls].append( m )
        
        # Set the probability distribution per activity
        for lbl,vCount in list(self.actDist.items()):
            self.actDist[lbl] = vCount / vCount.sum()
        
        # ------------------------------------------------------
        # Second: Fit the HMM Model for space transitions
        #
        self.spaceNames = list(filter(self.isASpace, features.feat_names))
        self.spaceIndex = { c:i for i,c in enumerate(self.spaceNames) }
        self.spaceIndices = [ i for i,s in enumerate(features.feat_names) if self.isASpace(s) ]

        xSpace = X[:, self.spaceIndices]
        domSpaces = xSpace.argmax(axis=1)

        with seamr.BlockWrapper("Fitting sequence estimator for %s" % dset):
            self.seqEst = SeqEstimator(xSpace.shape[1],
                                       domSpaces,
                                       times,
                                       randArgs,
                                       hmm_components,
                                       prefix = list(self.spaceIndex.values()))

    def __hash__(self):
        return hash(self.dataset)

    def getMat(self, start_day, end_day):
        events, times = self.getEventsAndTimes( list(range(start_day, end_day + 1)) )
        features = CrossEnvFeatures( self.store, self.dataset, **self.featArgs )
        return features.make_dataset(events, times), times

    def isASpace(self, typeName):
        try:
            return self.cose.is_a(typeName, "cose:FunctionalSpace")
        except:
            return False

    def getFittedModel(self, X, y):
        return self.getModel().fit(X,y)

    def getModel(self):
        return core.build( self._clsDef )

    def getEventsAndTimes(self, days):
        if not isinstance(days, list):
            days = list(range(days))

        events = lambda: self.store.get_events(self.dataset, days=days)
        times = core.get_times(events, step=self.step_by, max_steps=self.max_steps)

        return events, times

class ActivityReasoner:

    def __init__(self,
                    store,
                    binaryClassifier,
                    featArgs,
                    hmm_components=64,
                    randStepArgs = (8, 16),
                    weight_hours = 2,
                    weight_step = 0.5,
                    step_by=15,
                    max_steps=20,
                    num_level_checks=50,
                    xenv_dist = 'dtw'):
        
        self.xenv_dist = xenv_dist
        self.store = store
        self.cose = COSE()
        self.randStepArgs = randStepArgs

        self.num_level_checks = num_level_checks
        self.weight_hours = weight_hours
        self.weight_step = weight_step
        self.hmm_components = hmm_components
        self.step_by = step_by
        self.max_steps = max_steps

        self._binaryClassifier = binaryClassifier
        self.featArgs = featArgs

    def getActDistance(self, envName, act, probs, times, level):
        
        trueProb = np.zeros(24)

        for t,p in zip(times, probs):
            if p >= level:
                trueProb[ int((t % 86400) // 3600) ] += 1

        trueProb /= trueProb.sum() + epsilon

        d = dtw_distance if self.xenv_dist == 'dtw' else entropy

        altDists = [ d(trueProb + epsilon, e.actDist[act] + epsilon) 
                     for e in self.envs
                     if e.env != envName and act in e.actDist ]

        if len(altDists) > 0:
            return np.min(altDists)
        else:
            return float("inf")

    @staticmethod
    def envFactory(args):

        (dset, store, days, step_by, max_steps, hmm_components, binaryClassifier, featureArgs, randArgs,) = args

        return EnvModel(dset,
                        store,
                        days,
                        step_by,
                        max_steps,
                        hmm_components,
                        binaryClassifier,
                        featureArgs,
                        randArgs = randArgs )

    @staticmethod
    def getEnvMats(args):
        env, start_day, end_day = args
        return env, env.getMat( start_day, end_day )

    def fit(self, datasets, train_for, xenv_days):
        # ------------------------------------------
        # Support multi-label multi-class datasets
        # ------------
        self.envs = []

        factoryArgs = (self.store,
                       train_for,
                       self.step_by,
                       self.max_steps,
                       self.hmm_components,
                       self._binaryClassifier,
                       self.featArgs,
                       self.randStepArgs,)

        with Pool() as p:
            pArgs = [ (ds, *factoryArgs) for ds in datasets ]

            #  
            iterGen = p.imap_unordered(ActivityReasoner.envFactory, pArgs)
            
            for envModel in tqdm(iterGen, total=len(pArgs), desc="Fitting environments"):
                self.envs.append(envModel)
        
        self.envs.sort(key=lambda e: e.dataset)

        for env in self.envs:

            env_probs = self.model_proba(env, 0, xenv_days)
            lows = env_probs.min(axis=0)
            highs = env_probs.max(axis=0)
            
            _, times = env.getEventsAndTimes( list(range(0, xenv_days + 1)) )

            for col in range(env_probs.shape[1]):
                colBreaks = np.linspace(lows[col], highs[col], num=self.num_level_checks, endpoint=False)
                col_probs = env_probs[:, col]

                colCost = float("inf")
                colLevel = 1.0

                for level in colBreaks:
                    levelCost = self.getActDistance(env.env, env.classes_[col], col_probs, times, level)
                    if colCost > levelCost:
                        levelCost = colCost
                        colLevel = level
                env.classLevel[col] = colLevel

        return self

    def qTrim(self, spaces, times):
        tSec = self.weight_hours * 3600
        while len(times) > 1 and (times[-1] - times[0]) > tSec:
            times.popleft()
            spaces.popleft()
        assert len(times) == len(spaces)
        return spaces,times

    def getXEnvWeight(self, baseEnv, spaces, times):
        timeQ = deque([])
        spaceQ = deque([])

        # Takse the end of the sequence and pretends it is the begginning
        # Allows for a quick way to make a "hot-start"
        assoc_sec = self.weight_hours * 3600.0
        assoc_step = self.weight_step * 3600.0

        for i in range(len(times)-1, 0, -1):
            if times[-1] - times[i] >= assoc_sec:
                sDiff = times[-1] - times[i]
                for j in range(i, len(times)):
                    timeQ.append( times[0] + (times[j] - times[-1])  )
                    spaceQ.append( spaces[j] )
                break
        
        xenvWeights = defaultdict(list)

        datasets = sorted([ env for env in self.envs if env.env != baseEnv.env ], key=lambda e: e._weight, reverse=True)
        dsMat = np.zeros( (len(times), len(datasets)) )

        assert times[0] < times[-1]

        sequences = []

        pTime = 0
        rowStart = 0

        for row,(s,t) in enumerate(zip(spaces, times)):
            timeQ.append(t)
            spaceQ.append(s)
            self.qTrim(spaceQ, timeQ)
            
            if (t - pTime) > assoc_step:
                st_seq = baseEnv.seqEst.makeSeq(spaceQ, timeQ)
                sequences.append( (st_seq, rowStart, row + 1) )
                
                rowStart = row + 1
                pTime = t
                # -------------------------------
        sequences.append( (baseEnv.seqEst.makeSeq(spaceQ, timeQ), rowStart, len(times)) )
        
        assert len(times) == sum([ e-s for _,s,e in sequences ])
        
        # ActivityReasoner
        envSequences = [ (i, env, sequences) for i,env in enumerate(datasets) ]
        with Pool() as p:
            vecGen = p.imap_unordered(ActivityReasoner.getCrossEnvWeight, envSequences)
            for i,colVec in tqdm(vecGen, desc="Getting cross-env weights", total=len(envSequences)):
                for segStart, segEnd, segVal in colVec:
                    dsMat[segStart:segEnd, i] = segVal

        w = 1.0 / np.abs(dsMat)
        w /= w.sum(axis=1, keepdims=True) + epsilon

        return { env.dataset : w[:,i] for i,env in enumerate(datasets) }

    @staticmethod
    def getCrossEnvWeight(args):
        i, env, sequences = args
        return i, [ (rowStart, rowEnd, env.seqEst( st_seq )) for st_seq, rowStart, rowEnd in sequences ]

    def model_proba(self, env, start_day, end_day):            

        envMat, envTimes = env.getMat( start_day, end_day )
        envSpaces = envMat[:, env.spaceIndices].argmax(axis=1)

        envSpaceSeq = env.seqEst.makeSeq( envSpaces, envTimes )

        #weights = self.getXEnvWeight(env, envSpaces, envTimes)

        colWeight = np.zeros( (envMat.shape[0], env.classes_.size) )
        probs = np.zeros( (envMat.shape[0], env.classes_.size) )

        for altEnv in self.envs:
            if altEnv.dataset != env.dataset:
                for clsNum, cls in enumerate(env.classes_):
                    if len(altEnv.models.get(cls,[])) > 0:
                        clsProb = np.hstack([mod.predict_proba(envMat)[:,1:2] for mod in altEnv.models[cls] ]).mean(axis=1)
                        
                        probs[:, clsNum] += clsProb
        
        probs[ ~np.isfinite(probs) ] = 0.0

        return probs

    def predict(self, env, start_day, end_day):
        baseProbs = self.model_proba(env, start_day, end_day)
        
        adjProb = np.clip(baseProbs - env.classLevel, 0, 100)
        rowMax = adjProb.max(axis=1)

        mostLikely = env.classes_[ adjProb.argmax(axis=1) ]
        mostLikely[ rowMax < 0 ] = 'other'

        return mostLikely
    # ---------------------------------------
