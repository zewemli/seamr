import re
import os
import sys
import gzip
import json
import csv
import pickle
import argparse
from random import randint
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter, defaultdict
from itertools import groupby, combinations

import seamr
from seamr.sem import COSE
from seamr import spatial
from seamr import core
from seamr import sem
from seamr import load_config
from seamr import evaluate

from scipy.stats import entropy

from sklearn import cluster

from hmmlearn import hmm
from pomegranate import *

def seqEntropy(seq):
    sq = Counter()
    for x in seq:
        sq.update(x)
    return entropy([ n for k,n in sq.items() ]) / np.log( len(sq) )

def getDayLabelSeq(store, dset, days):
    envLabels = store.get_labels(dset, days=days)

    nActs = max(set(l.actID for l in envLabels.labels)) + 1

    resSeq = {}

    for res in envLabels.getResSubsets():
        
        sequences = []
        sDays = []
        for day,dayActs in groupby(res.labels, key=lambda l: l.day):
            daySeq = []
            for l in dayActs:
                daySeq.append( l.actID )

            sequences.append( daySeq )
            sDays.append(day)

        resSeq[ res.labels[0].resident ] = (sDays, sequences)
    
    return resSeq

def filterAndCondense(trainSeq, testSeq):
    cnt = Counter()
    for s in trainSeq:
        cnt.update( s )

    remaps = { c:i for i,(c,n) in enumerate(cnt.most_common()) }

    for seq in [trainSeq, testSeq]:
        for i in range(len(seq)):
            seq[i] = [ [remaps.get(x,0)] for x in seq[i] ]
    
    return (trainSeq, testSeq)

def modelSemanticSeq(args):
    store, dset, days, assoc = args
    
    pAct = defaultdict(Counter)
    pConcept = defaultdict(Counter)

    for ds, dCounts in assoc.items():
        if ds != dset:
            for act,actConcepts in dCounts.items():
                pConcept[act].update(actConcepts)
                for concept,cval in actConcepts.items():
                    pAct[concept][act] += cval
    
    for c in list(pAct.keys()):
        pAct[c] = toProb(pAct[c])

    for k in list(pConcept.keys()):
        pConcept[k] = toProb(pConcept[k])

    cose = sem.COSE()

    actMap = {}
    for k in set(sem.activity_type_map.values()):
        klow = k.split(":")[-1].lower()
        actMap[k] = klow
        actMap[klow] = k

    days = list(range(days))
    events = lambda: store.get_events(dset, days=days)
    times = core.get_times(events, step = 15.0, max_steps = 20)

    reasoner = spatial.Reasoner(store.get_sensors(dset))
    models = cose.get_activity_models()

    envLabels = store.get_labels(dset, days=days)
    resLabels = envLabels.make_dataset(times, prefer_mat = True)

    envHasActs = { l.activity for l in envLabels.labels }
    hasModel = sorted({ k for k in models.keys() if actMap[k] in envHasActs })

    print("%s | Loaded models: %s" % (dset, sorted(hasModel) ))

    acts = sorted({"other"} | { l.activity for l in envLabels.labels })
    actID = { c:i for i,c in enumerate(acts) }
    
    # -----------
    obsExp = list()
    obsInfer = list()
    for sensors,ts in tqdm(core.gen_active_sensors(events, times, maxActive = 300), total=len(times)):
        
        inferRow = np.zeros( len(hasModel) )
        expRow = np.zeros( len(hasModel) )
        
        for p in reasoner.getScores(sensors = sensors):
            if p.kind:
                for i,act in enumerate(hasModel):
                    inferRow[i] += pAct[ p.kind ].get(act, 0.0) * v
                    expRow[i] += pConcept[act].get(p.kind, 0.0) * v

        obsExp.append( expRow )
        obsInfer.append( inferRow )
    # -----------
    labels = [ [sem.activity_type_map[x] for x in T] for T in zip(*resLabels) ]

    classEst = np.array(obsExp) * np.array(obsInfer)
    
    reals = np.zeros_like( classEst )
    rIndex = { h:i for i,h in enumerate(hasModel) }
    for row, T in enumerate(labels):
        for c in T:
            if c in rIndex:
                reals[row, rIndex[c]] = 1

    roc = evaluate.ROCAnalysis( sorted(hasModel) )

    return dset, roc.setup(dset, reals, classEst)

def toProb(d):
    s = sum(d.values())
    return { k : n/s for k,n in d.items() }
#
# ========================================================================
#


parser = argparse.ArgumentParser(description="Evaluate the HMM")
parser.add_argument("store", help="Data store location")
parser.add_argument("semAssoc", help="Semantic associations file")
parser.add_argument("--days",'-d', type=int, default=14, help="Number of training days")
parser.add_argument("--test",'-e', type=int, default=10**3, help="Number of test days")
parser.add_argument("--n_components", "-n", default=64, type=int, help="Components for HMM")

 
args = parser.parse_args()
store = core.Store(args.store, use_sem_labels=True)

with open(args.semAssoc) as f:
    assoc = json.load(f)

with Pool() as p:
    #store.datasets()
    jobs = [ (store, ds, args.days, assoc) for ds in sorted(['hh104']) ]

    joints = {}
    #p.imap_unordered
    for ds,r in map(modelSemanticSeq, jobs):
        print(ds)
        r.show(dataset=ds).save("./")
    
    exit()
    # Reasoner.typeAdjust
    with open("semAssoc.json", "wt") as f:
        json.dump(joints, f, indent=2)
    exit()
    with open("label_probs.csv", "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["rel", "dataset", "resident", "n_components", "day", "entropy", "logprob"])
        for rows in p.imap_unordered(modelLabelSeq, jobs):
            writer.writerows( rows )

