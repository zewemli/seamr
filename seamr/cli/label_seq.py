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

#
# ========================================================================
#

def modelSensorSeq(args):
    store, dset, n_components, trainDays, testDays = args
    
    events = lambda: store.get_events(dset, days=list(range(trainDays)))
    times = core.get_times(events, step = 15.0, max_steps = 20)

    allSensors = Counter()

    # -----------
    sensor_obs = list()
    for sensors,ts in core.gen_active_sensors(events, times, maxActive = 300):
        day = int(ts // 86400)
        allSensors.update(sensors)
        sensor_obs.append( sensors )
    # -----------
    sensorMap = { c:i for i,c in enumerate(sorted(allSensors.keys())) }

    model = hmm.MultinomialHMM(n_components=n_components)

    mat = np.zeros( (len(times), len(sensorMap)), dtype=np.int32 )
    hours = [ int((ts % 86400) // 3600) for ts in times ]
    lengths = [ len(list(x)) for _,x in groupby(hours) ]

    for r,row in enumerate(sensor_obs):
        cols = [ sensorMap[k] for k in row ]
        mat[r,cols] = 1

    groups = np.expand_dims(cluster.KMeans(n_clusters=64).fit_predict(mat),-1)

    with seamr.BlockWrapper("Fitting HMM To sensors |%s|: %s" % (mat.shape, dset)):
        model.fit(groups, lengths)

#
# ========================================================================
#

def modelObjectSeq(args):
    store, dset, n_components, trainDays, testDays, patchMin = args
    
    events = lambda: store.get_events(dset, days=trainDays)
    times = core.get_times(events, step = 15.0, max_steps = 20)

    allSensors = Counter()
    reasoner = spatial.Reasoner( store.get_sensors(dset) )

    # -----------
    sensor_obs = list()
    for sensors,ts in gen_active_sensors(events, times, maxActive = 300):
        day = int(ts // 86400)

        patches = [ p.objKey for p in reasoner.getScores(sensors = sensors) if p.value >= patchMin ]

        allSensors.update( patches )
        sensor_obs.append( patches )
    # -----------
    
    sensorMap = { c:i for i,c in enumerate(sorted(allSensors.keys())) }

    model = hmm.MultinomialHMM(n_components=n_components)

    mat = np.zeros( (len(times), len(sensorMap)), dtype=np.int32 )
    hours = [ int((ts % 86400) // 3600) for ts in times ]
    lengths = [ len(list(x)) for _,x in groupby(hours) ]

    for r,row in enumerate(sensor_obs):
        cols = [ sensorMap[k] for k in row ]
        mat[r,cols] = 1.0

    with seamr.BlockWrapper("Fitting HMM To objects |%s|: %s" % (mat.shape, dset)):
        model.fit(mat, lengths)
#
# ========================================================================
#

def modelSemanticSeq(args):
    store, dset, n_components, trainDays, testDays, patchMin = args
    
    trainDays = list(range(trainDays))
    testDays = list(range(testDays))

    events = lambda: store.get_events(dset, days=trainDays)
    times = core.get_times(events, step = 15.0, max_steps = 20)

    allSensors = Counter()
    reasoner = spatial.Reasoner( store.get_sensors(dset) )

    # -----------
    obs = list()
    for sensors,ts in tqdm(core.gen_active_sensors(events, times, maxActive = 300), total=len(times)):
        day = int(ts // 86400)

        patches = { p.kindKey : p.value 
                    for p in reasoner.getScores(sensors = sensors)
                    if p.value >= patchMin and p.kind }

        allSensors.update( patches )
        obs.append( patches )
    # -----------
    
    envLabels = store.get_labels(dset, days=trainDays)
    resLabels = envLabels.make_dataset(times, prefer_mat = True)

    envCond = defaultdict(Counter)

    for res in resLabels:
        for kObs, lbl in zip(obs, res):
            envCond[ sem.activity_type_map[lbl] ].update( kObs )

    return dset, { k : toProb(v, s=len(times)) for k,v in envCond.items() }

def toProb(d, s = None):
    if s is None:
        s = sum(d.values())
    return { k : n/s for k,n in d.items() }
#
# ========================================================================
#

def modelLabelSeq(args):
    store, dset, n_components, trainDays, testDays = args
    
    trainSeq = getDayLabelSeq(store, dset, list(range(trainDays)))
    testSeq = getDayLabelSeq(store, dset, list(range(trainDays, trainDays + testDays)))

    results = []

    for resName in trainSeq.keys():
        if resName in testSeq:
            rTrainDays, resTrainSeq = trainSeq[resName]
            rTestDays, resTestSeq = testSeq[resName]

            startDay = min(min(rTrainDays), min(rTestDays))

            resTrainSeq, resTestSeq = filterAndCondense(resTrainSeq, resTestSeq)

            model = hmm.MultinomialHMM(n_components = n_components)

            lengths = list(map(len, resTrainSeq))

            trainMat = np.vstack(resTrainSeq)

            model.fit(trainMat, lengths)

            for rel, lDays,lSeq in [("train", rTrainDays, resTrainSeq), ("test", rTestDays, resTestSeq)]:
                for d,x in zip(lDays, lSeq):
                    if seqEntropy(x) > 0:
                        results.append([ rel, dset, resName, n_components, d - startDay, seqEntropy(x), model.score(x) / len(x) ])

    print("Done with %s" % dset)
    return dset, results

parser = argparse.ArgumentParser(description="Evaluate the HMM")
parser.add_argument("store", help="Data store location")
parser.add_argument("--train",'-t', type=int, default=14, help="Number of training days")
parser.add_argument("--test",'-e', type=int, default=10**3, help="Number of test days")
parser.add_argument("--n_components", "-n", default=64, type=int, help="Components for HMM")

 
args = parser.parse_args()
store = core.Store(args.store)

with Pool() as p:
    
    jobs = [ (store, ds, args.n_components, args.train, args.test, 0.1) for ds in sorted(store.datasets()) ]

    joints = {}

    for ds,r in p.imap_unordered(modelSemanticSeq, jobs):
        joints[ds] = r
    
    # Reasoner.typeAdjust
    with open("semAssoc.json", "wt") as f:
        json.dump(joints, f, indent=2)
    exit()
    with open("label_probs.csv", "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["rel", "dataset", "resident", "n_components", "day", "entropy", "logprob"])
        for rows in p.imap_unordered(modelLabelSeq, jobs):
            writer.writerows(rows)

