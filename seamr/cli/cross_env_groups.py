import pickle
import os
import sys
from seamr.evaluate import Perf
from tabulate import tabulate
import csv
import argparse
from tqdm import tqdm
import re
from multiprocessing import Pool
from collections import Counter, namedtuple
from itertools import groupby, combinations

import numpy as np

import seamr
from seamr import core
from seamr import spatial
from seamr.learn import smoothGroups
from sklearn.metrics import v_measure_score, adjusted_mutual_info_score
from sklearn import cluster
from scipy.stats import entropy

from tqdm import tqdm
import csv

DatasetRun = namedtuple("DatasetRun", ["dataset", "feats", "model", "remap", "local", "labels", "perfs"])

def setupDataset( args ):
    (store,
     smoothBy,
     nDays,
     nClusters,
     nExperts,
     concepts,
     cIndex,
     dset) = args

    days = list(range(nDays))
    sensors = store.get_sensors( dset )

    res = spatial.Reasoner( sensors )

    events = lambda: store.get_events( dset, days=days )
    times = core.get_times(events, step = 15, max_steps = 20)

    inSpace = { b.name : b.space for b in res.getObjBoxes() }

    feats = np.zeros( (len(times), len(concepts)) )

    for row, (patches, t) in enumerate(res.genPatches(events, times)):
        for (n,p,c,v) in patches:
            if n and inSpace[n]:
                space = cIndex[ inSpace[n] ]
                feats[row, space] = max(v, feats[row, space])
            
            if c:
                obj = cIndex[ c ]
                feats[row, obj] = max(v, feats[row, obj])
    
    km = cluster.MiniBatchKMeans( n_clusters = nClusters )
    smoothModel = cluster.AgglomerativeClustering(n_clusters = nExperts)
    local = km.fit_predict(feats)
    remap = smoothGroups( local, mapSize = nClusters, smoothBy = smoothModel )

    # Now get labels
    labels = store.get_labels(dset, days=days).make_dataset(times, prefer_mat = True)

    perfs = [ (v_measure_score(resident, local),
               adjusted_mutual_info_score(resident, local))
              for resident in labels ]
    
    return DatasetRun(dset, feats, km, remap, local, labels, perfs)

def crossCheck(args):
    runA, runB = args
    
    superPerf = []
    runs = []

    for a,b in [ (runA, runB), (runB, runA) ]:

        a_pred_b = a.remap[ a.model.predict( b.feats ) ]

        runs.append([ a.dataset,
                      b.dataset,
                      v_measure_score(b.local, a_pred_b),
                      adjusted_mutual_info_score(b.local, a_pred_b) ])
        
        for r,(resLabels, (vm, ami)) in enumerate(zip( b.labels, b.perfs )):
            a_vm = v_measure_score(resLabels, a_pred_b)
            a_ami = adjusted_mutual_info_score(resLabels, a_pred_b)

            idx = {c:i for i,c in enumerate(sorted(set(resLabels)))}

            superPerf.append([ a.dataset, 
                               b.dataset,
                               r,
                               entropy([ idx[c] for c in resLabels]),
                               a_vm,
                               a_ami,
                               vm,
                               ami ])

    return runs, superPerf

def getVocab(args):
    store, dset = args
    sensors = store.get_sensors(dset)
    boxes = spatial.Reasoner(sensors).getObjBoxes()
    concepts = set()
    
    for b in boxes:
        if b.type:
            concepts.add( b.type )
        if b.space:
            concepts.add( b.space )

    return concepts

parser = argparse.ArgumentParser(description="Convert performance objects to a nice simple csv")
parser.add_argument("store", help="Data store")
parser.add_argument("output", help="Output file")
parser.add_argument("--smoothing", default="markov", choices=["markov","affinity"], help="Smoothing clusters")
parser.add_argument("--days", default=15, type=int, help="Number of days for clustering")
parser.add_argument("--clusters", default=200, type=int, help="Number of clusters")
parser.add_argument("--experts", default=15, type=int, help="Number of clusters")

args = parser.parse_args()

store = core.Store(args.store)

pool = Pool()

with seamr.BlockWrapper("Getting concepts"):
    # ---------------------------------------
    allConcepts = set()
    for c in pool.map(getVocab, [ (store, ds) for ds in store.datasets() ]):
        allConcepts |= c
    # ---------------------------------------

conceptIndex = { c:i for i,c in enumerate(sorted(allConcepts)) }


mapArgs = [ (store,
             args.smoothing,
             args.days,
             args.clusters,
             args.experts,
             allConcepts,
             conceptIndex,
             ds) for ds in store.datasets() ]

eachDataset = list(pool.imap_unordered(setupDataset, mapArgs))

n = len( list(combinations(range(len(eachDataset)), 2)) )


with open( args.output + "_cluster_perf.csv", "wt" ) as cp:
    with open( args.output + "_supervised_perf.csv", "wt" ) as sp:
        cWriter = csv.writer(cp)
        sWriter = csv.writer(sp)

        cWriter.writerow(["src","target","vmeasure","adj_mi"])
        sWriter.writerow(["src","target","res","entropy","proj_vmeasure","proj_adj_mi", "lcl_vmeasure", "lcl_adj_mi"])
        
        for (clusterPerf, superPerf) in tqdm(pool.imap_unordered(crossCheck, combinations(eachDataset, 2) ), desc="Getting results", total=n):
            for row in clusterPerf:
                cWriter.writerow(row)
                
            for row in superPerf:
                sWriter.writerow(row)

