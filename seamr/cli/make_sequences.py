import re
import os
import sys
import gzip
import json
import csv
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter, defaultdict
from itertools import groupby

from seamr.sem import COSE
from seamr import spatial
from seamr import core

#from hmmlearn.hmm import MultinomialHMM
#from pomegranate import HiddenMarkovModel
#from sklearn.ensemble.forest import RandomForestClassifier

"""
self.reasoner = res
self.name = name
self.type = tp
self.space = space
self.left=left
self.top=top
self.right=right
self.bottom = bottom
"""

class OneHotIndex:
    def __init__(self, labels):
        self.classes_ = np.array( sorted(set(labels)) )
        self.index = { c:i for i,c in enumerate(self.classes_) }
        self.integers = np.array([ self.index[v] for v in labels ])
        self.labels = np.array(labels)
    
    def segmments(self):
        return [ len(list(gp)) for k,gp in groupby(self.integers) ]
    
    def sections(self):
        d = defaultdict(list)
        for l,lItems in groupby(enumerate(self.integers), key=lambda T: T[1]):
            d[l].append([ i for i,_ in lItems ])
        return d

    def seq(self, integers=False):
        if integers:
            return [ v for v,_ in groupby(self.integers) ]
        else:
            return [ v for v,_ in groupby(self.labels) ]

def envLearn( args ):
    store, dset, days, outdir = args
    sensors = store.get_sensors(dset)
    res = spatial.Reasoner(sensors)
    
    spaces = set()
    objects = set()

    for b in res.getObjBoxes():
        if b.type:
            if b.type == b.space:
                spaces.add( b.type )
            else:
                objects.add( b.type )
    
    oIndex = {None:0}
    for o in sorted(spaces) + sorted(objects):
        oIndex[o] = len(oIndex)

    stream = lambda: store.get_events(dset, days=days)
    times = core.get_times(stream, step = 15.0, max_steps = 20)
    
    spaceSeq = []
    objSeq = []

    for row,(patches,tm) in enumerate(res.genPatches(stream, times, adjust=True)):
        
        domSpace = (None,None,0)
        domObj = (None,None,0)

        for T in patches:
            if T[1] in objects:
                if T[2] > domObj[2]:
                    domObj = T
            else:
                if T[2] > domSpace[2]:
                    domSpace = T
        
        spaceSeq.append(domSpace)
        objSeq.append(domObj)
    
    with open("%s/%s.pkl" % (outdir, dset,), "wb") as f:
        for r in range(len(times)):
            pickle.dump((times[r], objSeq[r], spaceSeq[r],), f)

    return dset
        
parser = argparse.ArgumentParser(description="Evaluate the HMM")
parser.add_argument("store", help="Store Location")
parser.add_argument("--states", "-s", default=16, type=int, help="Number of HMM States")
parser.add_argument("--offset", "-o", default=0, type=int, help="Days offset")
parser.add_argument("--days", "-d", default=14, type=int, help="Days")

args = parser.parse_args()

store = core.Store(args.store)

days = list(range(args.offset, args.offset+args.days))
sdir = "%02d-%02d" % (min(days), max(days),)

if not os.path.exists(sdir):
    os.makedirs(sdir)

jobs = [ (store, ds, days, sdir) for ds in store.datasets() ]

with Pool() as p:
    for dset in p.imap_unordered(envLearn, jobs):
        print(dset)
    