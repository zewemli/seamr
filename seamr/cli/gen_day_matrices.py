import os, sys

from itertools import groupby, combinations, product
from collections import defaultdict, Counter, deque
from multiprocessing import Pool

from datetime import datetime

import seamr
from seamr import core, sem
from seamr import spatial
from seamr.features.semantic import PatchFeatures
import random

import networkx as nx
import time

import pickle
import argparse
import re
import csv
from math import ceil,sqrt

import numpy as np

#----------------------------------------------------------------------------------------
def point_time_label(sec_of_day, start, end, offset):
    h = 3600.0
    if (start * h) <= sec_of_day <= (end * h):
        return 1.0
    else:
        dist = min(abs((start * h + offset) - sec_of_day), abs((end * h + offset) - sec_of_day))
        try:
            return 0.975 ** (dist / 60)
        except:
            return 0.0

def time_label(sec_of_day, start, end):
    return max( point_time_label(sec_of_day, start, end, 0),
                point_time_label(sec_of_day, start, end, 86400) )

def get_times(ts):
    
    times = [
        ('cose:Night', 0, 5),
        ('cose:Morning', 5, 9),
        ('cose:MidMorning', 9, 12),
        ('cose:Midday', 12, 14),
        ('cose:Afternoon', 14, 17),
        ('cose:Twilight', 17, 21),
        ('cose:Evening', 21, 24)
    ]
    for fname, start, end in times:
        f_weight = time_label( ts % 86400, start, end )
        if f_weight > (10.0 ** -3):
            yield fname, f_weight

#----------------------------------------------------------------------------------------

class ContiguousIDs:
    def __init__(self, spacing=240.0):
        self.i = 0
        self.spacing = spacing
        self.cose = sem.COSE()
        self.node = {}
        self.first_obs = {}
        self.last_obs = {}
        self.start_row = {}
        self.end_row = {}
        self.id_key = {}
        self.n_obs = {}
        self.row_time = {}
        self.cose = sem.COSE()
        self.phInteract = { "cose:Movement","cose:Contact" }

    def key(self, p):
        if p.kind and self.cose.is_a(p.kind, "cose:FunctionalSpace"):
            return p.obj
        else:
            return p.objKey

    def isActive(self, pKey):
        prev = self.nums.get(pKey, -1)
        return (prev < 0) or ((pRow - prev) > self.spacing)

    def getRef(self, pKey, pRow, t, isSpace):
        self.row_time[pRow] = t

        pKey = pKey.replace("cose:","")
        if isSpace:
            pKey = pKey[:pKey.rindex("-")]

        if pKey in self.node:
            n = self.node[pKey]
            if abs(t - self.last_obs[n]) <= self.spacing:
                self.last_obs[ n ] = t
                self.end_row[ n ] = pRow
                self.n_obs[ n ].add(pRow)
                return n
        
        # Otherwise
        self.i += 1
        self.node[pKey] = self.i
        self.id_key[ self.i ] = pKey
        self.first_obs[ self.i ] = t
        self.last_obs[ self.i ] = t
        self.start_row[ self.i ] = pRow
        self.end_row[ self.i ] = pRow
        self.n_obs[self.i] = set([pRow])
        
        return self.i

    def update(self, pRow, t, patches):

        newNodes = []

        current_pos = self.i

        for p in patches:
            pKey = self.key(p)
            isSpace = self.cose.is_a(p.kind, "cose:FunctionalSpace")
            pNode = self.getRef(pKey, pRow, t, isSpace)
            if pNode > current_pos:
                isInteract = p.phenomenon in self.phInteract
                newNodes.append([pNode, p, isSpace and isInteract])

        return newNodes
    
    def getSegments(self, nodes):
        return [ (n, self.first_obs[n], self.last_obs[n]) for n in nodes ]
    
    def getKinds(self):
        kinds = set()
        for n, key in self.id_key.items():
            kParts = key.split("-")
            
            if kParts[0] == "space":
                kind = kParts[1]
            else:
                kind = kParts[0]
            ph = kParts[-1] if len(kParts[-1]) > 2 else "None"
            kinds.add("%s-%s" % (kind, ph))
        return kinds

    def getNodeProps(self, times, labels):
        for n, key in self.id_key.items():
            kParts = key.split("-")
            
            if kParts[0] == "space":
                kind = kParts[1]
            else:
                kind = kParts[0]
            
            yield (n, { "kind" : kind,
                        "start": self.first_obs[n],
                        "end": self.last_obs[n],
                        "start_row": self.start_row[n],
                        "end_row": self.end_row[n],
                        "ph" : kParts[-1] if len(kParts[-1]) > 2 else "None",
                        "n_obs": len(self.n_obs[n]),
                        "start_label": labels[ self.start_row[n] ],
                        "end_label": labels[ self.end_row[n] ]
                    })

#----------------------------------------------------------------------------------------

def storeMatrics(args):
    (storePath, dset, days, output, qTime) = args
    store = core.Store(storePath)

    sNames = { s.id : s.name for s in store.get_sensors(dset) }

    cose = sem.COSE()
    stream = lambda: store.get_events_lps(dset, days=list(range(days)))
    times = core.get_times( stream, step=15, max_steps=20 )

    labelVecs = store.get_labels( dset, days=list(range(days)) ).make_dataset(times, prefer_mat=True)

    env = dset.split(".")[0]
    patchFeats = PatchFeatures(store, dset, decay=0)

    spaceKind = { T[0] : T[1] for T in patchFeats.res.objBoxes }
    reasoner = patchFeats.res

    '''
    self.nodeSpace[ n['id'] ] = iSpaceID
    self.spaceNodes[ iSpaceID ].add( n['id'] )
    '''

    # --------------
    started = set()
    
    currentID = ContiguousIDs( spacing = qTime )
    
    seg = 0

    prev = set()

    start_label = {}
    end_label = {}
    
    patch_kind = {}

    time_label = {}
    spaceObjs = defaultdict(set)

    fields = set()
    for p in reasoner.getDefaultPatches():
        fields.add( p.kindKey )
        if p.obj in reasoner.nodeSpace:
            spaceType = reasoner.nodeType[ reasoner.nodeSpace[p.obj] ]
            spaceKey = "%s-%s" % (spaceType, p.phenomenon)
            fields.add(spaceKey)
            
    fields = sorted(fields)
    findex = { c:i for i,c in enumerate(fields) }

    def gen():
        for row,(patches, sensors, t) in enumerate(reasoner.genPatches(stream, times, withSensors=True)):
        
            ex = {}
            for p in patches:
                ex[p.kindKey] = max(ex.get(p.kindKey, 0.0), p.value)
                if p.obj in reasoner.nodeSpace:
                    spaceType = reasoner.nodeType[ reasoner.nodeSpace[p.obj] ]
                    spaceKey = "%s-%s" % (spaceType, p.phenomenon)
                    ex[spaceKey] = max(ex.get(spaceKey, 0.0), p.value)
            
            yield int( t // 86400 ), (t, [ lv[row] for lv in labelVecs ], { findex[c] : v for c,v in ex.items() })
        
    store.store_matrices(gen(), dset)
    with open( store.path(dset, "matrix_feats.pkl"), "wb" ) as f:
        pickle.dump(fields, f)

parser = argparse.ArgumentParser(description="Evaluate results from multiple")

parser.add_argument("store", help="Input file")
parser.add_argument("--days", type=int, default=30, help="Input file")
parser.add_argument("--qTime", type=int, default=30, help="Input file")
parser.add_argument("--output", default="graphs", help="Output directory")

args = parser.parse_args()
store = core.Store(args.store)

if not os.path.exists( args.output ):
    os.makedirs(args.output)

mapargs = [ ( args.store, ds, args.days, args.output, args.qTime) for ds in store.datasets() ]

with Pool() as pool:
    for res in pool.imap_unordered(storeMatrics, mapargs):
        print(res)
