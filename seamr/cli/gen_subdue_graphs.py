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

import argparse
import re
import csv
from math import ceil,sqrt

import numpy as np

#----------------------------------------------------------------------------------------
def gen_edges( inputs, tolerance = 30 ):
    '''
    inputs must be a list of (node, start, end) elements
    tolerance is the gap that we will be willing to span
    '''
    seen = set()
    active_until = {}
    latest = None
    for n,s,e in sorted(inputs, key=lambda T: T[1]):
        assert n not in seen
        seen.add(n)
        active_until[n] = e + tolerance
        rm = [ a for a,aNum in active_until.items() if aNum < s ]
        for node in rm:
            active_until.pop(node)
        
        for prior in active_until.keys():
            if n != prior:
                yield prior, n

        latest = n

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
                        "ph" : kParts[-1] if len(kParts[-1]) > 2 else "None",
                        "n_obs": len(self.n_obs[n]),
                        "start_label": labels[ self.start_row[n] ],
                        "end_label": labels[ self.end_row[n] ]
                    })

#----------------------------------------------------------------------------------------

def spiralPos(n):
    k = ceil((sqrt(n)-1)/2)
    t = 2*k+1
    m = t**2 
    t = t-1

    if n >= m-t:
        return k-(m-n), -k
    else:
        m = m-t

    if n >= m-t:
        return -k,-k+(m-n)
    else:
        m = m-t
        
    if n >= m-t:
        return -k + (m-n), k
    else:
        return k, k - (m-n-t)

def makeGraphs(args):
    (storePath, dset, days, output, qTime) = args
    store = core.Store(storePath)

    sNames = { s.id : s.name for s in store.get_sensors(dset) }

    cose = sem.COSE()
    stream = lambda: store.get_events_lps(dset, days=list(range(days)))
    times = core.get_times( stream, step=15, max_steps=20 )

    labelVecs = store.get_labels( dset, days=list(range(days)) ).make_dataset(times, prefer_mat=True)
 
    labels = [ ":".join(T) for T in zip(*labelVecs) ]

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

    dg = nx.DiGraph()
    time_label = {}

    isASpace = set()
    spaceObjs = defaultdict(set)

    for row,(patches, sensors, t) in enumerate(reasoner.genPatches(stream, times, withSensors=True)):
        time_label[t] = labels[row]

        newNodes = currentID.update(row, t, patches)
        
        spaces = { nodeID for (nodeID, p, isHead) in newNodes if isHead }
        
        isASpace.update(spaces)

        #-----------------| Now set all of the non-spaces nodes or space phenomenon nodes |---------------
        for (nodeID, p, isHead) in newNodes:
            if isHead is False and p.obj in reasoner.nodeSpace:
                spaceID = currentID.getRef(reasoner.nodeSpace[ p.obj ], row, t, True)

                isASpace.add( spaceID )
                spaceObjs[ spaceID ].add( nodeID )
        #--------------------| First set the spaces |---------------------

    #----------------------------------------------
    for n,d in currentID.getNodeProps(times, labels):
        dg.add_node( n, **d )
    
    dg.add_edges_from( gen_edges(currentID.getSegments(isASpace), tolerance = qTime), kind="move" )

    for spaceID, objs in spaceObjs.items():
        for o in objs:
            dg.add_edge( spaceID, o, kind="in" )
            dg.add_edge( o, spaceID, kind="in" )
    
        dg.add_edges_from( gen_edges( currentID.getSegments(objs), tolerance = qTime), kind="obj" )

    nx.write_graphml(dg, os.path.join(output, "%s.graphml" % dset))

    for n,d in dg.nodes(data=True):
        if "kind" not in d:
            print("------------")
            print(n,d)
            print( currentID.id_key[n] )
            raise ValueError("Ws")

    with open(os.path.join(output, "%s.g" % dset), "wt") as fOut:
        fOut.write("".join([ "v %s %s\n" % (n, nData['kind']) for (n,nData) in sorted(dg.nodes(data=True)) ]))
        fOut.write("".join([ "d %s %s %s\n" % (u, v, d['kind']) for (u,v,d) in dg.edges_iter(data=True) ]))

    nComps = 0
    purity = 0.0
    largest = 0
    for comp in nx.connected_components( nx.Graph(dg) ):
        nc = len(comp)
        largest = max(largest, len(comp))
        cTimes = { dg.node[x]['start'] for x in comp } | { dg.node[x]['end'] for x in comp } 
        cc = Counter( time_label[x] for x in cTimes )
        nComps += 1
        purity += cc.most_common(1)[0][1] / len(cTimes)

    return "\n%s | nodes %s | edges %s |:%s:| %s | %0.3f" % (dset, dg.order(), dg.size(), largest, nComps, purity / nComps)
        
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
    for res in pool.imap_unordered(makeGraphs, mapargs):
        print(res)
