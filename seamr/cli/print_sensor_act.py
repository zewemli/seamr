import argparse

import numpy as np

from collections import Counter, defaultdict

import seamr
from seamr import spatial
from seamr import core

from seamr import FlexLogger as Lg

def entropy(vals, norm):
    s = sum(vals)
    if s == 0:
        return 0
    else:
        e = 0.0
        for v in vals:
            p = v/s
            e -= p * np.log2(p)
            
        return e / np.log2(norm)

def toProb(d):
    s = sum(d.values())
    return { k : v/s for k,v in d.items() }

def findCut( valDist, minMass = 0.05 ):
    allK = Counter()
    for mk in kindVals.values():
        allK.update(mk)
    
    bestUtil = 0.0
    bestCut = 0.0

    norm = len(allK)
    maxMass = sum(allK.values())
    pMass = 1.0

    for kLevel, kDist in sorted(kindVals.items()):
        allK.subtract( kDist )
        nz = [ v for v in allK.values() if v > 0 ]
        kMass = sum(allK.values()) / maxMass
        if kMass < minMass:
            break
        else:
            kEnt = entropy(nz, len(co))
            kUtility = ((pMass - kMass) / (pMass + kMass)) / kEnt
            if kUtility > bestUtil:
                bestUtil = kUtility
                bestCut = kLevel
            pMass = kMass
    
    return bestCut

parser = argparse.ArgumentParser(description="Evaluate the HMM")
parser.add_argument("store", help="Data store location")
parser.add_argument("dataset", help="Dataset to print")
parser.add_argument("--sem", action="store_true", default=False, help="Dataset to print")
parser.add_argument("--days",'-d', type=int, default=14, help="Number of training days")

# --------------------------------------------------------------------------

args = parser.parse_args()

days = list(range(args.days))

store = core.Store(args.store, use_sem_labels=True)

sensors = store.get_sensors(args.dataset)

labels = store.get_labels(args.dataset, days=days).getResSubsets()

lbl_assoc = defaultdict(Counter)
sensor_assoc = defaultdict(Counter)

co = defaultdict(lambda: defaultdict(Counter))

sensorNames = { s.id : s.name for s in sensors }

sOK = { n for n,name in sensorNames.items() if not ( ":t" in name or ":ls" in name ) }

events = lambda:store.get_events_lps(args.dataset, days=days)

times = core.get_times( events, 15, 20 )

res = spatial.Reasoner( args.dataset, sensors )

key = lambda p: p.objKey

# --------------------------------------------------------------------------

if args.sem:
    for patches, aSensors, t in res.genPatches(events, times, withSensors=True):
        for p1 in patches:
            pco = co[ key(p1) ][ p1.value ]
            for p2 in patches:
                pco[ key(p2) ] += p2.value

    kindCut = {}
    for kindA, kindVals in co.items():
        kindCut[ kindA ] = findCut(kindVals, minMass=0.05)

# --------------------------------------------------------------------------

for patches, aSensors, t in res.genPatches(events, times, withSensors=True):
    t_labels = [ l.strLabel(t) for l in labels ]
    
    if args.sem:
        for p in patches:
            for l in t_labels:
                if "Light" not in key(p): 
                    sensor_assoc[key(p)][l] += p.value
                    lbl_assoc[l][key(p)] += p.value
    else:

        for s in aSensors:
            if s in sOK:
                sensor_assoc[s].update(t_labels)
        
        for l in t_labels:
            for s in aSensors:
                if s in sOK:
                    lbl_assoc[l][s] += 1 #aSensors[s]

# --------------------------------------------------------------------------

print("--------| Sensor Labels |-------------")

for s in sorted(sensor_assoc.keys(), key=lambda s: sensorNames.get(s,s)):
    print(Lg.RED + sensorNames.get(s,s) + Lg.ENDC)
    for l,v in sorted(toProb( sensor_assoc[s] ).items(), key=lambda T: T[1], reverse=True):
        print("\t%s : %0.4f" % (l,v))

print("--------| Label Sensors |-------------")
for l in sorted(lbl_assoc.keys()):
    print(Lg.RED + l + Lg.ENDC)
    for s,v in sorted(toProb( lbl_assoc[l] ).items(), key=lambda T: T[1], reverse=True):
        print("\t%s : %0.4f" % (sensorNames.get(s,s), v))

