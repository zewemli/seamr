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
from itertools import groupby

import seamr
from seamr.sem import COSE
from seamr import spatial
from seamr import core
from seamr import sem
from seamr import load_config

from sklearn import cluster

from pomegranate import HiddenMarkovModel
from pomegranate import LogNormalDistribution

def getVocab( args ):
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
    
    return spaces, objects

def singleEnvLearn( args ):
    
    (store,
    dset, 
    days, 
    index,
    conf,
    outdir,) = args
    
    threshold    = conf.get('threshold', 0.1)
    distribution = conf.get("distribution", LogNormalDistribution)
    n_components = conf.get('n_components', 32)
    seqLen       = conf.get('seqLen', 16)

    cose = sem.COSE()
    
    sensors = store.get_sensors(dset)
    res     = spatial.Reasoner(sensors)

    labelIndex = store.get_labels(dset, days=days)

    featNames = [ k for k,v in sorted(index.items(), key=lambda T: T[1]) ]

    upperKind = {}
    for k in featNames:
        try:
            for uk in ['cose:PartOfDay', 'cose:FunctionalSpace', 'cose:HouseholdItem']:
                if cose.is_a(k, uk):
                    upperKind[k] = uk
                    break
        except:
            upperKind[k] = k    

    if labelIndex.nRes() > 1:
        return dset
    else:
        stream = lambda: store.get_events(dset, days=days)
        times = core.get_times(stream, step = 15.0, max_steps = 20)

        resLabels = labelIndex.make_dataset(times, prefer_mat = True)
        labelRows = { s : [] for s in set(resLabels[0]) if s != 'other' }

        for g,items in groupby(enumerate(resLabels[0]), key=lambda T: T[1]):
            if g != "other":
                labelRows[ g ].extend([ i for i,_ in items ])
        
        sequence = np.zeros( (len(times), len(index)) )

        maxTime = 15 * 20

        with seamr.BlockWrapper("Getting data for %s" % dset):
            pTime = 0
            for row,(patches,tm) in enumerate(res.genPatches(stream, times, adjust=True)):
                sequence[row, index[cose.time_label(tm)]] = 1.0
                
                if tm - pTime > maxTime:
                    sequence[row, index['GAP']] = 1.0
                pTime = tm

                for (n, con, val) in patches:
                    if val >= threshold:
                        sequence[ row, index[con] ] = 1.0
        
        labelModels = {}

        for lbl,lRows in tqdm(labelRows.items(), desc="Fitting label models %s" % dset, total=len(labelRows)):
            tStart = time.monotonic()
            XSeq = sequence[lRows,:]
            print("Fitting %s : %s to |%d| samples" % (dset, lbl, len(lRows),))
            
            xstd = XSeq.std(axis=0)
            xmean = XSeq.mean(axis=0)

            mat = np.zeros( (xstd.size, 2) )
            mat[:,0] = xmean * xstd
            mat[:,1] = xstd
            
            clusters = cluster.KMeans(n_clusters=2).fit_predict(mat)

            
            with open( os.path.join(outdir, lbl, "%s.csv" % dset), "wt" ) as f:
                f.write("label,dset,feat,kind,clstr,xmean,xstd\n")
                for (fname, cl, xm, xs) in zip(featNames, clusters, xstd, xmean):
                    f.write("%s,%s,%s,%s,%s,%s,%s\n" % (lbl, dset, fname, upperKind.get(fname, fname), cl, xm, xs))
            
            print("Dist: %s :: %s" % ( XSeq.shape, " | ".join(["%s:%s" % T for T in zip(featNames, xstd)]) ))
            """
            model = HiddenMarkovModel.from_samples(distribution = distribution,
                                                    n_components = n_components,
                                                    batch_size = seqLen,
                                                    X = XSeq)
            """
            tEnd = time.monotonic()
            print("Fitting %s : %s to |%d| samples done in %0.3f seconds" % (dset, lbl, len(lRows), tEnd - tStart))
            
            #with open( os.path.join(outdir, lbl, "%s.json" % dset), "wt" ) as f:
            #    f.write( model.to_json() )

        return dset

parser = argparse.ArgumentParser(description="Evaluate the HMM")
parser.add_argument("conf", nargs="+", help="Store Location")
parser.add_argument("--store", "-s", help="Data store location")

args = parser.parse_args()

conf = load_config( *args.conf )

store = core.Store( args.store or conf.get("store",None), use_sem_labels=True )

offset = conf.get('offset',0)
days = list(range(offset, offset + conf.get('days', 15)))

sdir = "%02d-%02d" % (min(days), max(days),)

cose = sem.COSE()

for k,v in sem.activity_type_map.items():
    try:
        os.makedirs( os.path.join(sdir, v.split(":")[-1]) )
    except:
        pass

with Pool() as p:
    
    jobs = [ (store, ds, days, sdir) for ds in sorted(store.datasets()) ]

    if not os.path.exists( conf['vocab'] ):
        spaces = set()
        objects = set()
        for dSpaces, dObjs in p.imap_unordered(getVocab, jobs):
            spaces |= dSpaces
            objects |= dObjs
        
        index = {None: 0, "GAP": 1}
        for t in sorted(cose.time_order().items(), key=lambda T: T[1]):
            index[t[0]] = len(index)

        for s in sorted(spaces) + sorted(objects):
            index[s] = len(index)
        
        with open(conf['vocab'], "wb") as fo:
            pickle.dump( index, fo )
    else:
        with open(conf['vocab'], "rb") as fo:
            index = pickle.load( fo )

    jobs = [ (store, ds, days, index, conf, sdir) for ds in sorted(store.datasets()) ]
    for dset in p.imap_unordered(singleEnvLearn, jobs):
        print("Done with %s" % dset)