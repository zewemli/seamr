assert __name__ == "__main__"

import numpy as np
np.seterr(all='raise')

import os, sys
import time
from datetime import datetime
from itertools import chain
from collections import Counter
import yaml, json
import gzip
from tqdm import tqdm
import pstats, cProfile
import re

from os.path import join,basename,dirname, exists

from random import random

import seamr
from seamr import core
from seamr.core import construct
import argparse
import numpy as np
import shutil
import math

import inspect

from seamr import core
from seamr import sem
from seamr import spatial
from seamr.evaluate import Perf, ROCAnalysis
from seamr.features.semantic import CrossEnvFeatures,SingleEnvFeatures

from multiprocessing import Pool

import logging
import coloredlogs, logging
logger = logging.getLogger("seamr")
coloredlogs.install(level='INFO')

def storeEvents(args):
    store_in, store_out, dset, localOnly = args
    days = list(range(1000))

    try:
        os.
        store_out.path(dset)

    with seamr.BlockWrapper("Storing events for %s" % dset):
        events = lambda: store_in.get_events_lps(dset, days=days)

        res = spatial.Reasoner( store_in.get_sensors(dset) )

        index = {}
        eventStream = res.genEvents(events, index, useObjKey = localOnly)

        # Actually store all events
        nEvents = store_out.store_events(eventStream, dset)

        # Now save out the mask images
        res.saveImages( store_out.path(dset) )

        # Save the definitions for sensors
        store_out.store_sensors([ (i,c) for (c,i) in index.items() ], dset, prefix=False)

        return dset, nEvents

def storeLabels(args):
    store_in, store_out, dset, semLabel = args
    days = list(range(1000))

    semID = { c:i for i,c in enumerate(semLabel) }

    allLabels = store_in.get_labels(dset, days=days)

    def lblGen():
        for l in allLabels.labels:
            semLbl = sem.activity_type_map[ l.activity ].split(":")[-1].lower()
            resName = ("%s:%s" % (dset, l.resident)).lower()
            yield (l.line, semID[semLbl], l.resID, resName, semLbl, l.start, l.finish,)

    lblCount = store_out.store_labels(lblGen(), dset)

    return dset, lblCount

parser = argparse.ArgumentParser(description="Tool to run SEAMR experiments")

parser.add_argument("store_in", help="Input store path")
parser.add_argument("store_out", help="Output store path")
parser.add_argument("--local", action="store_true", default=False, help="Only local objects")
args = parser.parse_args()

store = core.Store( args.store_in, use_sem_labels=False )
store_out = core.Store( args.store_out )

datasets = store.datasets()

semLabel = sorted({ c.split(":")[-1].lower() for c in sem.activity_type_map.values() })
for dset in datasets:
    store_out.store_activities( enumerate(semLabel), dset )

remove_names = set(["events", "labels", "activity", "sensor"])
for (dirpath, dirnames, filenames) in os.walk( args.store_out ):
    for f in set(filenames) & remove_names:
        os.remove( os.path.join(dirpath, f) )

with Pool() as pool:
    
    with seamr.BlockWrapper("Storing events"):
        poolArgs = [ (store, store_out, ds, args.local) for ds in datasets ]
        for ds,dlines in pool.imap_unordered(storeEvents, poolArgs):
            print("Events : %s : %d" % (ds, dlines))

    with seamr.BlockWrapper("Storing labels"):
        poolArgs = [ (store, store_out, ds, semLabel) for ds in datasets ]
        for ds,dlines in pool.imap_unordered(storeLabels, poolArgs):
            print("Labels : %s : %d" % (ds, dlines))
