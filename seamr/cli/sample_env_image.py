assert __name__ == "__main__"
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

from multiprocessing import Pool

import seamr
from seamr import core
from seamr.core import construct
import argparse
import numpy as np
import shutil
import math

import inspect

import skvideo.io

from seamr import load_config
from seamr import core
from seamr import sem
from seamr import spatial
from seamr.evaluate import Perf

import imageio

import logging
import coloredlogs, logging
logger = logging.getLogger("seamr")
coloredlogs.install(level='INFO')

def create_all_animations(pArgs):

    (store_path,
    dset,
    k,
    eval_for,
    validate_for,
    step_by,
    max_steps,
    gridSize,
    numImages, numFrames, scale) = pArgs

    store = core.Store( store_path )

    env = dset.split(".")[0]

    days = set()
    times = set()

    for eval_mode in ['k-blocked','one-day']:

        res = None
        if ":" in dset:
            res,dset = dset.split(":")

        folds = core.get_schedule(store,
                                dset,
                                k,
                                eval_for,
                                validate_for,
                                eval_mode,
                                step = step_by,
                                max_steps = max_steps)

        for fold, (f_times, f_days) in enumerate(folds):
            days |= set(chain(*[ d for d in f_days if d is not None ]))
            times |= set(chain(*[ d for d in f_times if d is not None ]))
    
    sensors = store.get_sensors( dset )

    days = sorted(days)
    times = sorted(times)
    
    # ------------------------------------------------------
    print( "Dataset: %s : times %s" % (dset, len(times)) )
    # ------------------------------------------------------
    
    res = spatial.Reasoner(env, sensors, gridSize = gridSize)

    batch = []
    
    images = []

    pe = None
    
    resLabels = store.get_labels(dset, days=sorted(days)).getResSubsets()

    openVideos = {}
    lineName = {}

    img = None
    prevSensors = set()

    for tSensors, ts in core.gen_active_sensors(lambda: store.get_events(dset, days=sorted(days)), times):

        labels = [ r.getAt(ts) for r in resLabels ]

        if any([ l.activity != "other" for l in labels ]):
            if len(tSensors) and random() < 0.01:
                img = res.getImage(tSensors, scale=scale)
                dt = datetime.fromtimestamp(ts)
                imageio.imwrite("./sample_images/%s_%s.png" % (dset, dt.strftime("%Y-%m-%d_%H-%M-%S")), img)
                numFrames -= 1
                if numFrames <= 0:
                    break

    return dset

# ---------------------------------------
# CLI Argument Parser
# ---------------------------------------
default_conf = """
k: 5
store: "/data/phd/datasets/store"
eval_for: 30
eval_mode: "k-blocked"
validate_for: 30
max_steps: 15
step_by: 30
invert_eval: false
eval_batch: 1000
datasets: [] # Means all datasets
classifier: 
    sklearn.ensemble.RandomForestClassifier:
        criterion: gini
        n_jobs: -1
        n_estimators: 25
features:
    seamr.features.SEAMRFeatures:
        nbins: 10
        windowSize: 180
        sensors: null

"""

parser = argparse.ArgumentParser(description="Tool to run SEAMR experiments")

parser.add_argument("config", nargs="+", help="Config file(s) for experiment. The default is: \n%s" % default_conf)
parser.add_argument("--num", default=5, type=int, help="Number of animations to produce")
parser.add_argument("--frames", default=60, type=int, help="Number of animations to produce")
parser.add_argument("--scale", default=3, type=int, help="Scale the images by")
parser.add_argument("--store", help="Path to data")

args = parser.parse_args()

for c_arg in args.config:
    assert os.path.exists(c_arg), "%s does not exist :(" % c_arg

conf = load_config(default_conf, *args.config)

conf_name = ".".join([ basename(x).split(".")[0] for x in sorted(args.config) ])

store = args.store or conf["store"]

k            = conf["k"]
eval_for     = conf["eval_for"]
validate_for = conf["validate_for"]

max_steps = conf["max_steps"]
step_by   = conf["step_by"]

gridSize = conf["gridSize"]

datasets = sorted(core.Store(store).datasets())

print("Running over datasets: %s" % (datasets,))

try:
    os.makedirs("./sample_images")
except:
    pass

pargs = [(store,
          dset,
          k,
          eval_for,
          validate_for,
          step_by,
          max_steps,
          gridSize,
          args.num,
          args.frames,
          args.scale)
            for dset in datasets ]

# ---------

done_datasets = set(datasets)

pool = Pool()
for dset in tqdm(pool.imap_unordered(create_all_animations, pargs), desc="storing data"):
    done_datasets.add(dset)
    print("Done with %s (%s of %s)" % ( dset, len(done_datasets), len(datasets) ))
