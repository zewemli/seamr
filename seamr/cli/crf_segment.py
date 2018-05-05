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

import seamr
from seamr import core
from seamr.core import construct
import argparse
import numpy as np
import shutil
import math

import inspect

from seamr import load_config
from seamr import core
from seamr import sem
from seamr.evaluate import Perf, ROCAnalysis
from seamr.core import build

from sklearn import cluster
from pycrfsuite import Trainer, Tagger
from scipy import stats

from seamr.segmenter import CRFSegmenter

import logging
import coloredlogs, logging
logger = logging.getLogger("seamr")
coloredlogs.install(level='INFO')

def gen_fitted_model(args):
    store, dset, conf = args

    k            = conf["k"]
    results_dir  = conf["results"]
    train_for    = conf["train_for"]
    test_for     = conf["test_for"]

    model_dir    = conf["model_dir"]

    max_iterations = conf['max_iterations']
    step = conf['step']
    window = conf['window']
    gap = conf['gap']

    days = list(range(train_for))

    features_def   = conf["features"]

    features = build(features_def, locals())

    # ------------------------
    events = lambda: store.get_events_lps(dset, days=days)
    times = core.get_times(events, step=conf["step_by"], max_steps=conf["max_steps"])
    labelIndex = store.get_labels(dset, days=days)

    segmenter = CRFSegmenter(store, dset,
                                model_dir,
                                gap=gap,
                                step=step,
                                window=window,
                                minSegLen = conf['minSegLen'],
                                maxSegLen = conf['maxSegLen'],
                                max_iterations = max_iterations)

    segmenter.fit( features, labelIndex, events, times )

    # ------------------------
    days = list(range(train_for, train_for + test_for))
    evalEvents = lambda: store.get_events_lps(dset, days=days)
    evalTimes  = core.get_times(evalEvents, step=conf["step_by"], max_steps=conf["max_steps"])

    labelIndex = store.get_labels(dset, days=days)
    evalTags = segmenter.tag(features, evalEvents, evalTimes)

    subRes = labelIndex.getResSubsets()
    for r in subRes:
        print(r.residents, len(r.labels))
    print("-----------")

    evalSegs = segmenter.labelsToSegs(evalTags)

    evalLens = [ e-s for s,e in evalSegs ]

    print("SEGS",
            sum(evalLens), len(evalTimes), "||",
            len(evalLens), ":",
            np.min(evalLens), np.mean(evalLens), np.median(evalLens), np.max(evalLens))
    exit()

    return None


# ---------------------------------------
# CLI Argument Parser
# ---------------------------------------
default_conf = """
k: 5
store: "/data/phd/datasets/store"
results: "./results"
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
parser.add_argument("--store", help="Path to data")
parser.add_argument("--results", help="Save to this path")

args = parser.parse_args()

for c_arg in args.config:
    assert os.path.exists(c_arg), "%s does not exist :(" % c_arg

if args.results:
    conf['results'] = args.results

conf = load_config(default_conf, *args.config)

store = core.Store( args.store or conf["store"] )

cose = sem.COSE()

# ---------
for _ in map(gen_fitted_model, [ (store, ds, conf) for ds in store.datasets() ]):
    # ---
    pass