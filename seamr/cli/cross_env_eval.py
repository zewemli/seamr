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
from seamr.evaluate import Perf, ROCAnalysis, SequenceProb
from seamr.reasoner import ActivityReasoner

from seamr.core import build

from tqdm import tqdm

import logging
import coloredlogs, logging
logger = logging.getLogger("seamr")
coloredlogs.install(level='INFO')


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

conf = load_config(default_conf, *args.config)

conf_name    = ".".join([ basename(x).split(".")[0] for x in sorted(args.config) ])

store = core.Store( args.store or conf["store"], use_sem_labels = True )

results_dir  = args.results or conf["results"]
train_for = conf["train_for"]
eval_for = conf['eval_for']
rand_step = conf["rand_step"]

hmm_components = conf["hmm_components"]

weight_hours = conf["weight_hours"]
weight_step = conf["weight_step"]
num_level_checks = conf["num_level_checks"]

max_steps    = conf["max_steps"]
step_by      = conf["step_by"]

datasets       = (conf["datasets"] or sorted(store.datasets()))
classifier_def = conf["classifier"]
feat_args   = conf["feat_args"]

xenv_fit = conf["xenv_fit"]
xenv_dist = conf["xenv_dist"]

seamr.log.info("Running over datasets: %s" % (datasets,))

reas = ActivityReasoner(store,
                        classifier_def,
                        feat_args,
                        hmm_components=hmm_components,
                        randStepArgs = rand_step,
                        weight_hours = weight_hours,
                        weight_step = weight_step,
                        step_by = step_by,
                        max_steps = max_steps,
                        num_level_checks = num_level_checks,
                        xenv_dist = xenv_dist)

reas.fit(datasets, train_for, xenv_fit)

for env in tqdm(reas.envs, desc="Predicting", total=len(reas.envs)):
    eval_days = list(range(xenv_fit, xenv_fit + eval_for + 1))
    events, times = env.getEventsAndTimes( eval_days )

    realLabels = store.get_labels(env.dataset, days=eval_days)
    reals = np.zeros( (len(times), env.classes_.size) )

    for res in realLabels.make_dataset(times, prefer_mat=True):
        for row, l in enumerate(res):
            if l in env.classIndex:
                reals[row, env.classIndex[l]] = 1.0

    probs = reas.model_proba(env, xenv_fit, xenv_fit + eval_for)

    assert probs.shape[0] == reals.shape[0]
    
    ROCAnalysis(env.classes_)\
        .setup( env.dataset, reals, probs )\
        .show()\
        .save( results_dir )
    
    if realLabels.nRes() == 1:
        preds = reas.predict(env, xenv_fit, xenv_fit + eval_for)
        Perf()\
            .setup(env.dataset,
                   env.classes_[ reals.argmax(axis=1) ],
                   preds,
                   times)\
            .show()\
            .save( results_dir )
    print("=" * 50)