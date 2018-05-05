assert __name__ == "__main__"

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os, sys
import time
from datetime import datetime
from itertools import chain, combinations, groupby
from collections import Counter, namedtuple, deque
import yaml, json
import gzip
from tqdm import tqdm
import pstats, cProfile
import re
import csv

from os.path import join,basename,dirname, exists

from random import random

import seamr
from seamr import core
from seamr.core import construct
from sklearn import metrics

import argparse
import numpy as np
import shutil
import math
import inspect

from prg import prg

from seamr import load_config
from seamr import core
from seamr import sem
from seamr.features.semantic import CrossEnvFeatures
from seamr.core import build

from seamr.evaluate import Perf, ROCAnalysis

from tqdm import tqdm

from multiprocessing import Pool, cpu_count

import logging
import coloredlogs, logging
logger = logging.getLogger("seamr")
coloredlogs.install(level='INFO')

np.seterr(all="ignore")

Dataset = namedtuple("Dataset", ["dset","feats","classes","X","y", "times", "exp"])

def get_optimalF1(real, pred):
    
    predOrd = np.argsort(pred)

    real = real[ predOrd ]
    pred = pred[ predOrd ]
    
    fn = 0.0
    tp = sum(real)
    fp = real.size - tp

    optimalF1 = 0

    for pval, section in groupby(zip(real,pred), key=lambda T: T[1]):
        section = [ r for (r,_) in section ]
        
        n = len(section)
        npos = sum(section)

        fn += npos 
        fp -= n - npos
        tp -= npos
        
        optimalF1 = max(optimalF1, ( (2 * tp) / (2*tp + fp + fn) ))

    return optimalF1

def get_exp(conf, store, feats, dset, step_by, max_steps, classes, y, events, times):
    if conf.get('expectations', None) is None:
        return None
    else:

        iExp = { "feats" : list(feats.genPatches(events, times)) }

        for i,cls in enumerate(classes):
            try:
                expClass = build(conf['expectations'], locals())
                if hasattr(expClass, "update"):
                    expClass.update(iExp['feats'], y[:,i])
                
                iExp[cls] = expClass

            except KeyError:
                pass
        
        return iExp


def step1_get_datasets(args):
    """
    Getting dataset features, meant to be run in a subprocess
    """
    conf, store, dset, features_def, n_days, step_by, max_steps = args

    days = list(range(n_days))

    with seamr.BlockWrapper("Getting features for %s" % dset):
        events = lambda: store.get_events(dset, days = days)
        times = core.get_times(events, step = step_by, max_steps = max_steps)

        labelIndex = store.get_labels(dset, days = days)
        reals = labelIndex.make_dataset(times, prefer_mat=True)

        allClasses = set()
        for res in reals:
            allClasses |= set(res)

        classes = sorted(allClasses)
        cIndex = { c:i for i,c in enumerate(classes) }

        y = np.zeros( (len(times), len(classes)) )

        for res in labelIndex.make_dataset(times, prefer_mat=True):
            for row, l in enumerate(res):
                y[row, cIndex[l]] = 1.0
        

        feats = build(features_def, locals())
        
        exp_val = get_exp(conf, store,
                                feats,
                                dset,
                                step_by,
                                max_steps,
                                classes,
                                y,
                                events,
                                times)

        if exp_val is not None:
            X = feats.make_dataset(events, times, patches = exp_val['feats'])
        else:
            X = feats.make_dataset(events, times)

        return Dataset( dset, feats, classes, X, y, times, exp_val )

#--------------------------------------------------------------------------------------------

def step2_fit_and_predict(args):
    classifier_def, results_dir, conf, pair = args
    setA, setB = pair
    cose = sem.COSE()
    epsilon = 10.0 ** -5
    
    actModels = cose.get_activity_models()

    featsName = str(setA.feats)

    commonFeats = sorted(set(setA.feats.feat_names) & set(setB.feats.feat_names))
    commonClasses = set(setA.classes) & set(setB.classes)
    
    # Never build an "other" classifier
    commonClasses.discard("other")

    if not conf.get("model_filter", False):    
        aX = setA.feats.intersect(commonFeats, setA.X)
        bX = setB.feats.intersect(commonFeats, setB.X)
    
    results = []
    rocs = []

    def twoD(m):
        if m.ndim == 1:
            return np.expand_dims(m, -1)
        else:
            return m

    for cls in commonClasses:
    
        if cls in actModels:
            clsConcepts = { c.split(":")[-1] for c in actModels[cls].concepts() }
        else:
            seamr.log.info("No model for %s" % cls)
            clsConcepts = { c.split("-")[0] for c in commonFeats }

        if conf.get("model_filter", False) and cls in actModels:
            useFeats = { c for c in commonFeats if c.split("-")[0] in clsConcepts }
            seamr.log.info("Model filter, using %s |vs| %s" % (useFeats, commonFeats))
            aX = setA.feats.intersect(useFeats, setA.X)
            bX = setB.feats.intersect(useFeats, setB.X)
        else:
            useFeats = commonFeats


        # Select only the class of interest
        aCol = setA.classes.index(cls)
        bCol = setB.classes.index(cls)
        aY = setA.y[ :, aCol ]
        bY = setB.y[ :, bCol ]

        if aY.sum() in [1,0] or bY.sum() in [1,0]:
            continue
        else:
            cA = Dataset(setA.dset, None, [cls], aX, aY, setA.times, setA.exp)
            cB = Dataset(setB.dset, None, [cls], bX, bY, setB.times, setB.exp)

            for train,test in [(cA,cB), (cB,cA)]:

                model = core.build(classifier_def)

                y_true = test.y.ravel()
                y_true[ ~np.isfinite(y_true) ] = 0.0

                y_pred = model.fit(train.X, train.y).predict_proba(test.X)[:,1].ravel()
                y_pred[ ~np.isfinite(y_pred) ] = 0.0
                #          |----------------------|
                # ---------| Expectation modeling |------------------
                testModel = test.exp.get(cls, None) if test.exp is not None else None
                trainModel = train.exp.get(cls, None) if train.exp is not None else None

                if testModel and testModel.isMerged():
                    exp_vec = np.array( list( testModel(test.exp['feats']) ) )
                    exp_val = (exp_vec + epsilon) * y_pred
                    y_pred = exp_val / (exp_val.max() + epsilon)
                    
                elif trainModel:
                    exp_vec = np.array( list( trainModel(test.exp['feats']) ) )
                    exp_val = (exp_vec + epsilon) * y_pred
                    y_pred = exp_val / (exp_val.max() + epsilon)
                # ---------| Done with expectations |-----------------
                #          |------------------------|

                # ------------------
                # Metrics
                # ------------------
                cParams = { "activity": cls,
                            "source": train.dset,
                            "target": test.dset,
                            "classifier": str(model),
                            "features": featsName }
                            
                clsRoc = ROCAnalysis( [cls] )\
                            .setup("{classifier}_{features}_{source}_{target}".format(**cParams), twoD(y_true), twoD(y_pred))\
                            .setParams(cParams)
                
                rocs.append( clsRoc )

                results.append({
                    "optimalF1": get_optimalF1(y_true, y_pred),
                    "auprg": prg.calc_auprg( prg.create_prg_curve(y_true, y_pred) ),
                    "source": train.dset,
                    "target": test.dset,
                    "activity": cls,
                    "roc_auc": metrics.roc_auc_score(y_true, y_pred),
                    "classifier": str(model),
                    "features": featsName
                })

    return results, rocs
    
def combined_yaml(configs):
    conf = {}
    for c in configs:
        if c.endswith(".yaml"):
            with open(c) as cf:
                conf.update( yaml.load(cf) )
    return conf

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
features:
    seamr.features.semantic.CrossEnvFeatures:
        withTimes: true
        avg: 0
"""

parser = argparse.ArgumentParser(description="Tool to run SEAMR experiments")

parser.add_argument("config", nargs="+", help="Config file(s) for experiment. The default is: \n%s" % default_conf)
parser.add_argument("--store", help="Path to data")
parser.add_argument("--results", help="Save to this path")
parser.add_argument("--procs", default=cpu_count(), type=int, help="Number of processes")

args = parser.parse_args()

for c_arg in args.config:
    assert os.path.exists(c_arg), "%s does not exist :(" % c_arg

conf = load_config(default_conf, *args.config)

store = core.Store( args.store or conf.get('store',None), use_sem_labels=True )

run_name = "-".join([ os.path.basename(y) for y in args.config ]).replace(".yaml","")
results_dir = os.path.join(args.results or conf.get("results","./results"), run_name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(os.path.join(results_dir, "config.yaml"), "wt") as fOut:
    yaml.dump(combined_yaml(args.config), default_flow_style=False)

features_def = conf["features"]
classifier_def = conf["classifier"]

n_eval_days = conf.get("eval_for", 30)
t_step_by = conf.get("step_by", 15)
t_max_steps = conf.get("max_steps", 20)


# Step 1: Get all the datasets
with seamr.BlockWrapper("Step 1: Getting matrices"):
    with Pool( args.procs ) as p:
        run_args = [ (conf, store, ds, features_def, n_eval_days, t_step_by, t_max_steps ) for ds in store.datasets() ]

        all_datasets = []
        for dataset in p.imap_unordered( step1_get_datasets, run_args ):
            all_datasets.append( dataset )
            seamr.log.info("Got dataset for %s" % dataset.dset)

# -------------------------| Merge trained expectation filters : When necessary |------------------------------------
if all_datasets[0].exp is not None:
    mergeable = False
    
    for k,m in all_datasets[0].exp.items():
        if k != 'feats':
            mergeable = hasattr(m, "merge") and conf.get("expmerge", False)
            break

    if mergeable:
        dset_exp = {}
        for target in all_datasets:
            non_target = [t for t in all_datasets if t is not target]

            tgt_exp = {}
            for t in non_target:
                for cls,m in t.exp.items():
                    if cls in tgt_exp:
                        tgt_exp[cls] = tgt_exp[cls].merge(m)
                    else:
                        tgt_exp[cls] = m

            dset_exp[ target.dset ] = tgt_exp
       # Now reset
        all_datasets = [ Dataset(d.dset, d.feats, d.classes, d.X, d.y, d.times, dset_exp[d.dset]) for d in all_datasets ]

# --------------------------------------------------------------------------------------

classifierName = str( core.build(classifier_def) )
featuresName = str( all_datasets[0].feats )

# --- possible weighting ---
expected = {}
if conf.get("pred_mask", None) == "expected":
    with Pool( args.procs ) as p:
        expected = dict( tqdm(p.imap_unordered(getExpectations, all_datasets),
                              desc="Getting expectations",
                              total=len(all_datasets)) )

with seamr.BlockWrapper("Step 2: Learning and predicting"):
    with Pool( args.procs ) as p:
        run_args = [ (classifier_def, results_dir, conf, pair) for pair in combinations(all_datasets,2) ]

        runSaveTo = os.path.join(results_dir, "xenv-%s-%s.csv" % (featuresName, classifierName))

        with open(runSaveTo, "w") as fOut:
            writer = None
            for rows, rocs in tqdm(p.imap_unordered( step2_fit_and_predict, run_args ), total=len(run_args)):
                for r in rocs:
                    r.save( results_dir )
                    
                if len(rows):
                    if writer is None:
                        writer = csv.DictWriter(fOut, sorted(rows[0].keys()))
                        writer.writeheader()
                    writer.writerows( rows )

