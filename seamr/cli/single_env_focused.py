assert __name__ == "__main__"
import os, sys
import time
from datetime import datetime
from itertools import chain
from collections import Counter
import yaml, json, csv
import gzip
from tqdm import tqdm
import pstats, cProfile
import re

from os.path import join,basename,dirname, exists

from random import random

import argparse
import numpy as np
import shutil
import math

import inspect

from seamr.features.semantic import PatchFeatures
from seamr.evaluate import Perf, ROCAnalysis
from seamr.core import construct
from seamr import load_config
from seamr import core
from seamr import sem
import seamr

from multiprocessing import Pool

import logging
import coloredlogs, logging
logger = logging.getLogger("seamr")
coloredlogs.install(level='INFO')

def combined_yaml(configs):
    conf = {}
    for c in configs:
        if c.endswith(".yaml"):
            with open(c) as cf:
                conf.update( yaml.load(cf) )
    return conf

def getLabelMatrix(store, dset, days, times):
    events = lambda: store.get_events(dset, days = days)
    
    realLabels = store.get_labels(dset, days=days)
    classes = np.array(sorted(set(['other']) | set( l.activity for l in realLabels.labels )))   
    classIndex = { c:i for i,c in enumerate(classes) }

    reals = np.zeros( (len(times), classes.size) )

    for res in realLabels.make_dataset(times, prefer_mat=True):
        for row, l in enumerate(res):
            reals[row, classIndex[l]] = 1.0
    
    return classes, reals

def getMat(dset, features, days, times):
    events = lambda: store.get_events(dset, days = days)
    return features.make_dataset( events, times )

def fit_single(args):
    cls, modelDef, X, y = args
    if cls == "other":
        return cls, None
    else:
        try:
            return cls, core.build(modelDef).fit(X,y)
        except:
            return cls, None

def prepare_expectations(conf, store, dset, classes, events, times, labels):
    if conf.get("expectations", None) is None:
        return None
    else:
        expFeats = PatchFeatures(store, dset, decay=0)
        exp = {"feats": expFeats}
        trainPatches = None

        for i,cls in enumerate(classes):
            try:
                expClass = core.build(conf['expectations'], {"store":store, "cls":cls})
                if trainPatches is None and hasattr(expClass, "update"):
                    trainPatches = list( expFeats.genPatches(events, times) )
                
                if hasattr(expClass, "update"):
                    expClass.update(trainPatches, labels[:, i])
                
                exp[cls] = expClass
            except KeyError:
                pass
        
        return exp

def get_exp_patches(conf, exp, events, times):
    if conf.get("expectations", None) is None or events is None:
        return None
    else:
        return list( exp['feats'].genPatches(events, times) )

def handle_expectations(conf, pred_vec, models, cls, t_patches):
    if isinstance(models, dict) and cls in models:
        scores = np.array(list( models[cls]( t_patches ) ))
        scored = scores * pred_vec
        return scored / scored.max()
    else:
        return pred_vec

def get_results(args):
    store, dset, conf = args
    results_dir = conf.get("results", "./results")

    k = conf.get("k", 5)
    eval_for = conf.get("eval_for", 30)
    validate_for = conf.get("validate_for", 30)
    eval_mode = conf.get("eval_mode", "k-blocked")
    invert_eval = conf.get("invert_eval", False)
    step_by = conf.get("step_by", 15)
    max_steps = conf.get("max_steps", 20)

    eval_days = list(range(eval_for))

    log_params = {
        "eval_mode": eval_mode,
        "eval_for": eval_for,
        "k": k,
        "invert_eval": invert_eval,
    }

    sensors = store.get_sensors(dset)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    all_results = []

    # ---------------------------------------
    try:
        folds = core.get_schedule(store,
                                dset,
                                k,
                                eval_for,
                                validate_for,
                                eval_mode,
                                inverse = invert_eval,
                                step = step_by,
                                max_steps = max_steps)
    except:
        logger.warn("Unable to get schedule for %s from %s. This is usually caused by an underlying data problem." % (
                    dset, store.basedir))
        return None

    val_mat = None
    val_labels = None
    val_classes = None
    validation_times = None
    val_events = None
    val_patches = None

    classifier = core.build(conf['classifier'])

    for fold, (f_times, f_days) in enumerate(folds):
        (train_times, test_times, validation_times) = f_times
        (train_days, test_days, validation_days) = f_days

        features = core.build(conf['features'], locals())
        fold_name = "%s:f%02d:%s:%s" % (dset, fold, core.build(conf['classifier'], locals()), features)

        train_events = lambda: store.get_events(dset, days = train_days)
        test_events = lambda: store.get_events(dset, days = test_days)

        if hasattr(features, "getPatchLevels"):
            features.getPatchLevels(train_events, train_times)

        with seamr.BlockWrapper("%s: Creating datasets" % (fold_name,)):

            train_mat = getMat(dset, features, train_days, train_times)
            test_mat = getMat(dset, features, test_days, test_times)

            classes, train_labels = getLabelMatrix(store, dset, train_days, train_times)

            test_classes, test_labels = getLabelMatrix(store, dset, test_days, test_times)

            expModels = prepare_expectations(conf, store, dset,
                                              classes,
                                              train_events, train_times,
                                              train_labels)
            
            test_patches = get_exp_patches(conf, expModels, test_events, test_times)


        if len(validation_days) and val_mat is None:
            with seamr.BlockWrapper("%s: Creating Validation dataset" % (fold_name,)):
                val_events = lambda: store.get_events(dset, days=list(validation_days))

                val_mat = features.make_dataset( val_events, validation_times )
                
                if isinstance(val_mat, np.ndarray) and isinstance(train_mat, np.ndarray):
                    assert val_mat.shape[1] == train_mat.shape[1]

                val_classes, val_labels = getLabelMatrix(store, dset, validation_days, validation_times)
                val_patches = get_exp_patches(conf, expModels, val_events, validation_times)
            

        classifiers = {}

        fit_args = [ (cls, conf['classifier'], train_mat, train_labels[:, i]) for i,cls in enumerate(classes) ]
        modelGen = map(fit_single, fit_args)
        classifiers = dict( tqdm(modelGen, desc="%s: Fitting classes" % (fold_name,), total=len(classes)) )
        
        eval_groups = [("test",test_labels, test_classes, test_mat, test_patches),
                       ("validation", val_labels, val_classes, val_mat, val_patches)]
        
        for (t_rel, t_labels, t_classes, t_mat, t_patches) in eval_groups:
            if t_classes is not None:
                with seamr.BlockWrapper("%s: Predicting %s data" % (fold_name, t_rel,)):
                    t_probs = np.zeros_like( t_labels )
                    for i,cls in enumerate(t_classes):
                        if cls != "other" and classifiers.get(cls, None) is not None:
                            try:
                                t_probs[:,i] = handle_expectations(conf,
                                                                    classifiers[cls].predict_proba(t_mat)[:,1],
                                                                    expModels,
                                                                    cls,
                                                                    t_patches)
                            except IndexError:
                                pass
                    
                    roc = ROCAnalysis( t_classes )\
                        .setup(fold_name, t_labels, t_probs)\
                        .setParams(log_params,
                                relation = t_rel,
                                classifier = str(classifier),
                                features = str(features),
                                fold = fold,
                                dataset = dset,
                                k = k)\
                        .show()\
                        .save(results_dir)
                    
                    all_results.extend( roc.getClassPerf() )

    return all_results

def get_importances(model,feats):
    if hasattr(model, "feature_importances_") and hasattr(feats, "feat_names"):
        return list(zip(feats.feat_names, model.feature_importances_))

# ---------------------------------------
# CLI Argument Parser
# ---------------------------------------

parser = argparse.ArgumentParser(description="Tool to run SEAMR experiments")

parser.add_argument("config", nargs="+", help="Config file(s) for experiment.")
parser.add_argument("--store", help="Path to data")
parser.add_argument("--sem", default=False, action="store_true", help="Use mapped semantic labels")
parser.add_argument("--results", help="Save to this path")

args = parser.parse_args()

for c_arg in args.config:
    assert os.path.exists(c_arg), "%s does not exist :(" % c_arg

conf = load_config({}, *args.config)
if args.results:
    conf['results'] = args.results

run_name = "-".join([ os.path.basename(y) for y in args.config ]).replace(".yaml","")
conf['results'] = os.path.join( conf['results'], run_name )
if not os.path.exists( conf['results'] ):
    os.makedirs( conf['results'] )

with open(os.path.join(conf['results'], "config.yaml"), "wt") as fOut:
    fOut.write( yaml.dump(combined_yaml(args.config), default_flow_style=False) )

store = core.Store( args.store or conf["store"], 
                    use_sem_labels = conf.get("sem_labels", args.sem) )

datasets = store.datasets()

seamr.log.info("Running over datasets: %s" % (datasets,))

run_args = [ (store, ds, conf) for ds in datasets ]

with open( os.path.join(conf['results'], 'multilabel_single.csv'), "wt" ) as fOut:
    writer = None
    with Pool() as pool:
        for resblock in pool.imap_unordered(get_results, run_args):
            if writer is None:
                writer = csv.DictWriter( fOut, sorted(resblock[0].keys()) )
                writer.writeheader()
            
            writer.writerows( resblock )
