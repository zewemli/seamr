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

import logging
import coloredlogs, logging
logger = logging.getLogger("seamr")
coloredlogs.install(level='INFO')

class ReRunFilter:
    def __init__(self, results_dir):
        self._has = set()
        reg = re.compile("(?P<dset>\w+)\[(?P<res>\w+)\]fold(?P<fold>\d+)")

        if os.path.exists(results_dir):
            for fn in os.listdir(results_dir):
                fsplt = fn.split(".")
                if fsplt[-1] == "gz":
                    m = reg.match(fsplt[0])
                    if m:
                        self._has.add( (m.group("dset"), m.group("res"), int(m.group("fold"))) )
    
    def __call__(self, dset, res, fold):
        return (dset, res, fold,) in self._has


def getFeatures(definition, multi_input_model = False, args = {}):

    assert isinstance(definition, dict), "Features must be a list or dict"
    FeatType, feat_args = list(definition.items())[0]

    if "stack" not in feat_args:
        feat_args["stack"] = not multi_input_model
    
    return construct(FeatType, feat_args, args)

def describe(X, feats, imp):
    assert X.shape[1] == len(feats.feat_names)
    var = X.var(axis=0)
    low = X.min(axis=0)
    high = X.max(axis=0)

    print("-"*20, "| %s Features |" % feats.__class__.__name__, "-"*20)
    for i,n in enumerate(feats.feat_names):
        print( "%s | %0.5f | var %0.5f : low %0.3f : high %0.3f" % (n, imp[i], var[i], low[i], high[i]) )
    print("-"*50)

def get_results(model,
                feats,
                conf,
                events,
                times,
                batch_size,
                reals,
                targets=[],
                train_labels=[],
                index=None,
                multioutput=False, **args):

    params = {}
    params.update(conf)
    params.update(args)

    name = getattr(model, "name", str(model))

    if isinstance(reals[0], str) or isinstance(reals[0], int):
        reals = [reals]

    preds = []
    eval_fmt = "Evaluating {}[{}] fold {}".format(args.get("dataset","-"), targets, args.get("fold","-"))
    roc = None

    t_fmt = "{}[%s]-fold-{}".format(args.get("dataset","-"), args.get("fold","-"))
    
    with seamr.BlockWrapper(eval_fmt):
        
        if hasattr(model, "reset"):
            model.reset(events, times, reals=reals)
        
        X = feats.make_dataset( events, times )

        if multioutput:
            """
            # So we're going to assume this is an NN with dense return matrices 
            """
            preds = model.predict( feats.make_dataset( events, times ) )

        else:
            """
            And here we will assume we have an SKLearn model
            """
            assert len(targets) == 1
            tgt = targets[0]
            real = reals[0]
            probs = model.predict_proba(X)

            """
            print("-" * 100)
            print("---: " + params.get("relation") + " :---" )
            rvec = np.array(real)
            for i,p in enumerate(model.points):
                p.checkPoint(rvec, probs[:,i])
            print("-" * 100)
            exit()
            """

            preds = [ model.classes_[ probs.argmax(axis=1) ] ]

            roc = ROCAnalysis( model.classes_ )
            roc.setup( t_fmt % tgt, real, probs)

            
    assert len(targets) == len(reals), (len(targets), len(reals))
    assert len(targets) == len(preds), (len(targets), len(preds))

    for r,p in zip(reals, preds):
        assert len(r) == len(p)
        
    return [ Perf()\
                .setup(t_fmt % t, r, p, times)\
                .setParams(params,
                            resident=t,
                            model=name,
                            roc=roc,
                            features=str(feats))
            for t,r,p in zip(targets,reals,preds) ]

def get_importances(model,feats):
    if hasattr(model, "feature_importances_") and hasattr(feats, "feat_names"):
        return list(zip(feats.feat_names, model.feature_importances_))

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
    seamr.features.casas.CASASFeatures:
        qlen: 30
        sensors: null

"""

parser = argparse.ArgumentParser(description="Tool to run SEAMR experiments")

parser.add_argument("config", nargs="+", help="Config file(s) for experiment. The default is: \n%s" % default_conf)
parser.add_argument("--store", help="Path to data")
parser.add_argument("--sem", default=False, action="store_true", help="Use mapped semantic labels")
parser.add_argument("--results", help="Save to this path")

args = parser.parse_args()

for c_arg in args.config:
    assert os.path.exists(c_arg), "%s does not exist :(" % c_arg

conf = load_config(default_conf, *args.config)

conf_name    = ".".join([ basename(x).split(".")[0] for x in sorted(args.config) ])

use_sem_labels = conf.get("sem_labels", args.sem)

store = core.Store( args.store or conf["store"], use_sem_labels = use_sem_labels )

k            = conf["k"]
results_dir  = args.results or conf["results"]
eval_for     = conf["eval_for"]
validate_for = conf["validate_for"]

max_steps    = conf["max_steps"]
step_by      = conf["step_by"]
invert_eval  = conf["invert_eval"]
eval_batch   = conf["eval_batch"]

datasets       = conf["datasets"] or sorted(store.datasets())
classifier_def = conf["classifier"]
features_def   = conf["features"]
eval_mode      = conf["eval_mode"]

ModelClass, model_cls_args = list(classifier_def.items())[0]

multioutput    = conf.get('multioutput',False) or getattr(ModelClass, "multioutput", False)

cose = sem.COSE()

if eval_mode == "round-robin":
    # Basic Rule, you must match times exactly when using a round-robin strategy
    step_by = 0.0
    max_steps = 0

print("Running over datasets: %s" % (datasets,))

run_filter = ReRunFilter( results_dir )

# ---------
for dset in datasets:

    env = dset.split(".")[0]

    with seamr.BlockWrapper("Evaluating %s" % dset):

        res = None
        if ":" in dset:
            res,dset = dset.split(":")

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
                        dset, args.store or conf["store"]))
            continue
               
        sensors = store.get_sensors( dset )

        for fold, (f_times, f_days) in enumerate(folds):
            (train_times, test_times, validation_times) = f_times
            (train_days, test_days, validation_days) = f_days

            with seamr.BlockWrapper( "%s| %s : %s |%s" % ("-"*10, dset, fold, "-"*10) ):

                logger.info("Dims: Train %s : Test %s : Validation %s" % ( len(train_times),
                                                                     len(test_times),
                                                                     len(validation_times or []) ))

                all_labels = store.get_labels(dset, resident=res, days=list(range(eval_for + validate_for)))
                targets = sorted( all_labels.residents.values() )
                validation_labels = [None] * len(targets)

                train_events      = lambda: store.get_events_lps(dset, days=train_days)
                test_events       = lambda: store.get_events_lps(dset, days=test_days)
                validation_events = lambda: store.get_events_lps(dset, days=validation_days)

                one_hot_label = getattr(ModelClass, "one_hot_label", False)

                train_labels = all_labels.make_dataset(train_times, one_hot = one_hot_label)
                test_labels  = all_labels.make_dataset(test_times,  one_hot = one_hot_label)

                FeatType, feat_args = list(features_def.items())[0]
                features = getFeatures(features_def,
                                        multi_input_model = getattr(ModelClass, "multi_input", False),
                                        args=locals())

                # Fit the labels
                if hasattr(features, "fit"):
                    with seamr.BlockWrapper("Fitting features %s" % features.__class__.__name__):
                        features.fit(train_labels, train_events, train_times)

                if hasattr(features,"describe"):
                    features.describe()

                if validation_times:
                    validation_labels = all_labels.make_dataset(validation_times,  one_hot = one_hot_label)
                
                if len(targets) < len(train_labels):
                    train_labels = [train_labels]
                    
                if len(targets) < len(test_labels):
                    test_labels = [test_labels]

                if len(targets) < len(validation_labels):
                    validation_labels = [validation_labels]
                
                assert len(targets) == len(train_labels), ("Lengths train", len(targets), len(train_labels), len(test_labels))
                assert len(targets) == len(test_labels), ("Lengths test", len(targets), len(test_labels))

                with seamr.BlockWrapper("Creating training dataset for %s | %s" % 
                        (features.__class__.__name__, len(train_times))):
                    train_mat = features.make_dataset(train_events, train_times)
                
                try:
                
                    if isinstance(train_mat, list):
                        logger.info("Training sizes : {}".format([ m.shape for m in train_mat ]))
                    else:
                        logger.info("Training size : {}".format( train_mat.shape ))
                
                except AttributeError:
                    pass
                
                # --------------------------------------------------------------------------
                #                          | Fitting the DL Model |
                # --------------------------------------------------------------------------

                if multioutput:
                    model = construct(ModelClass, model_cls_args, locals())
                    
                    start_fit = time.monotonic()                    
                    model.fit(train_mat, train_labels)
                    fit_time = time.monotonic() - start_fit
                    logger.info("Multi-output training took %0.3f seconds" % (fit_time))

                    for rel,ev,lbls,tm in [ ("test", test_events, test_labels, test_times),
                                        ("validation", validation_events, validation_labels, validation_times) ]:

                        if tm:
                            for i,rPerf in enumerate(get_results(model,
                                                                features,
                                                                conf,
                                                                ev,
                                                                tm,
                                                                eval_batch,
                                                                lbls,
                                                                dataset  = dset,
                                                                targets  = targets,
                                                                train_labels = train_labels,
                                                                multioutput = multioutput,
                                                                index    = all_labels.lblName,
                                                                relation = rel,
                                                                importances = get_importances(model, features),
                                                                fit_time = fit_time,
                                                                fold     = fold)):
                                
                                rPerf.save(results_dir).show(resident = targets[i], rel=rel, with_confusion=True)

                else:

                    # In this situation you MUST build a seperate classfier for each target
                    
                    xferLabel = sem.LabelTransfer(store)

                    for i_res, i_train, i_test, i_valid in zip(targets, train_labels, test_labels, validation_labels):

                        if len(set(i_test)) < 2 or run_filter(dset, i_res, fold):
                            continue

                        """
                        i_train = xferLabel.remap(i_train)
                        i_test = xferLabel.remap(i_test)
                        i_valid = xferLabel.remap(i_valid)
                        """

                        model = construct(ModelClass, model_cls_args, locals())

                        start_fit = time.monotonic()
                        with seamr.BlockWrapper("Fitting model %s" % model.__class__.__name__):
                            model.fit(train_mat, i_train)

                        fit_time = time.monotonic() - start_fit

                        for rel,ev,lbls,tm in [ ("test", test_events, i_test, test_times),
                                        ("validation", validation_events, i_valid, validation_times) ]:

                            if tm:
                                
                                for rPerf in get_results(model,
                                                        features,
                                                        conf,
                                                        ev,
                                                        tm,
                                                        eval_batch,
                                                        lbls,
                                                        dataset  = dset,
                                                        targets  = [i_res],
                                                        train_labels = i_train,
                                                        multioutput = multioutput,
                                                        index    = all_labels.lblName,
                                                        relation = rel,
                                                        importances = get_importances(model, features),
                                                        fit_time = fit_time,
                                                        fold     = fold):
                                    
                                    assert len(rPerf) == len(tm), "Didn't get the right number of examples | %s != %s" % (len(rPerf), len(tm))

                                    if rPerf.roc:
                                        rPerf.roc.setParams(rPerf.params).show().save(results_dir)

                                    rPerf.save(results_dir).show(resident = i_res, rel=rel, with_confusion=True)

