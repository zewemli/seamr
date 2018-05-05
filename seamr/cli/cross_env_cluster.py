assert __name__ == '__main__'

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
from sklearn import cluster

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
logger = logging.getLogger('seamr')
coloredlogs.install(level='INFO')

np.seterr(all='ignore')

Dataset = namedtuple('Dataset', ['dset','feats','classes','X','y', 'times', 'exp'])

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

        iExp = { 'feats' : list(feats.genPatches(events, times)) }

        for i,cls in enumerate(classes):
            try:
                expClass = build(conf['expectations'], locals())
                if hasattr(expClass, 'update'):
                    expClass.update(iExp['feats'], y[:,i])
                
                iExp[cls] = expClass

            except KeyError:
                pass
        
        return iExp


def step1_get_datasets(args):
    '''
    Getting dataset features, meant to be run in a subprocess
    '''
    conf, store, dset, features_def, n_days, step_by, max_steps = args

    days = list(range(n_days))

    with seamr.BlockWrapper('Getting features for %s' % dset):
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
    clusterer_def, results_dir, conf, pair = args
    setA, setB = pair
    cose = sem.COSE()
    epsilon = 10.0 ** -5
    
    actModels = cose.get_activity_models()

    featsName = str(setA.feats)

    commonFeats = sorted(set(setA.feats.feat_names) & set(setB.feats.feat_names))
    commonClasses = set(setA.classes) & set(setB.classes)
    
    # Never build an 'other' classifier
    commonClasses.discard('other')

    if not conf.get('model_filter', False):
        aX = setA.feats.intersect(commonFeats, setA.X)
        bX = setB.feats.intersect(commonFeats, setB.X)
    
    results = []

    a_model = core.build(clusterer_def).fit( aX )
    b_model = core.build(clusterer_def).fit( bX )

    ba_cluster = b_model.predict( aX )
    aa_cluster = a_model.predict( aX )

    ab_cluster = a_model.predict( bX )
    bb_cluster = b_model.predict( bX )

    # Do the clusterings agree?
    ab_agreement = metrics.v_measure_score( aa_cluster, ba_cluster )
    ba_agreement = metrics.v_measure_score( bb_cluster, ab_cluster )

    for cls in commonClasses:

        # Select only the class of interest
        aY = setA.y[ :, setA.classes.index(cls) ]
        bY = setB.y[ :, setB.classes.index(cls) ]

        # Given a model in the target environment, is it consistent with the
        # model labels in source environment?
        xfer_homo_a = metrics.homogeneity_score( aY, ba_cluster )
        xfer_homo_b = metrics.homogeneity_score( bY, ab_cluster )

        local_homo_a = metrics.homogeneity_score( aY, aa_cluster )
        local_homo_b = metrics.homogeneity_score( bY, bb_cluster )

        results.append({
            'activity' : cls,
            'a': setA.dset,
            'b': setB.dset,
            'ab_agreement': ab_agreement,
            'ba_agreement': ba_agreement,
            'b_select_homo': xfer_homo_a,
            'a_select_homo': xfer_homo_b,
            'b_local_homo': local_homo_a,
            'a_local_homo': local_homo_b
        })

    return results
    
def combined_yaml(configs):
    conf = {}
    for c in configs:
        if c.endswith('.yaml'):
            with open(c) as cf:
                conf.update( yaml.load(cf) )
    return conf

# ---------------------------------------
# CLI Argument Parser
# ---------------------------------------
default_conf = '''
k: 5
store: '/data/phd/datasets/store'
results: './results'
eval_for: 30
eval_mode: 'k-blocked'
validate_for: 30
max_steps: 15
step_by: 30
invert_eval: false
eval_batch: 1000
features:
    seamr.features.semantic.CrossEnvFeatures:
        withTimes: true
        avg: 0
'''

parser = argparse.ArgumentParser(description='Tool to run SEAMR experiments')

parser.add_argument('config', nargs='+', help='Config file(s) for experiment. The default is: \n%s' % default_conf)
parser.add_argument('--store', help='Path to data')
parser.add_argument('--results', help='Save to this path')
parser.add_argument('--procs', type=int, default=int(cpu_count()/2), help='Number of processes')

args = parser.parse_args()

for c_arg in args.config:
    assert os.path.exists(c_arg), '%s does not exist :(' % c_arg

conf = load_config(default_conf, *args.config)

store = core.Store( args.store or conf.get('store',None), use_sem_labels=True )

run_name = '-'.join([ os.path.basename(y) for y in args.config ]).replace('.yaml','')
results_dir = os.path.join(args.results or conf.get('results','./results'), run_name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(os.path.join(results_dir, 'config.yaml'), 'wt') as fOut:
    yaml.dump(combined_yaml(args.config), default_flow_style=False)

features_def = conf['features']
clusterer_def = conf['cluster']

n_eval_days = conf.get('eval_for', 30)
t_step_by = conf.get('step_by', 15)
t_max_steps = conf.get('max_steps', 20)


# Step 1: Get all the datasets
with seamr.BlockWrapper('Step 1: Getting matrices'):
    with Pool( args.procs ) as p:
        run_args = [ (conf, store, ds, features_def, n_eval_days, t_step_by, t_max_steps ) for ds in store.datasets() ]

        all_datasets = []
        for dataset in p.imap_unordered( step1_get_datasets, run_args ):
            all_datasets.append( dataset )
            seamr.log.info('Got dataset for %s' % dataset.dset)

# -------------------------| Merge trained expectation filters : When necessary |------------------------------------
if all_datasets[0].exp is not None:
    mergeable = False
    
    for k,m in all_datasets[0].exp.items():
        if k != 'feats':
            mergeable = hasattr(m, 'merge') and conf.get('expmerge', False)
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

classifierName = str( core.build(clusterer_def) )
featuresName = str( all_datasets[0].feats )

# --- possible weighting ---
expected = {}
if conf.get('pred_mask', None) == 'expected':
    with Pool( args.procs ) as p:
        expected = dict( tqdm(p.imap_unordered(getExpectations, all_datasets),
                              desc='Getting expectations',
                              total=len(all_datasets)) )

with seamr.BlockWrapper('Step 2: Learning and predicting'):
    with Pool( args.procs ) as p:
        run_args = [ (clusterer_def, results_dir, conf, pair) for pair in combinations(all_datasets,2) ]

        runSaveTo = os.path.join(results_dir, 'xcluster-%s-%s.csv' % (featuresName, classifierName))

        if "(" in runSaveTo:
            runSaveTo = runSaveTo[:runSaveTo.index("(")]

        with open(runSaveTo, 'w') as fOut:
            writer = None
            for rows in tqdm(p.imap_unordered( step2_fit_and_predict, run_args ), total=len(run_args)):
                    
                if len(rows):
                    if writer is None:
                        writer = csv.DictWriter(fOut, sorted(rows[0].keys()))
                        writer.writeheader()
                    writer.writerows( rows )

