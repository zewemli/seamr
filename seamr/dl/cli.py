import os, sys
import seamr
from seamr import core
from seamr import sem, spatial
import networkx as nx

from torch import nn
from sklearn import metrics
import yaml

from tqdm import tqdm
from seamr.dl import Dataset, MultiEnvData, Trainer
from seamr.dl.adl_gan import ADLGAN
from seamr.dl.adl_gan import AEClassifier

from multiprocessing import Pool, cpu_count

from seamr.evaluate import ROCAnalysis

import numpy as np
import argparse

def prop(d, p, val=None):
    _, dprop = list(d.items())[0]
    if val is None:
        return dprop[p]
    else:
        dprop[p] = val
        return dprop

def load_dataset(args):
    store, dset, seqlen = args
    return Dataset(store, dset, seqlen = seqlen)

def runROC(args):
    name, c, vReal, vPred = args
    roc = ROCAnalysis( [c] )
    roc.setup(name, vReal, vPred)
    return roc
    
parser = argparse.ArgumentParser(description = 'Partitioning')
parser.add_argument('store', help='Store location')
parser.add_argument('config', help='Datasets to use')
parser.add_argument('model', help='Path to model')
parser.add_argument('--log', '-l', default="./tensorboard", help="Tensorboard log direction")
parser.add_argument('--perf', '-p', default="./results", help="Performance results saved here")
parser.add_argument('--folds','-k', default=3, help='Number of folds')
args = parser.parse_args()

config = seamr.load_config(args.config)

store = core.Store( args.store )

datasets = []
with Pool() as pool:
    worker_args = [ ( store,
                      dset,
                      config.get("seqlen", 30) ) for dset in store.datasets() ]

    for ds in tqdm(pool.imap_unordered( load_dataset, worker_args ), total=len(worker_args), desc="Loading datasets"):
        if len(datasets) > 0:
            assert datasets[-1].width() == ds.width()
            assert datasets[-1].allActs == ds.allActs
            
        datasets.append(ds)
        
    datasets.sort(key=lambda d: d.dset)

nClasses = len( ds.allActs )
insize = ds.width()

if 'cgan' in config:
    prop(config['cgan'], 'nClasses', nClasses)
    prop(config['cgan'], 'insize', prop(config['autoencoder'], 'dims'))

prop(config['classifier'], 'nClasses', nClasses)
prop(config['classifier'], 'insize', prop(config['autoencoder'], 'dims'))

prop(config['autoencoder'], 'insize', insize)

for fold in range( args.folds ):
    sys.stderr.write("Starting fold %s\n" % fold)
    
    ds_test  = [ ds for i,ds in enumerate(datasets) if (i % args.folds) == fold ]
    ds_train = [ ds for i,ds in enumerate(datasets) if (i % args.folds) != fold ]

    training_data = MultiEnvData( ds_train )
    
    trainer = Trainer( batch_size = config.get("batch_size", 32),
                       disable_cuda = config.get("disable_cuda",False) )

    if 'cgan' in config:
        model = ADLGAN( config['autoencoder'], config['cgan'], config['classifier'] )
    else:
        model = AEClassifier( config['autoencoder'], config['classifier'] )

    try:

        trainer.train( model,
                       training_data,
                       epochs = config.get("epochs", 100),
                       validate = config.get("validate", None),
                       sample = config.get("sample", None),
                       log_to = args.log,
                       save_to = args.model,
                       save_freq = config.get("save_freq", 5) )

    except KeyboardInterrupt:
        pass
    
    model.save( args.model )

    for ds in ds_test:
        reals, preds = trainer.predict(model, ds)
        real_cls = reals.sum(axis=0)

        roc_args = []
        for c,i in ds.actIndex.items():
            if real_cls[i] > 0:
                roc_args.append(["%s-%s-%s" % (str(model), c, fold), c, reals[:, i:i+1], preds[:, i:i+1] ])
        
        with Pool() as pool:
            for roc in tqdm(pool.imap_unordered(runROC, roc_args), desc="Analyzing results", total=len(roc_args)):
        
                roc.setParams(fold = fold,
                            classifier = str(model),
                            dataset = ds.dset,
                            model = str(model),
                            steps = model.step,
                            features = "crossenv")\
                    .save( args.perf )
                    