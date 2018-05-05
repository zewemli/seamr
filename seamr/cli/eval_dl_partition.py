import os, sys
import seamr
from seamr import core
from seamr import sem, spatial
import networkx as nx

from pytorch import nn

from sklearn import metrics
import yaml

import numpy as np

import argparse

def point_time_label(sec_of_day, start, end, offset):
    h = 3600.0
    if (start * h) <= sec_of_day <= (end * h):
        return 1.0
    else:
        dist = min(abs((start * h + offset) - sec_of_day), abs((end * h + offset) - sec_of_day))
        try:
            return 0.975 ** (dist / 60)
        except:
            return 0.0

def time_label(sec_of_day, start, end):
    return max( point_time_label(sec_of_day, start, end, 0),
                point_time_label(sec_of_day, start, end, 86400) )

def graph_to_example(g):
    
    times = [
        ('cose:Night', 0, 5),
        ('cose:Morning', 5, 9),
        ('cose:MidMorning', 9, 12),
        ('cose:Midday', 12, 14),
        ('cose:Afternoon', 14, 17),
        ('cose:Twilight', 17, 21),
        ('cose:Evening', 21, 24)
    ]

    nodes = sorted(g.nodes(data=True), key=lambda T: T[1]['start'])
    seq = []
    labels = []
    total_times = 0.0 

    for n,d in nodes:
        node_time = (d['start'] + d['end']) / 2
        total_times += node_time
        sec_of_day = node_time % 86400

        feats = {'kind-%s' % d['kind'] : 1.0,
                 'ph-%s' % d['ph'] : 1.0 }

        for fname, start, end in times:
            f_weight = time_label( sec_of_day, start, end )
            if f_weight > (10.0 ** -3):
                feats[fname] = f_weight
            
        seq.append(feats)

        labels.append('yes' if d['end_label'] == d['start_label'] else 'no')
    
    return int( (total_times / len(nodes)) // 86400 ), seq, labels

'''
{
        'feature.minfreq': float,
        'feature.possible_states': _intbool,
        'feature.possible_transitions': _intbool,
        'c1': float,
        'c2': float,
        'max_iterations': int,
        'num_memories': int,
        'epsilon': float,
        'period': int,  # XXX: is it called 'stop' in docs?
        'delta': float,
        'linesearch': str,
        'max_linesearch': int,
        'calibration.eta': float,
        'calibration.rate': float,
        'calibration.samples': float,
        'calibration.candidates': int,
        'calibration.max_trials': int,
        'type': int,
        'c': float,
        'error_sensitive': _intbool,
        'averaging': _intbool,
        'variance': float,
        'gamma': float,
    }
'''

default_params = {
    'algorithm' : 'pa',
    'params': { 'max_iterations': 1000 }
}

parser = argparse.ArgumentParser(description = 'Partitioning')
parser.add_argument('store', help='Store location')
parser.add_argument('dataset', help='Datasets to use')
parser.add_argument('model', help='Path to model')
parser.add_argument('--params', '-p', help="YAML parameters for CRF")
parser.add_argument('--folds','-k', default=3, help='Number of folds')
args = parser.parse_args()

if args.params:
    with open(args.params, "rt") as f:
        params = yaml.load(f)
else:
    params = default_params

store = core.Store(args.store)

g = nx.read_graphml( store.path(args.dataset, '%s.graphml' % args.dataset) )

for fold in range( args.folds ):
    sys.stderr.write("Starting fold %s\n" % fold)
    trainer = Trainer(**params)
    testX, testY = [], []
    for day, seq, labels in map(graph_to_example, nx.weakly_connected_component_subgraphs( g ) ):
        if (day % args.folds) == 0:
            testX.append(seq)
            testY.extend(labels)
        else:
            trainer.append(ItemSequence(seq), labels)
    
    k_model = "%s-fold%02d.crf" % (args.model, fold + 1)
    trainer.train( k_model )

    tagger = Tagger()
    tagger.open( k_model )

    preds = []
    for seq in testX:
        preds.extend( tagger.tag(seq) )
    
    sys.stderr.write("F1 Score for fold %d: %0.3f\n" % ( fold, metrics.f1_score(testY, preds, pos_label = 'no' )))