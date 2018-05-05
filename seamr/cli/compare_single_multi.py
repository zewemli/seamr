import os, sys
from collections import defaultdict
import seamr
import argparse
import re
import csv

joinby = ["dataset",
          "fold",
          "relation",
          "classifier",
          "features",
          "invert_eval",
          "eval_mode",
          "activity"]


def getKey(dct):
    return tuple([ dct[t] for t in joinby ])

def load(cfile):
    with open(cfile, "rt") as f:
        reader = csv.DictReader(f)
        yield from reader

parser = argparse.ArgumentParser(description="Evaluate results from multiple")

parser.add_argument("byclass", help="Results from multi-label classifiers")
parser.add_argument("multilabel", help="Results from binary classifiers")
parser.add_argument("binary", help="Results from binary classifiers")
parser.add_argument("results", help="Save to this path")

args = parser.parse_args()

with seamr.BlockWrapper("Loading keys"):
    okKeys = set(map(getKey, load(args.byclass))) & set(map(getKey, load(args.multilabel))) & set(map(getKey, load(args.binary)))

# by class
# dataset,fold,resident,relation,classifier,eval_mode,invert_eval,features,activity,f1,precision,recall

# multi
# activity,auprg,avg_prec,classifier,dataset,eval_for,eval_mode,features,fold,invert_eval,k,model,optimalF1,relation,roc


res = defaultdict(dict)

with seamr.BlockWrapper("Reading by class"):
    for row in load(args.byclass):
        rowKey = getKey(row)
        if rowKey in okKeys:
            # These have residents as well
            d = { "f1" : row['f1'] }
            d.update( dict(zip(joinby, rowKey)) )

            res[rowKey][ row['resident'] ] = d

with seamr.BlockWrapper("Reading multilabel"):
    for row in load(args.multilabel):
        rowKey = getKey(row)
        if rowKey in okKeys and row['resident'] in res[rowKey]:
            rd = res[rowKey][ row['resident'] ]
            rd['optimalF1'] = row["optimalF1"]
            rd['wOCC'] = row['wOCC']
            rd['wMicro'] = row['wMicro']
            
with seamr.BlockWrapper("Reading binary"):
    for row in load(args.binary):
        rowKey = getKey(row)
        if rowKey in okKeys:
            resRow = res[rowKey]
            for resident, rperf in resRow.items():
                rperf[ "binaryF1" ] = row['optimalF1']

fields = joinby + ["resident", "f1", "optimalF1", "binaryF1", "wOCC", "wMicro"]

with seamr.BlockWrapper("Writing"):

    with open(args.results,"wt") as fOut:
        writer = csv.DictWriter(fOut, fieldnames=fields)
        writer.writeheader()
        for byRes in res.values():
            for resident,vals in byRes.items():
                if vals['activity'] != "other":
                    d = {"resident": resident}
                    d.update(vals)
                    writer.writerow(d)