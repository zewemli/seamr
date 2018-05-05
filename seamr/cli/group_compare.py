import os, sys
from itertools import groupby, combinations, product
from collections import defaultdict
from tabulate import tabulate
import seamr
import argparse
import re
import csv

import numpy as np
from scipy.stats import ttest_1samp

cond_cols = ["dataset",
          "fold",
          "relation",
          "model",
          "classifier",
          "features",
          "invert_eval",
          "eval_mode",
          "activity",
          "resident"]

perf_cols = ["f1","precision","recall"]

class Matcher:
    def __init__(self, groupers, conds):
        self.indices = [ groupers.index(k) for k in conds ]
    
    def subkey(self, k):
        return tuple([ k[i] for i in self.indices ])

    def batch(self, records):
        if len(self.indices) == 0:
            yield (None, records)
        else:
            keys = [ self.subkey(T[0]) for T in records ]
            subRecs = sorted(zip(keys, records))
            for K,tItems in groupby(subRecs, key=lambda T: T[0]):
                yield K, [ T for _,T in tItems ]

def getKey(joinby, dct):
    return tuple([ dct.get(t, None) for t in joinby ])

def load(cfile):
    with open(cfile, "rt") as f:
        reader = csv.DictReader(f)
        yield from reader

def filters(args, stream):
    for row in stream:
        if args.where:
            if not any([ v in args.where for v in row.values() ]):
                continue
        
        if args.without:
            if any([ v in args.without for v in row.values() ]):
                continue
        
        yield row

parser = argparse.ArgumentParser(description="Evaluate results from multiple")

parser.add_argument("input", help="Input file")
parser.add_argument("by", choices=cond_cols, help="Conditional value")
parser.add_argument("val", default="f1", help="Target value")
parser.add_argument("results", help="Save to this path")
parser.add_argument("--agg", default="occ", help="Aggregation to use")
parser.add_argument("--pvalue", "-p", default=0.05, type=float, help="P-Value for 1-sample two-sided t-test")

parser.add_argument("--given", nargs="+", default=[], help="Conditionals")
parser.add_argument("--where", nargs="+", default=[], help="Only include rows with these values")
parser.add_argument("--without", nargs="+", default=[], help="Remove rows with these values")

args = parser.parse_args()

args.where = set(args.where)
args.without = set(args.without)

# dataset,fold,relation,classifier,features,invert_eval,eval_mode,activity,resident
groupers = [ k for k in cond_cols if k != args.by ]

input_csv = list(load(args.input))

try:
    records = [ (getKey(groupers, r), r[args.by], float(r[args.val]))
                for r in filters(args, input_csv)
                if r.get('agg', args.agg) == args.agg ]
except KeyError as err:
    raise KeyError("%s | available keys are %s" % (err, sorted(input_csv[0].keys())))

header = ["%s_a" % args.by,
            "%s_b" % args.by,
            "%s_dmin" % args.val,
            "%s_dmean" % args.val,
            "%s_dmedian" % args.val,
            "%s_dmax" % args.val,
            "%s_dstd" % args.val,
            "%s_dpval" % args.val,
            "num" ] + args.given

table = []

matcher = Matcher(groupers, args.given)

kFunc = lambda T: T[0]

with open(args.results, "wt") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for groupCond, gRecords in matcher.batch( records ):

        diffs = defaultdict(list)

        for g, items in groupby(sorted(gRecords, key=kFunc), key=kFunc):
            gitems = [(k,v) for  _, k,v in items]

            for a,av in gitems:
                for b,bv in gitems:
                    if a != b:
                        try:
                            diffs[(a,b)].append( av  ) # bv / max(av, bv)
                        except ZeroDivisionError:
                            diffs[(a,b)].append(0)

        for (a,b),vals in sorted(diffs.items()):
            vals = np.array( vals )

            _, pval = ttest_1samp(vals, 0.0, axis=0)

            row = [
                a,b,
                vals.min(),
                vals.mean(),
                np.median(vals),
                vals.max(),
                vals.std(),
                pval <= args.pvalue,
                vals.size
            ]

            if groupCond:
                row.extend(groupCond)

            table.append(row)

            writer.writerow(row)

print(tabulate(table, headers=header))