import pickle
import os
import sys
import gzip
import json
from seamr.evaluate import Perf
from tabulate import tabulate
import csv
import argparse
from tqdm import tqdm
import re
from multiprocessing import Pool
from collections import Counter
from itertools import groupby

from sklearn.ensemble.forest import RandomForestClassifier

from seamr.evaluate import shortName

def checkHeader(fp, h):
    if not os.path.exists(fp):
        with open(fp, "w") as f:
            csv.writer(f).writerow(h)

def getShortName(n):
    if n == "DecisionTreeClassifier":
        return "DT"
    elif n == "RandomForestClassifier":
        return "RF"
    elif n == "MultinomialNB":
        return "NB"

def load(p):
    try:
        perf = Perf()
        perf.load(p)
        return perf
    except:
        raise
        print(p)
        return None
    
def is_roc_pickle(fn):
    return fn.endswith("roc.pkl")

def is_json_result(fn):
    return fn.endswith("json.gz")

def getKeys(p):
    return (set(p.storeVars) - set(["params"])) | set(p.params.keys())

def getField(f, d):

    v = d[f]
    if f == 'resident':
        return v.lower()
    elif isinstance(v, dict):
        nm,val = list(v.items())[0]
        kv = lambda k: list(val[k].keys())[0]

        if isinstance(val, dict) and 'key' in val:
            return val['key']
        elif nm == "MultinomialNB": return "naive-bayes";
        elif nm == "RandomForestClassifier": return "forest" #-n{}-{}".format(val['n_estimators'], val["max_features"] or "all");
        elif nm == "DecisionTreeClassifier": return "tree";
        elif nm == "ACE":
            exp = getShortName( list(val["expert"].items())[0][0] )
            try:
                gen = getShortName( list(val["generalist"].items())[0][0] )
            except:
                gen = "x"
            return "ACE-%s-%s" % (gen, exp)

        elif nm == "SEAMRFeatures":

            if "infoClusters" in val:
                return "SensorGroups" if val["infoClusters"] else "InfoGroups"

            elif "contextMethod" in val:
                cm = val["contextMethod"]
                if cm == "info":
                    return "%s-%s" % (cm, val.get("contextSource", "sensor"),)
                else:
                    return cm
            
            else:
                return nm

        elif nm == "CRFFeatures":
            rval = "crf"
            if val.get("withSensors", False):
                rval += "-sensors"
            if val.get("withObj", False):
                rval += "-obj"
            if val.get("withSem", False):
                rval += "-sem"
            return rval
        
        elif nm == "ACEFeatures":
            featName = kv("feats")
            xfeats = shortName.get(featName, featName)

            if "groupby" in val:
                gname = kv("groupby")

                xg = shortName.get(gname, gname)
                return "ace-%s-%s" % (xfeats, xg)
            else:
                return "ace-%s" % xfeats
        
        elif nm == "SensibleFeatures":
            featName = kv("feats")
            return "sense-%s" % shortName.get(featName, featName)

        else:
            return shortName.get(nm, nm);
    else:
        return v

parser = argparse.ArgumentParser(description="Convert performance objects to a nice simple csv")
parser.add_argument("output", help="Output file")
parser.add_argument("inputs", nargs="+", help="Input perf files/directories")

args = parser.parse_args()

files = []
values = []
for f in args.inputs:
    if os.path.isdir(f):
        for r,d,subfiles in os.walk(f):
            for sub_file in subfiles:
                if is_json_result(sub_file):
                    files.append( os.path.join(r, sub_file) )

    elif is_json_result(f):
        files.append( f )

# --------------------------------
if len(files):
    for f in files:
        fl = load(f)
        if fl is not None:
            values.append( fl )
            if len(values) >= 10:
                break

    keys = getKeys( values[0] )

    for d in values:
        keys &= getKeys(d)
    
    print(len(values))
    print(keys)

    # --------------------------------

    fields = sorted(keys - set(["importances",
                                "store",
                                "results",
                                "fit_time",
                                "name",
                                "byClass",
                                "f1Curve",
                                "datasets",
                                "confusion", 'wEv', 'wOcc', 'wMicro']))

    byclass = None

    with open("%s_conf.csv" % args.output, "wt") as fpConf:
        wconf = csv.writer(fpConf)

        wconf.writerow(fields + [ 'real','pred','cnt','weight' ])

        pool = Pool()
        for row in tqdm(pool.imap(load, files), total=len(files)):
            if row is not None:
                row_desc = [ getField(n, row) for n in fields ]

                n = 0
                for r,dct in row.confusion.items():
                    for p,cnt in dct.items():
                        n += cnt
                
                for r,dct in row.confusion.items():
                    for p,cnt in dct.items():
                        wconf.writerow( row_desc + [r, p, cnt, cnt/n] )
                        