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

def getROCRecords(fname):
    with open(fname,"rb") as fROC:
        return pickle.load(fROC).getClassPerf()

def dumpROC(files, outpath):
    rocFiles = [ f for f in files if is_roc_pickle(f) ]
    if len(rocFiles):
        with open(outpath, "wt") as fOut:
            writer = None
            with Pool() as p:
                for recordBlock in tqdm(p.imap_unordered(getROCRecords, rocFiles), desc="Getting ROC output", total=len(rocFiles)):
                    for record in recordBlock:
                        if writer is None:
                            writer = csv.DictWriter(fOut, sorted(set(record.keys()) - {"f1Curve"}), extrasaction="ignore")
                            writer.writeheader()
                        writer.writerow( record )

parser = argparse.ArgumentParser(description="Convert performance objects to a nice simple csv")
parser.add_argument("output", help="Output file")
parser.add_argument("byclass", help="Per class output file")
parser.add_argument("--multi", help="Path for ROCAnalysis output")
parser.add_argument("inputs", nargs="+", help="Input perf files/directories")

args = parser.parse_args()

rocfiles = []
files = []
values = []
for f in args.inputs:
    if os.path.isdir(f):
        for r,d,subfiles in os.walk(f):
            for sub_file in subfiles:
                if is_json_result(sub_file):
                    files.append( os.path.join(r, sub_file) )
                elif is_roc_pickle(sub_file):
                    rocfiles.append( os.path.join(r, sub_file) )

    elif is_json_result(f):
        files.append( f )
    elif is_roc_pickle(f):
        rocfiles.append( f )

if args.multi and len(rocfiles):
    dumpROC(rocfiles, args.multi)    

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
                                "byClass",
                                "f1Curve",
                                "datasets",
                                "confusion", 'wOcc', 'wMicro']))

    byclass = None

    cls_fields = ["dataset","fold","resident","relation","classifier","eval_mode","invert_eval","features"]

    checkHeader(args.byclass, cls_fields + ["activity", 'wOcc', 'wMicro',"f1","precision","recall"])
    checkHeader(args.output, fields + ["agg", "f1", "precision", "recall"])

    with open(args.byclass, "a") as byclassFP:
        with open(args.output, "a") as out:
            wr = csv.writer(out)
            byclass = csv.writer( byclassFP )

            pool = Pool()
            for row in tqdm(pool.imap(load, files), total=len(files)):
                if row is not None:
                    row_desc = [ getField(n, row) for n in fields ]

                    for agg in row.getSchemes():
                        wr.writerow(row_desc + [agg, row.f1(agg=agg), row.precision(agg=agg), row.recall(agg=agg) ])

                    f1 = row.f1()
                    precision = row.precision()
                    recall = row.recall()

                    cls_row = [ getField(f, row) for f in cls_fields ]

                    for cls in f1.keys():
                        byclass.writerow(cls_row + [cls,
                                                    row.wOcc[cls],
                                                    row.wEv[cls],
                                                    f1[cls],
                                                    precision[cls],
                                                    recall[cls] ])
                else:
                    print(row)