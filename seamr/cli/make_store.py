import logging
import multiprocessing_logging

logging.basicConfig(filename="parsing.log", level=logging.INFO)
multiprocessing_logging.install_mp_handler()


import os
import sys
from seamr import parsers

from seamr.core import Store
import argparse
from tqdm import tqdm
from datetime import datetime

from multiprocessing import Pool


parse_classes = [parsers.CASASParser,
                parsers.HorizonHouse,
                parsers.ArasParser]

def addLabelIDs(act_id, res_id, stream):
    for (line, res, act, s, e) in stream:
        if res.startswith("r") and act != "other" and 5 <= (e - s).total_seconds() <= 86400:
            yield (line, act_id[act], res_id[res], res, act, s.timestamp(), e.timestamp())

def setSensorID(sensor_ids, stream):
    for (ln, sname, state, tm) in stream:
        yield (int(ln), int(sensor_ids[sname]), float(state), float(tm.timestamp()))

def convert( dset_dir ):

    Parser = [ c for c in parse_classes if c.can_parse(os.path.basename(dset_dir)) ][0]

    dset = os.path.basename(dset_dir)
    env_name = dset.split(".")[0]
    p = Parser( dset_dir )

    residents = set()
    sensors = set()
    acts = dict(other=0.0)
    for line, res, act, _s, _e in tqdm(p.gen_labels(), desc="Setting up labels for %s" % dset):
        acts[act] = acts.get(act,0.0) + (_e - _s).total_seconds()
        residents.add(res)

    act_tups = list(enumerate(sorted(acts.keys())))
    res_id    = dict([ (r,i) for i,r in enumerate(sorted(residents)) ])
    act_id    = dict([ (a,i) for i,a in act_tups ])

    print("-"*20)
    print(dset_dir)
    for a,dur in sorted(acts.items()):
        print(": %s %0.2f hrs" % (a, dur / 3600))

    """
    Expects tuples like:
        (int id, unicode name,)
    """
    store.store_activities(act_tups, dset)

    """
    Expects tuples like:
        (int actID, int resID, unicode act, unicode res, float start, float finish)
    """
    store.store_labels(addLabelIDs(act_id, res_id, p.gen_labels()), dset)

    # -----------------------------------
    for (ln,s,state,tm) in tqdm(p.gen_events(), desc="Setting up sensors for %s" % dset):
        sensors.add(s)

    sensor_tups = list(enumerate(sorted(sensors)))

    sensor_id = dict([ (s,i) for i,s in sensor_tups ])

    """
    Expects tuples like:
        (int lid, unicode name)
    """
    store.store_sensors(sensor_tups, dset)

    """
    Expects tuples like:
        (int sensor, float state, float timestamp)
    """
    store.store_events(tqdm(setSensorID(sensor_id, p.gen_events()), desc="Storing events for %s" % dset), dset)

    print("Done with %s" % dset_dir)

    return (dset, dset.split(".")[0], len(residents))

# sem.support_activities
# ==========================================================================
parser = argparse.ArgumentParser(description="Import raw data into SEAMR formats")

parser.add_argument("store", help="store directory")
parser.add_argument("raw", nargs="+", help="Raw data")

args = parser.parse_args()

store = Store(args.store)

dirs = sorted([ d for d in args.raw if os.path.isdir(d) ])
parsable_dirs = []

for d in dirs:
    parsable = [ c.can_parse(os.path.basename(d)) for c in parse_classes ]
    if not any(parsable):
        print("Don't know how to parse: %s : will be ignoring" % d)
    else:
        parsable_dirs.append(d)
        
pool = Pool()

mapper = pool.imap_unordered
"""
Expects tuples like:
    (int id, unicode name, unicode env, int n_res)
"""
store.store_datasets([ (i,d,e,r) for (i, (d,e,r)) in enumerate(mapper(convert, parsable_dirs))])
print("Done with datasets, closing pool")
