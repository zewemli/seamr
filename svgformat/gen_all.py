#!/usr/bin/env python3
import os, sys
from seamr import core, sem
import argparse
import time
from subprocess import Popen
from collections import defaultdict

parser = argparse.ArgumentParser(description="Tool to run SEAMR experiments")

parser.add_argument("store", help="store location")
parser.add_argument("--dataset", nargs="+", help="Dataset")
parser.add_argument("--info", action="store_true", help="Dataset")
args = parser.parse_args()

cose = sem.COSE()
store = core.Store( args.store )
datasets = sorted(store.datasets())

sensor_sets = defaultdict(set)
for dset in datasets:
  env = dset.split(".")[0]

  sensors = store.get_sensors( dset )
  sensors_to_find = [ s.name.split(":")[-1] for s in sensors if cose.sensorInfoType(s.name) != "cose:EnvironmentStateSensor" ]
  sensor_sets[env] |= set(sensors_to_find)

procs = {}
for dset in datasets:
  env = dset.split(".")[0]
  if env not in procs:
    sensors_to_find = sorted(sensor_sets[env])
    env_svg = "../layouts/%s.svg" % env    
    pArgs = ["phantomjs", "./delauney.js", env_svg, env] + sensors_to_find
    
    if args.info:
      print(" ".join(pArgs))
    else:
      print("Starting %s " % (env_svg,))
      pr = Popen(pArgs)
      procs[env] = pr

while len(procs):
  keys = list(procs.keys())
  for k in keys:
    procs[k].poll()
    if procs[k].returncode is None:
      time.sleep(0.01)
    else:
      kproc = procs.pop(k)
      print("%s | Done with %s" % (kproc.returncode, k))
