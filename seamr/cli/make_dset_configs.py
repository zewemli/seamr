import os, sys
import yaml
import seamr
from seamr import core
import argparse
from itertools import groupby


parser = argparse.ArgumentParser(description="Tool to run SEAMR experiments")

parser.add_argument("store", help="Location of datastore")
parser.add_argument("k", type=int, help="Number of blocks")
parser.add_argument("--out", default="./", help="Output location")

args = parser.parse_args()
def byK(T):
    return T[0] % args.k

store = core.Store( args.store or conf["store"] )

for n,T in groupby(sorted(enumerate(sorted(store.datasets())), key=byK), key=byK):

    with open(os.path.join(args.out, "sets%02d.yaml" % n), "w") as yaml_file:
        yaml.dump({"datasets": [ dset for _,dset in T ]}, yaml_file, default_flow_style=False)