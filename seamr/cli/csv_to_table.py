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

parser = argparse.ArgumentParser(description="CSV to table")
parser.add_argument("csv", help="Input csv")
parser.add_argument("--format", "-f", default="plain", help="Format for the tabulate package")
parser.add_argument("--columns", "-c", default=[], nargs = "+", help="Only output these columns")

parser.add_argument("--remove", "-x", default=[], nargs = "+", help="Remove these columns")

parser.add_argument("--nice", "-n", default=[], nargs = "+", help="List of columns to treat as floating point values, will be truncated to 2 units of precision.")

parser.add_argument("--sort", "-s", default=[], nargs = "+", help="Sort the table by these columns")

args = parser.parse_args()

with open(args.csv,"rt") as f:
    reader = csv.reader(f)
    header = next(reader)

    table = list(reader)

    for col in args.nice:
        c = header.index(col)
        for row in table:
            row[c] = "%0.2f" % float(row[c])
    
    if args.columns:
        indices = [ header.index(col) for col in args.columns if col in header ]

        header = [ header[i] for i in indices ]
        table = [ [ row[i] for i in indices ] for row in table ]

    if args.remove:
        indices = [ i for i,col in enumerate(header) if col not in args.remove ]

        header = [ header[i] for i in indices ]
        table = [ [ row[i] for i in indices ] for row in table ]

    if args.sort:
        indices = [ header.index(col) for col in args.sort ]
        table.sort(key=lambda row: tuple([ row[i] for i in indices ]))
    
    print(tabulate(table, header, tablefmt=args.format))