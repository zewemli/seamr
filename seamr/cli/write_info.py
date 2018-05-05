import os
import sys

import seamr
from seamr import core, spatial, sem

from datetime import datetime
from functools import reduce

store = core.Store("/phd/data/semstore")

cose = sem.COSE()

models = cose.get_activity_models()

mObjs = { k : m.concepts() for k,m in models.items() }

"""

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    FUCSIA = '\033[95m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'
"""

for ds in store.datasets():

    ps = 0
    sensors = store.get_sensors(ds)

    names = set()

    labels = store.get_labels(ds, days=[1]).getResSubsets()

    with open("%s.log" % ds, "wt") as fout:
        for e in store.get_events_lps(ds, days=[1]):
            if e.timestamp != ps:
                pt = datetime.fromtimestamp(ps)
                p_labels = [ l.strLabel(ps) for l in labels ]

                obj_sets = reduce( lambda a,b: a|b, [ mObjs.get(l, set()) for l in p_labels ], set())

                lbl = " - ".join(p_labels)
                fout.write(pt.strftime("%H:%M:%S") + "] ")
                fout.write(lbl)
                fout.write(" |\033[94m ")
                fout.write(" - ".join( sorted(obj_sets & names) ))
                fout.write("\033[0m")

                fout.write(" |\033[95m ")
                fout.write(" - ".join( sorted(names - obj_sets) ))
                fout.write("\033[0m")
                fout.write("\n")
                names = set()
            
            names.add( sensors[e.sensorID].name )
            ps = e.timestamp
    