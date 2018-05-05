import sys
import logging
import yaml
import os
import math
import time

from itertools import product, chain, combinations
from importlib import import_module

import inspect

import collections
from collections import deque, defaultdict, Counter
from datetime import datetime
import random

import hashlib

class Registry(object):
    def __init__(self):
        self.items = {}
    
    def __contains__(self, key):
        return key in self.items

    def add(self, item):
        self.items[item.__name__] = item
        return self.items[item.__name__]

    def __getitem__(self, key):
        return self.items[key]
    
    def build(self, name, *args, **kw):
        return self.items[name](*args, **kw)

class TimeHistogram:
    def __init__(self, bins_per_day = 48):

        self.bins_per_day = bins_per_day
        self.secs_per_bin = 86400 // bins_per_day
        self.histogram = {} 

    def sec_of_day( self, tm ):
        dt = datetime.fromtimestamp(tm)

        return dt.hour * 3600 + dt.minute * 60 + dt.seconds

    def fit(self, stream):

        for (tm, obs) in stream:
            
            b = self.sec_of_day(tm) // self.secs_per_bin
            
            for o in obs:
                try:
                    self.histogram[o][b] += 1
                except KeyError:
                    self.histogram[o] = np.zeros(self.bins_per_day, dtype=np.float32)
                    self.histogram[o][b] += 1

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def toProb(d):
    s = float(sum( d.values() ))
    for k in d.keys():
        d[k] = d[k] / s
    return d

def npmi( joint_cnt ):
    a_cnt = defaultdict(float)
    b_cnt = defaultdict(float)
    
    for (a,b),w in joint_cnt.items():
        a_cnt[a] += w
        b_cnt[b] += w

    a_prob = toProb(a_cnt)
    b_prob = toProb(b_cnt)
    ab_prob = toProb(joint_cnt)

    r = {}
    for (a,b), p_ab in ab_prob.items():
        p_a = a_cnt[a]
        p_b = b_cnt[b]

        pmi = math.log( p_ab / (p_a * p_b) )
        npmi = pmi / -math.log(p_ab)
        r[(a,b)] = npmi
    return r

def dual_npmi(stream):

    a_cnt = Counter()
    b_cnt = Counter()
    a_given_b = defaultdict(Counter)
    joint_cnt = Counter()

    n = 0.0
    get_p = lambda d: dict([ (k, v/n) for (k,v) in d.items() ])

    info = {}
    for a_set, b_set in stream:

        n += 1.0
        for b in b_set:
            a_given_b[b].update(a_set)
        a_cnt.update( a_set )
        b_cnt.update( b_set )
        joint_cnt.update( product(a_set, b_set) )
        
    a_prob = get_p(a_cnt)
    b_prob = get_p(b_cnt)
    ab_prob = get_p(joint_cnt)

    for (a,b), p_ab in ab_prob.items():
        p_a = a_prob[a]
        p_b = b_prob[b]

        pmi = math.log( p_ab / (p_a * p_b) )
        npmi = pmi / -math.log( p_ab )

        info[ (a,b) ] = npmi

    return info

def variable_npmi(stream):
    obs_cnt = Counter()
    joint_cnt = Counter()

    n = 0.0
    get_p = lambda d: dict([ (k, v/n) for (k,v) in d.items() ])

    info = {}

    for obs in stream:
        n += 1
        obs_cnt.update(obs)
        joint_cnt.update( product(obs, obs) )
    
    p_obs = get_p(obs_cnt)
    p_joint = get_p(joint_cnt)

    for (a,b), p_ab in p_joint.items():
        p_a = p_obs[a]
        p_b = p_obs[b]

        pmi = math.log( p_ab / (p_a * p_b) )
        npmi = pmi / -math.log( p_ab )
        
        info[ (a,b) ] = npmi
    
    return info

def segmentEntropy(segments, normalize=True):

    if normalize and not (normalize is True):
        total = normalize
        gap = total - sum(segments)
        if gap > 0:
            p = gap/total
            ent = -(p*math.log(p))
        else:
            ent = 0.0
    else:
        total = sum(segments)
        ent = 0.0
        
    for x in segments:
        if x > 0:
            p = x/total
            ent -= p * math.log(p)
    
    if normalize is True:
        ent /= math.log(total)
    elif normalize:
        ent /= math.log(normalize)
    
    return ent


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def get_dyn_imports(d):
    
    if isinstance(d, dict):
        nd = {}
        for k,v in d.items():
            nd[ get_dyn_imports(k) ] = get_dyn_imports(v)
        return nd
    elif isinstance(d, list):
        return list(map(get_dyn_imports, d))
    else:
        if isinstance(d, str) and d.count(".") > 0:
            try:
                p = d.split(".")
                pClass = p[-1]
                pc = ".".join(p[:-1])
                return getattr(import_module(pc), pClass)
            except Exception as e:
                print(e)
                return d
    
    return d

def build(cdef, *opts):
    from seamr.core import construct
    
    d = {}
    for o in opts: 
        d.update(o)

    Type,Args = list(cdef.items())[0]
    return construct(Type, Args, d)
    
def load_config(*files):
    conf = {}
    
    for filename in files:
        if isinstance(filename, dict):
            conf.update( filename )
        else:
            if not filename.endswith(".yaml"):
                yfile = filename + ".yaml"
            else:
                yfile = filename

            if os.path.isfile(yfile):
                with open(yfile) as cf:
                    conf.update( yaml.load(cf) )

            else:
                try:
                    conf.update( yaml.load(filename) )
                except ValueError:
                    print("Tried to load %s as a literal string, missing file?" % filename)
                    raise

    return get_dyn_imports(conf)

# -------------------------------------------------------
class NameGen:
    # originally from: https://gist.github.com/1266756
    # with some changes
    # example output:
    # "falling-late-violet-forest-d27b3"
    adjs = [ "autumn", "hidden", "bitter", "misty", "silent", "empty", "dry", "dark",
          "summer", "icy", "quiet", "white", "cool", "spring", "winter",
          "patient", "twilight", "dawn", "crimson", "wispy", "weathered", "blue",
          "broken", "cold", "damp", "falling", "frosty", "green",
          "long", "late", "bold", "little", "morning", "muddy", "old",
          "red", "rough", "still", "small", "sparkling", "shy",
          "wandering", "withered", "wild", "black", "young", "holy", "solitary",
          "fragrant", "aged", "snowy", "proud", "floral", "restless", "divine",
          "polished", "ancient", "purple", "lively", "nameless"
      ]
    nouns = [ "waterfall", "river", "breeze", "moon", "rain", "wind", "sea", "morning",
          "snow", "lake", "sunset", "pine", "shadow", "leaf", "dawn", "glitter",
          "forest", "hill", "cloud", "meadow", "sun", "glade", "bird", "brook",
          "butterfly", "bush", "dew", "dust", "field", "fire", "flower", "firefly",
          "feather", "grass", "haze", "mountain", "night", "pond", "darkness",
          "snowflake", "silence", "sound", "sky", "shape", "surf", "thunder",
          "violet", "water", "wildflower", "wave", "water", "resonance", "sun",
          "wood", "dream", "cherry", "ninja", "turtle", "tree", "fog", "frost", "voice", "paper",
          "frog", "smoke", "star"
      ]

    def __len__(self):
        return max(map(len, self.adjs)) + 1 + max(map(len, self.nouns))

    def __call__(self):
        return random.choice(self.adjs) + "-" + random.choice(self.nouns)

haiku = NameGen()

class FlexLogger(object):

    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    FUCSIA = '\033[95m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    UNDERLINE = '\033[4m'

    def __init__(self, output=sys.stdout):
        self.levels = set(["info","warn","error","lps","stats"])
        self._log = logging.getLogger(__name__)
        self.output = output
        self._pid_name = {}
        self.start = time.monotonic()
        self.colors = [
            self.RED, self.GREEN, self.YELLOW, self.BLUE, self.FUCSIA, self.CYAN
        ]

        self.name_color = {
            "INFO" : self.YELLOW,
            "WARN" : self.FUCSIA,
            "DEBUG": self.CYAN,
            "ERROR": self.RED,
            "LPS"  : self.GREEN
        }

    def red(self, s):
        return "%s%s%s" % (self.RED, s, self.ENDC)

    def green(self, s):
        return "%s%s%s" % (self.GREEN, s, self.ENDC)

    def yellow(self, s):
        return "%s%s%s" % (self.YELLOW, s, self.ENDC)

    def blue(self, s):
        return "%s%s%s" % (self.BLUE, s, self.ENDC)

    def fucsia(self, s):
        return "%s%s%s" % (self.FUCSIA, s, self.ENDC)

    def cyan(self, s):
        return "%s%s%s" % (self.CYAN, s, self.ENDC)

    def get_name(self):
        pid = os.getpid()
        if pid not in self._pid_name:
            clr = random.choice(self.colors)
            self._pid_name[ pid ] = clr + haiku() + self.ENDC
        return self._pid_name[ pid ]

    def _null(self, *args, **kw):
        return self

    def _writer(self, prefix, msg, *args, **kwargs):
        
        msg = msg % tuple(args)
        
        if self.output in [sys.stdout, sys.stderr]:
            self.output.write("%s%0.2f%s | " % (self.CYAN, time.monotonic() - self.start, self.ENDC))
            self.output.write(prefix)
            
            if "color" in kwargs:
                self.output.write(kwargs["color"])

            self.output.write(msg)
            self.output.write(self.ENDC)
        else:
            self.output.write(msg)

        self.output.write("\n")

        return self

    def __getattr__(self, name):

        if name in self.levels:

            NAME = name.upper()
            prefix = "%s %s%s%s | " % (
                                    self.get_name(),
                                    self.name_color.get(NAME, self.GREEN), 
                                    name.upper(),
                                    self.ENDC)

            func = lambda *args, **kw: self._writer(prefix, *args, **kw)
            setattr(self, name, func)
            return func
        else:
            return self._null

    def set_levels(self, ok_levels):
        self.levels = set([ok_levels] if isinstance(ok_levels, basestring) else ok_levels)
        return self
    
    def enable(self, *levels):
        for l in levels:
            self.levels.add(l)
    
    def disable(self, *levels):
        for l in levels:
            self.levels.discard(l)

    def lps_stream(self, s, msg, total=None, output=sys.stdout, lps=1500):
        n = 0
        started = time.monotonic()
        line = 0

        for l in s:
            n += 1
            line += 1

            if n > lps:
                lps = float(n) / (time.monotonic() - started)
                lps_msg = "%s: %s/sec" % (msg, round(lps, 3),)
                
                if total:
                    lps_end = "(%0.2f%%%% done | eta %0.2f seconds)" % (100*float(line)/total, (total-line)/lps )

                else:
                    lps_end = "(%d lines)" % line

                self.lps(lps_msg + " " + lps_end, color=log.YELLOW)
                n = 0
                lps *= 3
                started = time.monotonic()

            yield l
        
        self.lps("%s done | %s items total" % (msg, line))

log = FlexLogger()

# -------------------------------------------------------

_log_depth = 0

class BlockWrapper():
    def __init__(self, msg, logger = None):
        log.levels.add("wrapper")
        self.msg = msg
        self.started = 0

    def __enter__(self):
        global _log_depth
        _log_depth += 1
        self.msg = (" " * _log_depth) + self.msg
        self.started = time.monotonic()
        log.wrapper("%s | started %s" % (self.msg,
                                        datetime.fromtimestamp(
                                            time.time()
                                        ).strftime('%Y-%m-%d %H:%M:%S')))
        return self

    def __exit__(self, type, value, traceback):
        global _log_depth
        _log_depth -= 1
        log.wrapper("%s | Done in %0.4f seconds" % (self.msg, time.monotonic() - self.started ))
