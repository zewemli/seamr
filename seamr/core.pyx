from libcpp.vector cimport vector
from libcpp.deque cimport deque
from libcpp cimport bool

from tqdm import tqdm
import json
import hashlib
import os
import sys
import pickle
import numpy as np
import math
import seamr
from seamr import sem
from random import sample
from datetime import datetime
from itertools import chain, groupby, product
from collections import defaultdict, Counter
from collections import deque as PyDeque

from cython.operator cimport dereference as deref, preincrement as inc

import sys, traceback
import inspect

import cython

cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport floor, fabs
from libc.math cimport pow as cPow

import re

cdef double epsilon = 0.0000001

cdef double clamp(double a, double low, double high):
    return min(high, max(a, low))

cpdef double ldist(np.ndarray[np.float64_t, ndim=1] A, np.ndarray[np.float64_t, ndim=1] B, double f):
    cdef int i
    cdef int size = A.size
    cdef double d = 0.0
    cdef double[:] vA = A
    cdef double[:] vB = B
   
    for i in range(size):
        d += cPow( fabs(vA[i] - vB[i]), f )
    
    return cPow(d, 1.0 / f)

cpdef double ldist_05(np.ndarray[np.float64_t, ndim=1] A, np.ndarray[np.float64_t, ndim=1] B):
    return ldist(A,B,0.5)

cpdef double ldist_01(np.ndarray[np.float64_t, ndim=1] A, np.ndarray[np.float64_t, ndim=1] B):
    return ldist(A,B,0.1)

cdef class OptimalF1:
    cdef np.ndarray reals
    cdef np.ndarray preds
    cdef np.ndarray order

    cdef double max_freq
    cdef double npos
    cdef double prob_pos
    cdef double pmax
    cdef double pstep
    cdef int n
    cdef public double bestF1
    cdef public double bestCut
    cdef public list curve

    def __init__(self, np.ndarray reals, np.ndarray preds, float max_freq = 1.0):
        self.max_freq = max_freq
        self.reals = reals
        self.preds = preds
        self.n = reals.shape[0]
        self.order = np.argsort(preds).astype(np.int64)

        self.npos = reals.sum()
        self.prob_pos = reals.mean()
        self.curve = list()
        self.pmax = preds.max()
        self.pstep = self.pmax / 20.0

        self.set_curve()
    
    cpdef set_curve(self):
        cdef double[:] reals = self.reals
        cdef double f1Real = 0.0

        cdef int i = 0
        cdef int n = 0

        self.bestF1, self.bestCut = self.get_optimalF1_cut( self.reals.size )

        f1Real = self.get_f1_at_cut( self.prob_pos )

        self.curve.append([ 0, f1Real, self.prob_pos, f1Real ])

        for i in range(1, len(self.reals)):
            if reals[i-1] == 1.0 and reals[i] == 0.0:
                n += 1
                f1Est, cutVal = self.get_optimalF1_cut( iMax = i )
                f1Real = self.get_f1_at_cut( cutVal )

                self.curve.append([ n, f1Est, cutVal, f1Real ])

    cpdef get_optimalF1_cut(self, int iMax=0):
        cdef int i = 0
        cdef int row = 0

        cdef int i_start = int( self.n - (self.n * self.max_freq) )
        
        cdef bool is_true = False
        cdef bool pred_true = False

        cdef double fn = 0.0
        cdef double fp = self.reals.size - self.npos
        cdef double tp = self.npos
        
        cdef long[:] order = self.order
        
        cdef double[:] reals = self.reals
        cdef double[:] preds = self.preds

        cdef double optimalF1 = 0.0
        cdef double optimalCut = 0.0
        cdef double cutVal = 0.0
        cdef double currF1 = 0.0
        cdef double pcheck = 0.0
        cdef double denom = 0.0
        cdef double stepBy = self.pstep

        if iMax == 0:
            iMax = int(self.order.shape[0])

        for i in range(i_start):
            row = order[i]
            if row < iMax:
                if reals[row] > 0:
                    tp -= 1
                    fn += 1
                else:
                    fp -= 1

        if (tp + fp + fn) > 0:
            optimalF1 = (2 * tp) / ( (2*tp) + fp + fn )
            optimalCut = preds[ row ]

        for i in range( i_start, self.order.size ):
            row = order[i]
            if row < iMax:
                if reals[row] > 0:
                    tp -= 1
                    fn += 1
                else:
                    fp -= 1

                denom = ( (2*tp) + fp + fn )
                if denom > 0:
                    currF1 = (2 * tp) / denom
                    if currF1 > optimalF1:
                        optimalF1 = currF1
                        optimalCut = preds[ row ]
            else:
                break

        return optimalF1, optimalCut

    cpdef double get_f1_at_cut(self, double cut):
        cdef double[:] reals = self.reals
        cdef double[:] preds = self.preds

        cdef double tp = 0.0
        cdef double fp = 0.0
        cdef double fn = 0.0
        cdef bool is_true = False
        cdef bool pred_true = False

        cdef int i = 0
        for i in range( self.reals.shape[0] ):
            is_true = reals[i] > 0
            pred_true = preds[i] >= cut
            if is_true:
                if pred_true:
                    tp += 1.0
                else:
                    fn += 1
            elif pred_true:
                fp += 1

        return (2 * tp) / (2*tp + fp + fn)
        
def ensure(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def getClassArgs(Cls):

    ok_args = set()
    ok_args |= set(inspect.getfullargspec(Cls.__init__)[0])
    
    if hasattr(Cls, "__cinit__"):
        ok_args = set(inspect.getfullargspec(Cls.__cinit__)[0])

    if hasattr(Cls, "__bases__"):
        for subC in Cls.__bases__:
            if subC is not object:
                ok_args |= getClassArgs(subC)

    return ok_args

def makeStrDict(d):
    
    if isinstance(d, dict):
        nd = {}
        for k,v in d.items():
            nd[str(k)] = makeStrDict(v)
        return nd

    elif isinstance(d,list):
        return list(map(makeStrDict, d))
    
    elif isinstance(d, tuple):
        return tuple(map(makeStrDict, d))
    
    else:
        return str(d)

def getBuildArgs(Cls, decl_args, local_args):

    ok_args = getClassArgs(Cls) - set(["self"])

    args = {}
    args.update(decl_args)
    for k,p in decl_args.items():
        if p is None and k in local_args:
            args[k] = local_args[k]

    for k in ok_args:
        if decl_args.get(k, None) is not None:
            args[k] = decl_args.get(k, None)
        elif local_args.get(k, None) is not None:
            args[k] = local_args.get(k, None)
        
    if "uuid" in ok_args:
        argstr = json.dumps(makeStrDict(args), sort_keys=True)
        args['uuid'] = hashlib.md5(argstr.encode("UTF-8")).hexdigest()
        
    return args

def build(defDict, local_args = {}):
    Cls,decl_args = list(defDict.items())[0]
    return construct(Cls, decl_args, local_args)

def construct(Cls, decl_args, local_args):
    
    if isinstance(Cls,str):
        raise TypeError("Could not find a class named %s" % Cls)

    try:
        buildArgs = getBuildArgs(Cls, decl_args, local_args)
    
        while len(buildArgs):
            try:
                return Cls( **buildArgs )
            except TypeError as err:
                serr = str(err)
                
                if "unexpected keyword argument" in serr:
                    serr = serr.strip().split(" ")[-1].strip("'")
                    if serr:
                        del buildArgs[serr]
                    else:
                        raise
                else:
                    raise

        return Cls( **buildArgs )

    except TypeError as e:
        print("Exception in user code:")
        print("-"*60)
        traceback.print_exc(file=sys.stdout)
        print("-"*60)
        raise

def pickle_gen(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            try:
                while True:
                    yield pickle.load(f)
            except EOFError:
                pass

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mat_jaccard(np.ndarray A, np.ndarray B):
    cdef int i = 0
    cdef int j = 0
    cdef int t = 0
    cdef int n_rows = min(A.shape[0], B.shape[0])

    cdef double c_and = 0.0
    cdef double c_or = 0.0

    cdef np.ndarray mat = np.zeros( (A.shape[1], B.shape[1],) )
    cdef double[:,:] vM = mat
    cdef double[:,:] vA = A
    cdef double[:,:] vB = B
    
    for i in range(A.shape[1]):
        for j in range(B.shape[1]):
            c_and = 0.0
            c_or  = epsilon
            for t in range( n_rows ):
                c_and += min(vA[t,i], vB[t,j])
                c_or  += max(vA[t,i], vB[t,j])
            vM[i,j] = c_and / c_or

    return mat

cpdef np.ndarray set_times(np.ndarray mat, np.ndarray times):
    cdef double t
    cdef double sec
    cdef double hr
    cdef int i = 0
    cdef int r = mat.shape[0]

    cdef int i_sec = mat.shape[1] - 2
    cdef int i_hr = mat.shape[1] - 1

    assert len(times) == r

    while i < r:
        t = times[i]
        sec = t % 86400
        mat[i, i_sec] = sec / 86400.0
        mat[i, i_hr] = int(sec / 3600.0)
        i += 1

    return mat.astype(np.float64)

cpdef np.ndarray get_times_mat(np.ndarray times):
    mat = np.zeros((len(times), 2))
    return set_times(mat, times)

cpdef get_times(stream, double step = 0.0, int max_steps = 15):
    cdef Event e
    cdef double tm = 0.0
    cdef double latestTime = 0.0
    cdef vector[double] times = []
    cdef int n_steps = 0
    cdef int end = -1

    if step == 0:
        times = sorted(set([ e.timestamp for e in stream() ]))
    else:
        tm = 0.0
        for e in stream():
            if tm == 0.0:
                tm = e.timestamp
            
            n_steps = int((e.timestamp - tm) // step)
            
            if n_steps < max_steps:
                while n_steps > 0:
                    tm += step
                    n_steps -= 1
                    times.push_back(tm)
            
            else:
                tm = ((e.timestamp // step) + 1) * step
                times.push_back( tm )
            
    return times

cdef class ID_Index:
    cdef readonly dict _index
    cdef readonly int idx

    def __cinit__(self):
        self.idx = 0

    def __init__(self):
        self._index = {}

    def __contains__(self, key):
        return key in self._index

    def __call__(self, key):
        return self.get(key)
    
    cpdef get(self, key):
        if key in self._index:
            return self._index.get(key)
        else:
            self._index[key] = self.idx
            self.idx += 1
            return self.idx-1
            

    def __len__(self):
        return len(self._index)
    
    def invert(self):
        return dict([ (k,v) for (v,k) in self._index.items() ])

    def apply(self, to_iter):
        return [ self(x) for x in to_iter ]

class PeekQ:
    def __init__(self, src):
        self.src = iter(src)
        self.done = False

        try:
            self.front = next(src)
        except:
            self.done = True
            self.front = None
    
    def step(self):
        if self.done:
            raise StopIteration()
        else:
            f = self.front

            try:
                self.front = next(self.src)
            except StopIteration:
                self.front = None
                self.done = True
            
            return f

    def __iter__(self):
        return self

    def __next__(self):
        return self.step()

def gen_event_blocks(stream, times):
    """
    Generate batches of events to match the times
    """
    times = iter(times)
    t = next(times)
    events = []
    for e in stream():

        while e.timestamp > t:
            yield events, t
            events = []
            t = next(times)
        
        events.append(e)
    
    yield events, t

    for t in times:
        yield [], t

def gen_active_sensors(stream, times):
    
    tm = {}
    state = {}
    
    tPrev = times[0] - (times[1] - times[0])
    
    for events, t in gen_event_blocks(stream, times):
        vals = Counter()

        for e in events:
            ePrev = max(tm.get(e.sensorID, tPrev), tPrev)
            vals[ e.sensorID ] += state.get(e.sensorID, 0.0) * ( e.timestamp - ePrev )

            state[ e.sensorID ] = e.state
            tm[ e.sensorID ] = e.timestamp

        tDur = t - tPrev

        for eID,eTM in tm.items():
            if eTM < tPrev and state.get(eID, 0.0) > 0.0:
                vals[ eID ] = state[eID] * tDur

        yield { s : st/tDur for s,st in vals.items() }, t
        tPrev = t


def get_schedule(store, dset, k, eval_for, validate_for,
                 eval_mode, inverse=False, step = 0.0, max_steps = 15):
    
    stream = lambda: store.get_events_lps(dset, days=list(range(eval_for + validate_for)), msg="Getting schedule")
    
    all_times = get_times(stream, step = step, max_steps = max_steps)

    doffset = int(all_times[0] // 86400)

    times_by_day = [ (d,list(tm)) for d,tm in groupby(all_times, key=lambda t: int(t // 86400) - doffset) ]
    
    folds = []

    eval_days = [ d for d,_ in times_by_day[:eval_for] ]
    eval_times = [ ev for _,ev in times_by_day[:eval_for] ]

    validation_days = [ d for d,_ in times_by_day[eval_for:] ]
    validation_times = list(chain(*[ev for _,ev in times_by_day[eval_for:]]))

    if eval_mode == "k-blocked":
        step_days = (len(eval_days) + (k-1)) // k
        for i in range(k):
            train_days = [ x for j,x in enumerate(eval_days) if (j // step_days) != i ]
            test_days  = [ x for j,x in enumerate(eval_days) if (j // step_days) == i ]

            train_times = list(chain(*[ tm for j,tm in enumerate(eval_times) if (j // step_days) != i ]))
            test_times = list(chain(*[ tm for j,tm in enumerate(eval_times) if (j // step_days) == i ]))

            if len(test_times):
                x_times = [ train_times, test_times, validation_times ]
                x_days = [ train_days, test_days, validation_days ]
                folds.append( (x_times, x_days) )
    
    elif eval_mode == "one-day":
        for i in range(k):
            train_days = [ x for j,x in enumerate(eval_days) if (j % k) != i ]
            test_days  = [ x for j,x in enumerate(eval_days) if (j % k) == i ]

            train_times = list(chain(*[ tm for j,tm in enumerate(eval_times) if (j % k) != i ]))
            test_times = list(chain(*[ tm for j,tm in enumerate(eval_times) if (j % k) == i ]))
            
            if len(test_times):
                x_times = [ train_times, test_times, validation_times ]
                x_days = [ train_days, test_days, validation_days ]
                folds.append( (x_times, x_days) )
    
    elif eval_mode == "round-robin":
        eval_times = list(chain(*eval_times))
        
        for i in range(k):
            train_times = [ t for j,t in enumerate(eval_times) if (j % k) != i ]
            test_times  = [ t for j,t in enumerate(eval_times) if (j % k) == i ]

            folds.append(([ train_times, test_times, validation_times ],
                          [eval_days, eval_days, validation_days],) )
    
    else:
        raise NotImplementedError("eval_mode '%s' not available use k-blocked one-day or round-robin" % eval_mode)
    
    if inverse:
        folds = [ ((test_t, train_t, val_t), (test_d, train_d, val_d)) 
                    for ((train_t, test_t, val_t), (train_d, test_d, val_d)) in folds ]
    
    c = lambda s,clr: "{}{}{}".format( getattr(seamr.log,clr.upper()), s, seamr.log.ENDC )
    b = lambda s: '\033[1m{}\033[0m'.format(s)

    for fold, ((_et,_tt,_vt), (train_days, test_days, val_days)) in enumerate(folds):
        if val_days:
            val = "| Validating: %s - %s" % (c(min(val_days),"FUCSIA"), c(max(val_days),"FUCSIA"))
        else:
            val = ""
        
        bd = sorted(set(train_days) | set(test_days))
        clrs = []
        for d in bd:
            if d in train_days:
                clrs.append( c(d, "blue") )
            else:
                assert d in test_days
                clrs.append( b(c(d, "yellow")) )

        print( "{} : {} | {}{}".format(c("Train","blue"), b(c("Test","yellow")), " | ".join(clrs), val ) )
    
    return folds

cdef class Dataset:
    cdef readonly int id
    cdef readonly unicode name
    cdef readonly unicode env
    cdef readonly int n_res

    def __init__(self, int id, unicode name, unicode env, int n_res):
        self.id = id
        self.name = name
        self.env = env
        self.n_res = n_res
    
    def __getstate__(self):
        return (self.id,
                self.name,
                self.env,
                self.n_res)

    def __setstate__(self, state):
        (self.id,
        self.name,
        self.env,
        self.n_res) = state

cdef class Sensor:
    cdef readonly int id
    cdef readonly unicode name
    cdef readonly unicode kind
    
    def __init__(self, int lid, unicode name, unicode kind):
        self.id = lid
        self.name = name
        self.kind = kind
    
    cpdef setKind(self, unicode to):
        self.kind = to

    def __getstate__(self):
        return (self.id, self.name, self.kind,)
    
    def __setstate__(self, state):
        self.id, self.name, self.kind = state

cdef class SensorState:
    
    def __cinit__(self, double decay):
        self.times = new deque[double]()
        self.decay = decay
        self.state = 0.0
        self.mag = 0.0
        self.newestTime = 0.0
        self.isBinary = True
    
    def __del__(self):
        free(self.times)

    cpdef void reset(self):
        self.trimLength(0)
        self.state = 0.0
        self.mag = 0.0
        self.newestTime = 0.0
        # Don't reset isBinary

    cpdef void update(self, double atTime, double state):
        cdef double diff = 0.0
        self.isBinary &= (state == 0.0) | (state == 1.0)
        self.times.push_back( atTime )
        self.newestTime = atTime
        
        if self.isBinary:
            self.state = state
        else:
            diff = state - self.state
            self.state = self.state * (1-self.decay) + state * self.decay
            self.mag   = self.mag   * (1-self.decay) + diff  * self.decay

    cpdef int trimLength(self, int toLen)  except -1:
        cdef int n = 0
        while self.times.size() > toLen:
            self.times.pop_front()
            n += 1
        return n

    cpdef int trimSize(self, double minTime) except -1:
        cdef int n = 0
        while self.times.size() > 0 and self.times.front() < minTime:
            self.times.pop_front()
            n += 1
        return n

    cpdef int trim(self, int toLen, double minTime) except -1:
        if toLen > 0:
            return self.trimLength(toLen)
        else:
            return self.trimSize(minTime)

    cpdef double getState(self, double atTime):
        return self.state

    cpdef double getCount(self, double atTime):
        return self.times.size()

    cpdef double getSpan(self, double atTime):
        if self.times.size() > 0:
            return (self.times.back() - self.times.front()) / atTime - self.times.front()
        else:
            return 0.0

    cpdef double getDelta(self, double atTime):
        return 1.0 - clamp((atTime - self.newestTime) / 86400.0, 0.0, 1.0)

    cpdef double getDutyCycle(self, double atTime):

        cdef double pt = 0.0
        cdef double t = 0.0
        cdef double tOn = 0.0
        cdef double tOff = 0.0
        
        cdef deque[double].iterator it
        it = self.times.begin()
        
        if self.times.size() > 0:
            pt = deref(it)
            
            while it != self.times.end():
                t = deref(it)
                tOn += t - pt
                pt = t
                it = inc(it)

                t = tOn
                tOn = tOff
                tOff = t
            
            tOn += atTime - pt
            t = tOn + tOff
            if t > 0:
                if self.state > 0.0:
                    return tOn / t
                else:
                    return tOff / t
                    
        return 0.0
    
    cpdef np.ndarray getVector(self, double atTime):
        cdef np.ndarray[dtype=np.float64_t, ndim=1] v

        if self.isBinary:
            v = np.zeros( 5 )
            v[0] = self.getState(atTime)
            v[1] = self.getCount(atTime)
            v[2] = self.getSpan(atTime)
            v[3] = self.getDelta(atTime)
            v[4] = self.getDutyCycle(atTime)
        
        else:
            v = np.zeros(2)
            v[0] = self.state
            v[1] = self.mag
        
        return v

cdef class Activity:
    cdef readonly int id
    cdef readonly unicode name
    
    def __init__(self, int id, unicode name):
        self.id = id
        self.name = name

    def __getstate__(self):
        return (self.id, self.name,)
    
    def __setstate__(self, state):
        self.id, self.name = state

cdef class Event:

    def __init__(self,
                 int srcLine,
                 int sensor,
                 double state, 
                 int day,
                 double sec_of_day,
                 double timestamp):
        
        self.srcLine = srcLine
        self.sensorID = sensor
        self.day = day
        self.state = state
        self.sec_of_day = sec_of_day
        self.timestamp = timestamp

    def __getstate__(self):
        return (self.srcLine,
                self.sensorID,
                self.state,
                self.day,
                self.sec_of_day,
                self.timestamp)
    
    def __setstate__(self, state):
        (self.srcLine,
        self.sensorID,
        self.state,
        self.day,
        self.sec_of_day,
        self.timestamp) = state
    
cdef class Label:
    cdef readonly int line
    cdef public int actID
    cdef readonly int resID
    cdef readonly unicode resident
    cdef public unicode activity
    cdef readonly int day
    cdef readonly double start
    cdef readonly double finish
    
    def __init__(self, int line,
                       int actID,
                       int resID,
                       unicode resident,
                       unicode activity,
                       int day,
                       double start,
                       double finish):
        
        self.line = line
        self.actID = actID
        self.resID = resID
        self.resident = resident
        self.activity = activity
        self.day = day
        self.start = start
        self.finish = finish
    
    def makeSem(self, mapping, index):
        self.activity = mapping[ self.activity ]
        self.actID = index[self.activity]

    def __getstate__(self):
        return (self.line,
                self.actID,
                self.resID,
                self.resident,
                self.activity,
                self.day,
                self.start,
                self.finish)
    
    def __setstate__(self, state):
        (self.line,
         self.actID,
         self.resID,
         self.resident,
         self.activity,
         self.day,
         self.start,
         self.finish) = state


cdef class LabelIndex:
    cdef public dict residents
    cdef public list labels
    cdef public dict names
    cdef public dict lblName
    cdef public int on_index
    cdef Label default
    cdef int max_id

    def __cinit__(self, int default, dict lblName, list labels):
        self.on_index = 0
        self.lblName = lblName
        self.labels = list()
        self.residents = dict()
        self.max_id = 0
        
        pl = None
        for l in sorted(labels, key=lambda x: x.start):
            self.max_id = max(self.max_id, l.actID)
            self.residents[l.resID] = l.resident
            
            assert pl is None or pl.start <= l.start, (pl is None, pl.start, l.start)
            assert self.lblName[l.actID] == l.activity, (l.actID, self.lblName[l.actID], l.activity,)
            
            pl = l
            self.labels.append(l)

        self.default = Label(0, default, 0, "resident", self.lblName[default], 0, 0.0, 0.0)
            
    def residentSubset(self, res):
        res_labels = [l for l in self.labels if l.resident == res]
        return LabelIndex(self.default.actID, self.lblName, res_labels)

    def getResSubsets(self):
        return [self.residentSubset(rname) for rname in self.resNames()] 

    def resNames(self):
        return [rname for rid,rname in sorted(self.residents.items())]

    cpdef int nRes(self):
        return len(self.residents)

    cpdef reset(self):
        self.on_index = 0
        return self

    cpdef object getDefault(self):
        return self.default

    cpdef Label getAt(self, double tm):
        while self.on_index < len(self.labels) and self.labels[self.on_index].finish < tm:
            self.on_index += 1

        if self.on_index >= len(self.labels) or tm < self.labels[self.on_index].start:
            return self.default
        else:
            return self.labels[self.on_index]

    cpdef int numLabel(self, double tm):
        lbl = self.getAt(tm)
        if lbl:
            return lbl.actID
        else:
            return self.default

    cpdef strLabel(self, double tm):
        cdef Label lbl = self.getAt(tm)

        if lbl:
            return lbl.activity
        else:
            return self.defaultStr
    
    def make_dataset(self, times, one_hot = False, as_int = False, prefer_mat = False):
        self.reset()

        if self.nRes() > 1 or prefer_mat:

            return [ self.residentSubset(self.residents[res])\
                         .make_dataset(times, one_hot=one_hot, as_int=as_int)
                        for res in sorted(self.residents.keys()) ]
        
        else:
            
            if one_hot:
                r = np.zeros((len(times), self.max_id+1))
                for i,t in enumerate(times):
                    r[i, self.numLabel(t)] = 1
                return [ r ]

            elif as_int:
                return list(map(self.numLabel, times))
            else:
                return list(map(self.strLabel, times))
    
    def get_dense_mat(self, times):
        resLabels = self.make_dataset(times, prefer_mat=True)
        labels = set()
        for res in resLabels:
            labels |= set(res)
        classes = sorted(labels)
        ci = {c:i for i,c in enumerate(classes)}

        mat = np.zeros( (len(times), len(classes)) )

        for res in resLabels:
            for i,l in enumerate(res):
                mat[i, ci[l]] = 1.0
        
        return classes, mat


cdef class Features:

    cpdef int partitionCount(self):
        return 0

    def width(self):
        return 0
    
    cpdef list gen_batches(self, stream, times):

        cdef double t
        cdef Event e
        cdef list batch = []
        cdef list allBatches = []
        tIter = iter(times)

        t = next(tIter)

        try:
            for e in stream:
                if e.timestamp > t:
                    allBatches.append((batch, t))
                    batch = [e]
                    t = next(tIter)
                else:
                    batch.append(e)
            
            allBatches.append((batch, t))
            
            for t in tIter:
                allBatches.append(([], t))
        except StopIteration:
            pass
            
        return allBatches

    def make_dataset(self, event_stream, times):

        cdef np.ndarray mat
        cdef int time_w = self.time_width()
        cdef int m_w = 0

        cdef np.ndarray np_times = np.array(times)

        with tqdm(desc="making dataset with %s" % self.__class__.__name__,
                    total=len(times), leave=False) as progress:

            pi = 0
            
            if isinstance(self.width(), list):

                i = 0
                
                X = [ np.zeros((len(times), w.shape[-1])) for w in self.fetch( times[i] ) ]
                
                for e in event_stream():
                    while (i < len(times)) and (times[i] < e.timestamp):
                        for xi,vi in zip(X, self.fetch( times[i] )):
                            xi[i,:] = vi 
                        i += 1
                        if i % 10 == 0:
                            progress.update(i - pi)
                            pi = i
                    
                    self.update(e.srcLine, e.sensorID, e.timestamp, e.state)
                    
                while i < len(times):
                    for xi,vi in zip(X, self.fetch( times[i] )):
                        xi[i,:] = vi
                    i += 1
                    progress.update(1)
                
                if self.stack:
                    w_total = sum([ m.shape[1] - time_w for m in X ]) + time_w
                    mat = np.zeros(( X[0].shape[0], w_total ))
                    col = 0
                    for m in X:
                        assert m.shape[0] == mat.shape[0], ("Stacking matrices of sizes",
                                                            m.shape[0], mat.shape[0])

                        m_w = m.shape[1] - time_w
                        mat[:, col : col + m_w] = m[:,0:m_w]
                        col += m_w
                    
                    mat[ :, -time_w:] = X[-1][:, -time_w:]

                    return set_times(mat, np_times)
                else:
                    return [ set_times(m, np_times) for m in X ]

            else: # Just a single vector
            
                X = np.zeros((len(times), self.width()))
                
                i=0
                for e in event_stream():
                        
                    while (i < X.shape[0]) and (times[i] < e.timestamp):
                        X[i,:] = self.fetch( times[i] )
                        i += 1
                        if i % 10 == 0:
                            progress.update(i - pi)
                            pi = i
                    
                    self.update(e.srcLine, e.sensorID, e.timestamp, e.state)
            
                while i < X.shape[0]:
                    X[i,:] = self.fetch( times[i] )
                    i += 1
                    progress.update(1)
                
                X[~np.isfinite(X)] = 0.0
                X[ X < 0 ] = 0.0
                
                return set_times(X, np_times)

    @staticmethod
    def zip_events_and_times(events, times):
        i=0
        for e in events():
            while i < len(times) and times[i] < e.timestamp:
                yield times[i]
                i += 1

            yield e
        
        while i < len(times):
            yield times[i]
            i += 1

    cpdef void reset(self):
        pass

    def fit(self, labels, stream, times, debounce=None):
        pass

    cdef int time_width(self):
        return 2

cdef class MatrixStore:

    def __init__(self, store, dataset):     
        self.store = store

cdef class Store:
    cdef readonly list dataset
    cdef readonly unicode basedir
    cdef readonly bool use_sem_labels
    
    def __init__(self, unicode basedir, bool use_sem_labels = False):
        self.basedir = basedir
        self.use_sem_labels = use_sem_labels

        self.dataset = []
        for D in pickle_gen( self.path("dataset") ):
            self.dataset.append(D)

    def datasets(self):
        dirs = os.listdir(self.basedir)
        flt = lambda m: os.path.exists(os.path.join(self.basedir, m, "0000", "labels"))
        return list(filter(flt, dirs))

    def availableDays(self, dset):
        imatch = re.compile("^\d+$")
        j = os.path.join(self.basedir, dset)
        return list(map(int, filter(lambda k: imatch.match(k), os.listdir(j))))
        
    def daysWith(self, dset, objKind):
        allDays = self.availableDays(dset)
        okDays = []
        for d in allDays:
            if self.path(self.numstr(d), objKind):
                okDays.append(d)
        return okDays

    def numstr(self, n):
        return "%04d" % n

    def path(self, *args):
        return os.path.join(self.basedir, *args)

    def get_sensors(self, dset):
        cose = sem.COSE()
        sensors = list(pickle_gen( self.path(dset, "sensor") ))
        for s in sensors:
            s.setKind( cose.sensorType(s.name) )

        return sensors

    def get_acts(self, dset):
        return list(pickle_gen( self.path(dset, "activity") ))
    
    def store_events(self, events, dset_name):
        """
        Expects tuples like:
        (int sensor, double state, double timestamp)
        The sensor id should be the local id, not a global id
        """
        n_lines = 0
        d_ev = self.path(dset_name)
        ensure(d_ev)
        
        fp = None
        day = None
        day_offset = 0

        prevTime = 0.0

        for (line, sensor, state, tm) in events:
            if tm >= prevTime:
                prevTime = tm
                e_day = int(tm // 86400)

                if e_day != day:
                    if day is None:
                        day_offset = e_day

                    if fp is not None:
                        fp.close()

                    e_path = self.path(dset_name, self.numstr(e_day - day_offset))

                    fp = open(os.path.join(ensure(e_path), "events"), "ab")
                
                day = e_day

                pickle.dump(Event(
                    line,
                    sensor,
                    state,
                    int(e_day - day_offset),
                    tm % 86400,
                    tm
                ), fp)
                n_lines += 1
        return n_lines
        
    def store_labels(self, labels, dset_name):
        """
        Expects tuples like:
        int actID, int resID, unicode activity, unicode resident, int day, double start, double finish
        """
        from seamr import sem
        d_lb = self.path(dset_name)
        ensure(d_lb)
        
        n_lines = 0
        fp = None
        day = None
        day_offset = None

        prevTime = 0
        
        for (line, actID, resID, res, act, s, e) in labels:
            n_lines +=  1
            if s >= prevTime and act in sem.activity_type_map:
                sDay = int(s // 86400)
                eDay = int(e // 86400)
                
                for d in range(sDay, eDay + 1):
                    if d != day:
                        if day is None:
                            day_offset = d

                        day = d
                        d_path = self.path(dset_name, self.numstr(d - day_offset))
                        ensure(d_path)
                        if fp is not None:
                            fp.close()
                        fp = open(os.path.join(d_path, "labels"), "ab")

                    pickle.dump(Label(line, actID, resID, res, act, d, min(s,e), max(s,e)), fp)

        return n_lines

    def store_activities(self, acts, dset):
        """
        Expects tuples like:
        (int id, unicode name,)
        """
        ensure( self.path(dset) )
        with open(self.path(dset, "activity"), "wb") as f:
            for T in sorted(acts):
                pickle.dump(Activity(*T), f)
    
    def store_datasets(self, datasets):
        """
        Expects tuples like:
        (int id, unicode name, unicode env, int n_res)
        """
        ensure(self.basedir)
        with open(self.path("dataset"), "wb") as f:
            for T in sorted(datasets):
                pickle.dump(Dataset(*T), f)
    
    def store_sensors(self, sensors, dset, prefix=True):
        """
        Expects tuples like:
        (int lid, unicode name)
        """
        from seamr.sem import COSE
        cose = COSE()
        ensure(self.path(dset))

        env_name = dset.split(".")[0]

        with open(self.path(dset, "sensor"), "wb") as f:

            for sID, sName in sensors:
                if prefix:
                    full_name = "%s:%s" % (env_name, sName)
                    sType = cose.sensorType(sName)
                else:
                    full_name = sName
                    sType = 'concept'

                pickle.dump(Sensor(sID, full_name, sType), f)
    
    def _store_stream(self, kind, stream, dset, pday = -1):
        fp = None
        offset_day = 0
        day_dir = None

        for ts,mat in stream:
            day = int(ts // 86400)
            
            if day != pday:
                if pday < 0:
                    offset_day = day

                pday = day

                if fp is not None:
                    fp.close()

                day_dir = self.path(dset, self.numstr(day - offset_day))

                if not os.path.exists( day_dir ):
                    os.makedirs( day_dir )

                fp = open( os.path.join(day_dir, kind), "wb")

            pickle.dump((ts, mat,), fp, protocol = -1)
        
        if fp is not None:
            fp.close()
    
    def store_matrices(self, matrix_stream, dset, pday = -1):
        self._store_stream("matrices", matrix_stream, dset, pday=pday)

    def store_embeddings(self, embedding_stream, dset, pday = -1):
        self._store_stream("embeddings", embedding_stream, dset, pday=pday)

    def get_days(self, dset_name, start, end, days):
    
        if days:
            return sorted( set(days) & set(self.availableDays(dset_name)) )
        elif end:
            return [ d for d in self.availableDays(dset_name) if start <= d <= end ]
        else:
            return [ d for d in self.availableDays(dset_name) if start <= d ]

    def get_labels(self, dset_name, start=0, end = None, days=None, resident=None):
        from seamr.sem import activity_type_map

        days = list(self.get_days(dset_name, start, end, days))
        day_paths = [ self.path(dset_name, self.numstr(d), "labels") for d in days]
        labels = [ l for l in chain(*[ pickle_gen(d) for d in day_paths if os.path.exists(d) ])
                    if l.activity in activity_type_map ]

        useMapping = {}
        index = {}

        if self.use_sem_labels:
            
            useMapping = { a.name : activity_type_map[a.name].split(":")[-1].lower()
                            for a in self.get_acts(dset_name)
                            if a.name in activity_type_map }

            index = { c : i for i,c in enumerate(sorted(set(useMapping.values()))) }

            for l in labels:
                l.makeSem(useMapping, index)

            acts = [ Activity(i,c) for c,i in sorted(index.items(), key=lambda T: T[1]) ]
        else:
            acts = self.get_acts(dset_name)

        if resident:
            labels = [ l for l in labels if l.resident == resident ]

        default = [ a.id for a in acts if a.name == 'other' ][0]
        index = dict([ (a.id, a.name,) for a in acts ])
        return LabelIndex(default, index, labels)
        
    def get_events(self, dset_name, days=None):
        for d in (days or self.availableDays(dset_name)):
            yield from pickle_gen( self.path(dset_name, self.numstr(d), "events") )
    
    def get_events_lps(self, dset_name, msg="Reading", **kw):
        yield from tqdm(self.get_events(dset_name, **kw), desc="%s %s" % (msg, dset_name,), miniters=50, leave=False)

    def get_matrices(self, dset_name, days=None):
        for d in (days or self.availableDays(dset_name)):
            yield from pickle_gen( self.path(dset_name, self.numstr(d), "matrices") )

    def get_matrices_lps(self, dset_name, **kw):
        yield from tqdm(self.get_matrices(dset_name, **kw), desc="Matrices")

    def get_stream(self, kind, dset_name, days=None):
        for d in (days or self.availableDays(dset_name)):
            yield from pickle_gen( self.path(dset_name, self.numstr(d), kind) )

    def get_stream_lps(self, kind, dset_name, **kw):
        yield from tqdm(self.get_stream(kind, dset_name, **kw), desc = kind.capitalize())


#---------------------------------------------------------------------
# Core algorithms
#---------------------------------------------------------------------


def get_ent(vec, n):
    e = 0.0
    pe = (n - vec.sum()) / n

    e -= pe * math.log(pe)
    for x in vec.nditer():
        p = x/n
        e -= p * math.log(p)
    return e

def trimmer(q, window_len = 0, window_dur = 0.0):
    if window_len:
        while len(q) > window_len:
            yield q.popleft()
    else:
        while len(q) > 0 and (q[-1].timestamp - q[0].timestamp) > window_dur:
            yield q.popleft()

def get_npmi(mat, n, col_name):
    row_sum = mat.sum(axis=1) / n
    rows = np.flatnonzero( row_sum )

    col_sum = mat.sum(axis=0) / n
    cols = np.flatnonzero( col_sum )

    for r in rows:
        pr = row_sum[r]
        for c in cols:
            pc = col_sum[c]
            if pc > 0:
                p_rc = mat[r,c] / n
                
                if p_rc > 0:
                    rc_npmi = (math.log(pr * pc) / math.log(p_rc)) - 1.0
                else:
                    rc_npmi = -1.0

                yield r, col_name[c], rc_npmi

def get_sensor_overlap(nsensors, stream, times):

    state = np.zeros(nsensors)
    mat = np.zeros((len(times), nsensors))
    n = 0

    for e in stream():
        if n < len(times):
            while n < len(times) and times[n] < e.timestamp:
                mat[n,:] = state
                n += 1

            state[e.sensorID] = e.state

    while n < len(times) and times[n] < e.timestamp:
        mat[n,:] = state
        n += 1

    return mat_jaccard(mat, mat)

def get_stream_matrices(nsensors, labels, stream, times):

    if not isinstance(labels[0], list):
        labels = [labels]

    assert all([len(times) == len(ls) for ls in labels])

    acts = set()
    for l in labels:
        acts = acts | set(l)
    acts = sorted(acts)

    act_index = dict([ (s,i) for i,s in enumerate( acts ) ])

    state = np.zeros(nsensors)
    act_on = np.zeros((len(times), len(act_index)))
    sensor_on = np.zeros((len(times), nsensors))

    i = 0
    n = 0

    for e in stream():
        while i < len(times) and times[i] < e.timestamp:
            
            sensor_on[i,:] = state

            for lset in labels:
                l = lset[i]
                act_id = act_index[ l ]
                act_on[i, act_id] = 1

            i += 1

        if i >= len(times):
            break
        
        state[e.sensorID] = e.state
        
    return sensor_on, act_on, acts

def sensor_assoc(nsensors, stream, width=60):

    mat = np.zeros( (nsensors, nsensors) )
    q = PyDeque()
    cnt = Counter()

    for e in stream():
        q.append(e)
        cnt[e.sensorID] += 1
        while (q[-1].timestamp - q[0].timestamp) > width:
            pe = q.popleft()
            cnt[pe.sensorID] -= 1
        
        for n in cnt.keys():
            mat[e.sensorID, n] += 1
    
    return mat
