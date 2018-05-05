cimport cython
from seamr import core
from seamr.core cimport Features
from libcpp cimport bool
from libcpp.deque cimport deque

from libc.math cimport sin, cos, acos, exp, sqrt, fabs
from libc.math cimport fmax, fmin
cimport numpy as np
import numpy as np
import math

from cymem.cymem cimport Pool
cdef Pool mem = Pool()


cdef struct EventStruct:
    int eventLine
    int sensor
    double time
    double state

cdef class CASASFeatures(Features):

    cdef deque[EventStruct] queue
    cdef public int nFeats
    cdef readonly int nSensors
    cdef readonly int latest_sensor
    cdef readonly double latest_time
    cdef readonly int qlen
    cdef readonly int n
    cdef int prev_dom
    cdef int dom
    cdef bool is_first
    cdef np.ndarray mi
    cdef double * state
    cdef double * ptime
    
    cdef public list feat_names

    """
    /* Compute the feature vector for each window-size sequence of sensor events.
    fields
    0: time of the last sensor event in window (hour)
    1: time of the last sensor event in window (seconds)
    2: window size in time duration
    3: time since last sensor event
    4: dominant sensor for previous window
    5: dominant sensor two windows back
    6: last sensor event in window
    7: ** last location in window **
    8 - NumSensors+7: counts for each sensor
    NumSensors+8 - 2*NumSensors+7: time since sensor last fired (<= SECSINDAY)
    """

    def __cinit__(self, qlen, sensors, stack=False):
        self.stack = stack

        cdef int n = 1 + max(s.id for s in sensors)
        self.is_first = True
        self.n = n
        self.latest_sensor = 0

        self.mi    = np.ones( (n,n) )
        self.state = <double*> mem.alloc(n, sizeof(double))
        self.ptime = <double*> mem.alloc(n, sizeof(double))

        assert self.n > max([ s.id for s in sensors ])

        x_sensors = sorted(sensors, key=lambda s: s.id)

        self.feat_names = ["time-dur", "time-since", "dom-2","dom-1","dom","latest"]
        for s in x_sensors:
            self.feat_names.append("%s-state" % s.name)

        for s in x_sensors:
            self.feat_names.append("%s-delta" % s.name)
        
        self.feat_names += ["sec-of-day","hour"]

        self.nSensors = len(sensors)
        self.nFeats = len(self.feat_names)
        self.qlen = qlen
        self.dom = 0
        self.prev_dom = 0

        for i in range(n):
            self.state[i] = 0.0
            self.ptime[i] = 0.0

    def __str__(self):
        return "casas"

    cpdef void reset(self):
        for i in range(self.n):
            self.state[i] = 0.0
            self.ptime[i] = 0.0
        
        while self.queue.size() > 0:
            self.queue.pop_back()
        
        self.is_first = True

    def width(self): 
        return self.nFeats

    def fit(self, labels, stream, times):
        mat = np.zeros((self.n, self.n))

        prev = 0
        for ev in stream():
            mat[prev, ev.sensorID] += 1
            prev = ev.sensorID
        
        mat = mat / mat.sum(axis=0, keepdims=True)
        prob = mat.sum(axis=0) / mat.sum()

        for i in range(self.n):
            for j in range(self.n):
                self.mi[i,j] = 1.0
                p_ij = mat[i,j]
                if 1.0 > p_ij > 0:
                    p_val = (math.log(prob[i] * prob[j]) / math.log(p_ij)) - 1

                    if np.isfinite( p_val ):
                        self.mi[i,j] = p_val
        
    cdef double weigh(self, double delta, int sensor, int latest):
        return 1.0

    cdef double mi_weigh(self, double delta, int sensor, int latest) except? 0.0:
        return self.mi[ self.latest_sensor, sensor ]

    cdef double time_weigh(self, double delta, int sensor, int latest) except? 0.0:
        return exp(-0.125 * min(max(0, delta), 86400))

    cdef int index(self, row, col):
        return row * self.nSensors + col

    cdef void trimQ(self):
        while self.queue.size() > 0 and ( self.queue.front().eventLine - self.queue.back().eventLine ) > self.qlen:
            self.queue.pop_back()

    cpdef void update(self, int eLine, int sensor, double timestamp, double state):

        if self.is_first:
            self.is_first = False
            i=0
            while i < self.nSensors:
                self.ptime[i] = timestamp
                i += 1

        if sensor >= 0 and sensor < self.n:
            self.queue.push_front( EventStruct(eLine, sensor, timestamp, state) )
            self.state[sensor] = state
            self.ptime[sensor] = timestamp

            self.latest_sensor = sensor
            self.latest_time = timestamp

            self.trimQ()
            
    cpdef np.ndarray fetch(self, double timestamp):

        cdef np.ndarray np_vec = np.zeros(self.nFeats)

        cdef double[:] vec = np_vec
        cdef double delta
        cdef EventStruct e
        cdef int i = 0
        cdef int time_diff = 0
        cdef int idx = 0
        cdef double v = 0.0
        cdef double v_max = 0.0

        """
        No definition of location here...
        7: ** last location in window **        
        """
        
        if timestamp < self.latest_time:
            for i in range(self.n):
                self.ptime[i] = timestamp

            # In case the time order is off, this corrects by popping off the front
            while self.queue.size() > 0 and ((self.queue.front().time - self.queue.back().time) < 0):
                self.queue.pop_front()

        if self.queue.size() > 0:
            # 2: window size in time duration
            vec[0] = fmin(86400.0, fmax(0.0, self.queue.front().time - self.queue.back().time)) / 86400.0

            # 3: time since last sensor event
            vec[1] = max(0.0, timestamp - self.queue.front().time)
        else:
            vec[0] = 0.0
            vec[1] = 0.0

        # 5: dominant sensor two windows back
        vec[2] = self.prev_dom

        # 4: dominant sensor for previous window
        vec[3] = self.dom

        self.prev_dom = self.dom

        for e in self.queue:
            # 8 - NumSensors+7: counts for each sensor
            idx = 6 + e.sensor
            v = max(0.0, self.weigh(timestamp - e.time, e.sensor, self.latest_sensor))
            
            try:
                vec[ idx ] += v
                if vec[idx] > v_max:
                    v_max = vec[ idx ]
                    self.dom = e.sensor
            except:
                print("Error, sensor out of bounds s = %s [index %s] | n feats %s | n sensors %s |" % (e.sensor, idx, self.nFeats, self.n))

        # Dominant event in current window
        vec[4] = self.dom
        
        # 6: last sensor event in window
        vec[5] = self.latest_sensor

        # NumSensors+8 - 2*NumSensors+7: time since sensor last fired (<= SECSINDAY)
        for i in range(self.nSensors):
            idx = 6 + i + self.nSensors
            vec[ idx ] = fmin(86400.0, fmax(0.0, timestamp - self.ptime[i])) / 86400.0
            
        """
        These are set in the self.with_time() function
        0: time of the last sensor event in window (hour)
        1: time of the last sensor event in window (seconds)
        """
        return np_vec

cdef class CASASOneHotFeatures(Features):

    cdef deque[EventStruct] queue
    cdef public int nFeats
    cdef readonly int nSensors
    cdef readonly int latest_sensor
    cdef readonly double latest_time
    cdef readonly int qlen
    cdef readonly int n
    cdef int prev_dom
    cdef int dom
    cdef bool is_first
    cdef np.ndarray mi
    cdef np.ndarray miNorm
    cdef double * state
    cdef double * ptime
    cdef public list feat_names

    """
    /* Compute the feature vector for each window-size sequence of sensor events.
    fields
    0: time of the last sensor event in window (hour)
    1: time of the last sensor event in window (seconds)
    2: window size in time duration
    3: time since last sensor event
    4: dominant sensor for previous window
    5: dominant sensor two windows back
    6: last sensor event in window
    7: ** last location in window **
    8 - NumSensors+7: counts for each sensor
    NumSensors+8 - 2*NumSensors+7: time since sensor last fired (<= SECSINDAY)
    """
    def __init__(self, qlen, sensors, stack=False):
        pass

    def __cinit__(self, qlen, sensors, stack=False):
        self.stack = stack

        cdef int n = len(sensors)
        self.is_first = True
        self.n = n
        self.latest_sensor = 0

        self.mi    = np.ones((n,n))
        self.miNorm = np.ones((n,n))
        self.state = <double*> mem.alloc(n, sizeof(double))
        self.ptime = <double*> mem.alloc(n, sizeof(double))

        for i in range(n):
            self.state[i] = 0.0
            self.ptime[i] = 0.0

    def __init__(self, qlen = None, sensors = None):
        self.nSensors = len(sensors)
        self.nFeats = 2 + (self.nSensors * 6) + self.time_width()
        
        self.feat_names = ["window-size", "time-from-last"]
        self.feat_names.extend(["p2dom-%s" % s.name for s in sensors])
        self.feat_names.extend(["p1dom-%s" % s.name for s in sensors])
        self.feat_names.extend(["count-%s" % s.name for s in sensors])
        self.feat_names.extend(["dom-%s" % s.name for s in sensors])
        self.feat_names.extend(["last-sensor-%s" % s.name for s in sensors])
        self.feat_names.extend(["delta-%s" % s.name for s in sensors])
        self.feat_names.extend(["hr", "sec-of-day"])

        assert self.nFeats == len(self.feat_names)

        self.qlen = qlen
        self.dom = 0
        self.prev_dom = 0


    def __str__(self):
        return "casas-onehot"

    cpdef void reset(self):
        for i in range(self.n):
            self.state[i] = 0.0
            self.ptime[i] = 0.0
        
        while self.queue.size() > 0:
            self.queue.pop_back()
        
        self.is_first = True

    def width(self): 
        return self.nFeats

    def fit(self, labels, stream, times):
        mat = np.zeros((self.n, self.n))

        prev = 0
        for ev in stream():
            mat[prev, ev.sensorID] += 1
            prev = ev.sensorID
        
        mat = mat / mat.sum(axis=0, keepdims=True)
        prob = mat.sum(axis=0) / mat.sum()

        for i in range(self.n):
            for j in range(self.n):
                self.mi[i,j] = 0.0
                p_ij = mat[i,j]
                if 1.0 > p_ij > 0:
                    p_val = (math.log(prob[i] * prob[j]) / math.log(p_ij)) - 1

                    if np.isfinite( p_val ):
                        self.mi[i,j] = p_val
        
        self.miNorm = self.mi / (self.mi.sum(axis=1, keepdims=True) + 0.000001)
        for i in range(self.n):
            self.mi[i,i] = 1.0
            self.miNorm[i,i] = 1.0
        
    cdef double weigh(self, double delta, int sensor, int latest):
        return 1.0

    cdef double mi_weigh(self, double delta, int sensor, int latest) except 0:
        return self.mi[ self.latest_sensor, sensor ]

    cdef double time_weigh(self, double delta, int sensor, int latest):
        return exp(-0.125 * min(max(0, delta), 86400))

    cdef int index(self, row, col):
        return row * self.nSensors + col

    cdef void trimQ(self):
        while self.queue.size() > 0 and ( self.queue.front().eventLine - self.queue.back().eventLine ) > self.qlen:
            self.queue.pop_back()

    def update(self, int eLine, int sensor, double timestamp, double state):

        if sensor < self.n:

            if self.is_first:
                self.is_first = False
                i=0
                while i < self.nSensors:
                    self.ptime[i] = timestamp
                    i += 1

            self.queue.push_front( EventStruct(eLine, sensor, timestamp, state) )

            self.state[sensor] = state
            self.ptime[sensor] = timestamp

            self.latest_sensor = sensor
            self.latest_time = timestamp

            self.trimQ()

    @cython.boundscheck(False) 
    def fetch(self, double timestamp):

        cdef np.ndarray np_vec = np.zeros(self.nFeats)

        cdef double[:] vec = np_vec
        cdef double delta
        cdef EventStruct e
        cdef int i = 0
        cdef int time_diff = 0
        cdef int idx = 0
        cdef double v = 0.0
        cdef double v_max = 0.0

        """
        No definition of location here...
        7: ** last location in window **        
        """
        if timestamp < self.latest_time:
            for i in range(self.n):
                self.ptime[i] = timestamp

            # In case the time order is off, this corrects by popping off the front
            while self.queue.size() > 0 and ((self.queue.front().time - self.queue.back().time) < 0):
                self.queue.pop_front()

        if self.queue.size() > 0:
            # 2: window size in time duration
            vec[0] = fmin(86400.0, fmax(0.0, self.queue.front().time - self.queue.back().time)) / 86400.0

            # 3: time since last sensor event
            vec[1] = max(0.0, timestamp - self.queue.front().time)
        else:
            vec[0] = 0.0
            vec[1] = 0.0


        i = 2
        # 5: dominant sensor two windows back
        vec[i + self.prev_dom] = 1
        i += self.n

        # 4: dominant sensor for previous window
        vec[i + self.dom] = 1
        i += self.n

        self.prev_dom = self.dom

        for e in self.queue:
            # 8 - NumSensors+7: counts for each sensor
            idx = i + e.sensor
            v = max(0.0, self.weigh(timestamp - e.time, e.sensor, self.latest_sensor))
            vec[ idx ] += v
            
            if vec[idx] > v_max:
                v_max = vec[ idx ]
                self.dom = e.sensor
        i += self.n

        # Dominant event in current window
        vec[i + self.dom] = 1
        i += self.n
        
        # 6: last sensor event in window
        vec[i + self.latest_sensor] = 1.0
        i += self.n

        # NumSensors+8 - 2*NumSensors+7: time since sensor last fired (<= SECSINDAY)
        for idx in range(self.n):
            vec[ i + idx ] = min(86400.0, max(0.0, timestamp - self.ptime[idx])) / 86400.0
            
        """
        These are set in the self.with_time() function
        0: time of the last sensor event in window (hour)
        1: time of the last sensor event in window (seconds)
        """
        return np_vec


cdef class CASASTimeFeatures(CASASFeatures):
    def __str__(self):
        return "casas-time"
    cdef double weigh(self, double timestamp, int sensor, int latest):
        return self.time_weigh(timestamp, sensor, latest)

cdef class CASASTimeMiFeatures(CASASFeatures):
    def __str__(self):
        return "casas-mi-time"
    cdef double weigh(self, double timestamp, int sensor, int latest):
        return 0.5 *  (self.time_weigh(timestamp, sensor, latest)
                        + self.mi_weigh(timestamp, sensor, latest))

cdef class CASASMiFeatures(CASASFeatures):
    def __str__(self):
        return "casas-mi"
        
    cdef double weigh(self, double timestamp, int sensor, int latest):
        return self.mi_weigh(timestamp, sensor, latest)

cdef class SERCFeatures(CASASOneHotFeatures):

    cdef int clip


    def __cinit__(self, qlen, sensors, clip=1500, stack=False):
        self.clip = int(clip)

    def __str__(self):
        return "serc"

    @cython.boundscheck(False)
    def fetch(self, double timestamp):

        cdef np.ndarray np_vec = np.zeros(self.nFeats)

        cdef double[:] vec = np_vec
        cdef double[:,:] mi = self.miNorm
        cdef double delta
        cdef EventStruct e
        cdef int i = 0
        cdef int j = 0
        cdef int time_diff = 0
        cdef int idx = 0
        cdef double v = 0.0
        cdef double v_max = 0.0
        cdef np.ndarray[np.float64_t, ndim=1] cnt = np.zeros(self.nSensors, dtype=np.float64)

        """
        No definition of location here...
        7: ** last location in window **        
        """
        if timestamp < self.latest_time:
            for i in range(self.n):
                self.ptime[i] = timestamp

            # In case the time order is off, this corrects by popping off the front
            while self.queue.size() > 0 and ((self.queue.front().time - self.queue.back().time) < 0):
                self.queue.pop_front()

        if self.queue.size() > 0:
            # 2: window size in time duration
            vec[0] = min(86400.0, max(0.0, self.queue.front().time - self.queue.back().time)) / 86400.0

            # 3: time since last sensor event
            vec[1] = max(0.0, timestamp - self.queue.front().time)
        else:
            vec[0] = 0.0
            vec[1] = 0.0


        i = 2
        # 5: dominant sensor two windows back
        vec[i + self.prev_dom] = 1
        i += self.n

        # 4: dominant sensor for previous window
        vec[i + self.dom] = 1
        i += self.n

        self.prev_dom = self.dom

        for e in self.queue:
            # 8 - NumSensors+7: counts for each sensor
            idx = i + e.sensor
            cnt[ e.sensor ] += fmax(0.0, self.weigh(timestamp - e.time, e.sensor, self.latest_sensor))
            
            if cnt[ e.sensor ] > v_max:
                v_max = cnt[ e.sensor ]
                self.dom = e.sensor

        for idx in range(self.n):
            for j in range(self.n):
                vec[idx + i] += cnt[idx] * mi[idx,j]

        i += self.n

        # Dominant event in current window
        vec[i + self.dom] = 1
        i += self.n
        
        # 6: last sensor event in window
        vec[i + self.latest_sensor] = 1.0
        i += self.n

        # NumSensors+8 - 2*NumSensors+7: time since sensor last fired (<= SECSINDAY)
        for idx in range(self.n):
            vec[ i + idx ] = fmin(self.clip, fmax(0.0, timestamp - self.ptime[idx])) / self.clip

        """
        These are set in the self.with_time() function
        0: time of the last sensor event in window (hour)
        1: time of the last sensor event in window (seconds)
        """
        return np_vec

cdef class SERCTimeFeatures(SERCFeatures):
    def __str__(self):
        return "serc-time"
    cdef double weigh(self, double timestamp, int sensor, int latest):
        return self.time_weigh(timestamp, sensor, latest)

cdef class SERCTimeMiFeatures(SERCFeatures):
    def __str__(self):
        return "serc-mi-time"
    cdef double weigh(self, double timestamp, int sensor, int latest):
        return 0.5 *  (self.time_weigh(timestamp, sensor, latest)
                        + self.mi_weigh(timestamp, sensor, latest))

cdef class SERCMiFeatures(SERCFeatures):
    def __str__(self):
        return "serc-mi"
        
    cdef double weigh(self, double timestamp, int sensor, int latest):
        return self.mi_weigh(timestamp, sensor, latest)
