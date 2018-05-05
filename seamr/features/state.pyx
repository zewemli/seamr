from seamr import core
from seamr.core cimport Features

import numpy as np
cimport numpy as np

from cymem.cymem cimport Pool as MemPool
cdef MemPool mem = MemPool()


# -----------------------------------------------------------------------
# Simple State Features
# -----------------------------------------------------------------------
cdef class StateFeatures(Features):
    cdef int nSensors
    cdef int nFeats
    cdef double * state
    cdef double * low_obs
    cdef double * obs_range
    
    cdef public list feat_names

    def __cinit__(self, list sensors, stack=False):
        self.stack = stack

        self.nSensors = len(sensors)
        self.state    = <double *> mem.alloc(self.nSensors, sizeof(double))
        self.low_obs  = <double *> mem.alloc(self.nSensors, sizeof(double))
        self.obs_range = <double *> mem.alloc(self.nSensors, sizeof(double))
        self.nFeats = self.nSensors + self.time_width()
        self.feat_names = []

        for i in range(self.nSensors):
            self.state[i] = 0.0
            self.low_obs[i] = 0.0
            self.obs_range[i] = 1.0
            self.feat_names.append( sensors[i].name )

        self.feat_names.append( "sec-of-day" )
        self.feat_names.append( "hour" )
    
    def __str__(self):
        return "state"

    def width(self):
        return len(self.feat_names)

    def fit(self, labels, stream, times):
        cdef np.ndarray high = np.zeros(self.nSensors)
        cdef double[:] view = high
        cdef double rng = 0.0
        
        for e in stream():
            self.low_obs[e.sensorID]  = min(self.low_obs[e.sensorID], e.state)
            high[e.sensorID] = max(high[e.sensorID], e.state)
        
        for i in range(self.nSensors):
            rng = high[i] = self.low_obs[i]
            if rng > 0:
                self.obs_range[i] = rng

    cpdef void update(self, int eLine, int sensor, double timestamp, double state):
        self.state[sensor] = state

    cpdef np.ndarray fetch(self, double timestamp):
        cdef np.ndarray vec = np.zeros(self.nFeats)
        cdef double[:] view = vec

        for i in range(self.nSensors):
            view[i] = (self.state[i] - self.low_obs[i]) / self.obs_range[i]

        return vec
