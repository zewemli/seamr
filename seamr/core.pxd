cimport numpy as np
from libcpp cimport bool
from libcpp.deque cimport deque

# -----------------------------------------------------------------------
# Common Features structures and class
# -----------------------------------------------------------------------

cdef class SensorState:
    cdef deque[double] * times
    cdef double decay
    cdef double state
    cdef double mag
    cdef double newestTime
    cdef bool isBinary

    cpdef void reset(self)
    cpdef void update(self, double atTime, double state)
    cpdef int trimLength(self, int toLen) except -1
    cpdef int trimSize(self, double minTime) except -1
    cpdef int trim(self, int toLen, double minTime) except -1

    cpdef double getState(self, double atTime)
    cpdef double getCount(self, double atTime)
    cpdef double getSpan(self, double atTime)
    cpdef double getDelta(self, double atTime)
    cpdef double getDutyCycle(self, double atTime)

    cpdef np.ndarray getVector(self, double atTime)

cdef class EventBatcher:
    cdef int maxSteps
    cdef float stepBy

# -----------------------------------------------------------------------
# Base Features class
# -----------------------------------------------------------------------
cdef class Features:
    cdef public bool stack

    cpdef int partitionCount(self)
    cpdef void reset(self)
    cdef int time_width(self)
    cpdef list gen_batches(self, stream, times)

# -----------------------------------------------------------------------
# Base Event class
# -----------------------------------------------------------------------
cdef class Event:
    cdef readonly int srcLine
    cdef readonly int sensorID
    cdef readonly double state
    cdef readonly int day
    cdef readonly double sec_of_day
    cdef readonly double timestamp
