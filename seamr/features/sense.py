from seamr import core

import numpy as np
import math

class SensibleFeatures( core.Features ):
    def __init__(self, feats, store=None, sensors=None, dset=None):
        self.feats = core.build(feats, local_args = locals())

    def make_dataset(self, event_stream, times):
        return event_stream, times, self.feats.make_dataset(event_stream, times)
    
    def __str__(self):
        return "sense-%s" % str(self.feats)