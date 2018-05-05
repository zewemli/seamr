from seamr.core import build

class ACEFeatures():

    def __init__(self, feats, groupby=None, store=None, sensors = None, dset = None):
        self.feats = build(feats, locals())
        if groupby is None:
            self.groupby = None
        else:
            self.groupby = build(groupby, locals())
    
    def __str__(self):
        if self.groupby is None:
            return str(self.feats)
        else:
            return str(self.feats) + "-wrt-" + str(self.groupby)

    def make_dataset(self, *args, **kw):
        feats = self.feats.make_dataset(*args, **kw)
        
        if self.groupby is None:
            grouping = feats
        else:
            grouping = self.groupby.make_dataset(*args, **kw)

        return grouping, feats