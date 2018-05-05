from seamr.core import build

class Classifier:
    def __init__(self, key, model):
        self._key_ = key
        self._model_ = build(model)
    
    def __getattr__(self, attr):
        return getattr(self._model_, attr)
    
    def __str__(self):
        return self._key_
    
    def __repr__(self):
        return self._key_