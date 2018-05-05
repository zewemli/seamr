import os
import sys
from seamr import rules
import yaml
from tqdm import tqdm
import numpy as np

class RuleGroupFeatures:
    def __init__(self, store, dset, rules_path, after_space = 120):
        self.store = store
        self.dset = dset
        self.after_space = after_space

        self.rules = {}
        self.rulesName = os.path.basename(rules_path).replace(".yaml", "")
        for rfile in os.listdir(rules_path):
            if rfile.endswith(".yaml"):
                rule_name = rfile.replace(".yaml","")
                with open(os.path.join(rules_path, rfile)) as fr:
                    y = yaml.load(fr)
                    r = rules.Rule( *y['given'] )
                    if y.get('after', None):
                        r.after( *y['after'] )
                    self.rules[ rule_name ] = r
        
        self.feat_names = sorted(self.rules.keys()) + [ "hr%02d" % (h+1) for h in range(24) ]
    
    def __str__(self):
        return "rulegroup-%s" % self.rulesName

    def make_dataset(self, events, times):
        engine = rules.Engine(self.store, self.dset, events=events, times=times)
        rvec = []
        for k,rule in tqdm(sorted( self.rules.items() ), total=len(self.rules), desc="Applying rules"):
            rvec.append( engine.apply(rule) )
        
        hours = np.zeros( (len(times), 24) )
        for i,t in enumerate(times):
            hours[ i, int((t % 86400) // 3600) ] = 1
        
        rMat = np.hstack([ np.stack(rvec).T, hours])
        assert rMat.ndim == 2
        assert rMat.shape[0] == len(times)
        return rMat

#-------------------------------------------------------------------