import os
from seamr import core
from seamr import sem
from seamr.evaluate import ROCAnalysis
from seamr.features.semantic import CrossEnvFeatures

from collections import Counter, deque
from itertools import groupby, product, chain, combinations
from tqdm import tqdm
import yaml
import numpy as np

from prg import prg

#-------------------------------------------------------------------
class Token:
    def __init__(self, engine, raw_tok):
        self.engine = engine

        try:
            self.neg = raw_tok.startswith("~")
            tok = raw_tok.strip("~").split("-")

            if len(tok) == 1:
                self.kind = engine.typeLookup.get(tok[0].lower(), tok[0])
                self.ph = None
            else:
                k,p = tok
                k = engine.typeLookup.get(k.lower(),k)
                p = engine.typeLookup.get(p.lower(),p)
                self.kind, self.ph = k,p

        except Exception as e:
            raise ValueError("Invalid token [{}] : valid forms are ~?kind(-phenomenon)?".format(raw_tok) )

    def startswith(self, s):
        if s == "~":
            return self.neg
        else:
            return self.kind.startswith(s)

    def phOK(self, other):
        return (self.ph is None) or (other.ph is None) or (self.ph == other.ph)

    def __hash__(self):
        return hash((self.kind, self.ph))

    def __str__(self):
        return "%s-%s" % (self.kind, self.ph if self.ph else "*")

    def __lt__(self, other):
        if not self.phOK(other):
            return False
        else:
            return self.engine.cose.is_a( self.kind, other.kind )
        
    def __le__(self, other):
        return self < other

    def __eq__(self, other):
        if not self.phOK(other):
            return True
        else:
            return self.kind == other.kind
        
    def __ne__(self, other):
        if not self.phOK(other):
            return True
        else:
            return self.kind != other.kind
        
    def __gt__(self, other):
    
        if not self.phOK(other):
            return False
        else:
            n = self.engine
            return n.cose.is_a( other.kind, self.kind )
        

    def __ge__(self, other):
        return self > other

#-------------------------------------------------------------------
class Engine:
    def __init__(self,
                store,
                dset,
                days = None,
                after_space = 120,
                step_by = 15,
                max_steps = 20,
                events = None,
                times = None):
                
        self.feats = CrossEnvFeatures(store, dset, decay=0)
        self.after_space = after_space
        self.typeLookup = {}
        self.cose = sem.COSE()

        if events is None:
            events = lambda: store.get_events(dset, days=days)
            times = core.get_times(events, step=step_by, max_steps = max_steps)

        for n in self.cose.typeGraph().nodes():
            self.typeLookup[ n.split(":")[-1].lower() ] = n

        self.segments = []
        
        for patches,ts in tqdm(self.feats.genPatches(events, times), total=len(times), desc="Building rules dataset"):
            pset = set([ Token(self, p.kindKey) for p in patches ])
            if len(pset) == 0:
                pset.add( Token(self, "cose:Nothing") )
            pset.add( Token(self, self.cose.time_label(ts % 86400)) )
            self.segments.append( (ts, pset) )
        
        self.store = store
        self.dset = dset
        self.days = days
        self.times = times

    def jaccard(self, tokens, obs):
        mCount = 0
        notMatched = 0
        matchedObs = set()

        for tok in tokens:
            missed = True
            for o in obs:
                if o <= tok:
                    ok = False
                    mCount += 1
                    matchedObs.add( o )
                    break
            notMatched += int(missed)

        denom = float(len(tokens) + len(obs)) - mCount
        return mCount / denom 

    def _weigh(self, rule, obsSet, afterSet = None):
        w = 0.0

        if rule.pos_token:
            w += self.jaccard( rule.pos_token, obsSet )
        
        if rule.neg_token:
            w -= self.jaccard( rule.neg_token, obsSet )
        
        if afterSet and rule.prior:
            w += self.jaccard( rule.prior.pos_token, afterSet )
            w -= self.jaccard( rule.prior.neg_token, afterSet )
        
        return w

    def apply(self, rule):
        """
        Returns a list of matching weights
        weight = jaccard(example, rule-positive) - jaccard(example, rule-negative) + weight(prior, afterSet)
        """

        tokenRule = rule.tokenize(self)

        tups = []
        after = Counter()
        aq = deque()
        for ts, obsSet in self.segments:
            for o in obsSet:
                aq.append( (ts, o) )
                after[o] += 1

            while len(aq) and (ts - aq[0][0]) > self.after_space:
                old_ts, old_o = aq.popleft()
                if after[old_o] == 1:
                    after.pop(old_o)
                else:
                    after[old_o] -= 1

            tups.append( self._weigh(tokenRule, obsSet, set( after.keys() )) )

        return tups

    @staticmethod
    def get_perf(real, pred):
        
        predOrd = np.argsort(pred)

        real = real[ predOrd ]
        pred = pred[ predOrd ]
        
        fn = 0.0
        tp = sum(real)
        fp = real.size - tp

        precision = 0
        recall = 0
        optimalF1 = 0

        for pval, section in groupby(zip(real,pred), key=lambda T: T[1]):
            section = [ r for (r,_) in section ]
            
            n = len(section)
            npos = sum(section)

            fn += npos 
            fp -= n - npos
            tp -= npos
            
            i_f1 = ( (2 * tp) / (2*tp + fp + fn) )
            if i_f1 > optimalF1:
                optimalF1 = i_f1
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)

        return optimalF1, precision, recall, prg.calc_auprg( prg.create_prg_curve(real, pred) )
    
    def get_labels(self):
        if getattr(self, "_classIndex", None) is None:
            lIndex = self.store.get_labels(self.dset, days = self.days)
            classes, labels = lIndex.get_dense_mat( self.times )
            self._classIndex = { c:i for i,c in enumerate(classes) }
            self._labels = labels
        
        return (self._classIndex, self._labels)

    def positiveBaseline(self, check_class):
        classIndex, labels = self.get_labels()
        reals = labels[ :, classIndex[check_class] ]
        preds = np.ones( labels.shape[0] )
        preds[0] = 0.0
        return self.get_perf( reals, preds )

    def negativeBaseline(self, check_class):
        classIndex, labels = self.get_labels()
        reals = labels[ :, classIndex[check_class] ]
        preds = np.zeros( labels.shape[0] )
        preds[0] = 1.0
        return self.get_perf( reals, preds )

    def evalRule(self, rule, check_class):

        classIndex, labels = self.get_labels()
     
        preds = self.apply(rule)
        reals = labels[ :, classIndex[check_class] ]
        return self.get_perf( reals, np.array(preds) )

#-------------------------------------------------------------------
class Rule:
    def __init__(self, *conds):
        self.conds = conds
        self.pos_token = None
        self.neg_token = None
        self.prior = None
        self.consequent = None

        self.given(conds)
    
    def given(self, *conds):
        
        if isinstance(conds, str):
            self.conds = [conds]
        elif isinstance(conds[0], tuple):
            self.conds = list(chain(*conds))
        else:
            self.conds = conds
        
        self.pos_token = set([ c for c in self.conds if not c.startswith("~") ])
        self.neg_token = set([ c for c in self.conds if c.startswith("~") ])
        return self
    
    def after(self, *priors):
        self.prior = Rule().given( priors )
        return self
    
    def tokenize(self, engine):
        r = Rule(*[ Token(engine, c) for c in self.conds ])
        if self.prior is None:
            return r
        else:
            return r.after(*[ Token(engine, c) for c in self.prior.conds ])
    
    @staticmethod
    def load(from_file):
        with open(from_file, "rt") as fi:
            dc = yaml.load(fi)
            return Rule(*dc.get('given',[])).after( *dc.get('after',[]) )

    def save(self, to_file):
        with open(to_file,"wt") as fo:
            gvn = list(self.conds)
            aftr = []
            if self.prior:
                aftr = list(self.prior.conds)

            yaml.dump( { "given": gvn, "after": aftr }, fo, default_flow_style=False )

def models_to_rules( to_path ):
    cose = sem.COSE()

    if not os.path.exists(to_path):
        os.makedirs(to_path)

    models = cose.get_activity_models()

    ok_acts = set([ k.split(":")[-1].lower() for k in sem.activity_type_map.values() ])

    for act,m in models.items():
        act = act.split(":")[-1].lower()
        if act in ok_acts:
            concepts = [ c.split(":")[-1].lower() for c in m.concepts() ]

            Rule( *concepts ).save( os.path.join(to_path, "%s.yaml" % act) )