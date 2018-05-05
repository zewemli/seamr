import os,sys
import pickle
import shutil
import numpy as np
import seamr
from seamr import core
from seamr import sem
import random

from tensorboardX import SummaryWriter

from datetime import datetime
from multiprocessing import Pool, cpu_count

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch
from tqdm import tqdm

def point_time_label(sec_of_day, start, end, offset):
    h = 3600.0
    if (start * h) <= sec_of_day <= (end * h):
        return 1.0
    else:
        dist = min(abs((start * h + offset) - sec_of_day), abs((end * h + offset) - sec_of_day))
        try:
            return 0.975 ** (dist / 60)
        except:
            return 0.0

def time_label(sec_of_day, start, end):
    return max( point_time_label(sec_of_day, start, end, 0),
                point_time_label(sec_of_day, start, end, 86400) )

def get_times(ts):
    
    times = [
        ('cose:Night', 0, 5),
        ('cose:Morning', 5, 9),
        ('cose:MidMorning', 9, 12),
        ('cose:Midday', 12, 14),
        ('cose:Afternoon', 14, 17),
        ('cose:Twilight', 17, 21),
        ('cose:Evening', 21, 24)
    ]
    for fname, start, end in times:
        f_weight = time_label( ts % 86400, start, end )
        if f_weight > (10.0 ** -3):
            yield fname, f_weight


class BalancedSampler(sampler.Sampler):
    """Get balanced sample data for training.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, samples_per_dataset):
        self.data_source = data_source
        self.rows = data_source.balancedSample( samples_per_dataset )

    def __iter__(self):
        self.i=0
        return self
    
    def __next__(self):
        row = self.i
        self.i += 1
        try:
            return self.rows[row]
        except IndexError:
            raise StopIteration()

    def __len__(self):
        return len(self.rows)

def semName(x):
    return sem.activity_type_map[x].split(":")[-1].lower()

class Dataset:
    
    def __init__(self, store, dset, start=0, end=0, seqlen=30):

        self.store = store
        self.dset = dset

        if isinstance(dset, Dataset):

            if end <= start:
                end = len(self)

            for p in ['seqlen', 'actIndex', 'allFields', 'structVec', 'actVec']:
                setattr(self, p, getattr(dset, p, None))
            
            self.mat = dset.mat[start : end, :]
            self.labels = dset.labels[start : end]
            self.index = {}

            if start > 0:
                for k,rows in dset.index.items():
                    self.index[k] = { r - start for r in rows if start <= r < end }

            else:
                for k,rows in dset.index.items():
                    self.index[k] = { r for r in rows if start <= r < end }

        else:

            self.seqlen = seqlen
            self.index = {}

            allFields = set( sem.COSE.time_order().keys() )
            allActs = set()
            for ds in store.datasets():
                with open( store.path(ds, "matrix_feats.pkl"), "rb" ) as f:
                    allFields |= set(self.cleanNames(pickle.load(f)))
                    
                allActs |= set( semName( x.name ) for x in store.get_acts(ds) if x.name in sem.activity_type_map )

            self.allActs = sorted( allActs )
            self.actIndex = { c:i for i,c in enumerate(self.allActs) }

            self.allFields = sorted(allFields)

            with open( store.path(dset, "matrix_feats.pkl"), "rb" ) as f:
                self.fields = self.cleanNames(pickle.load(f))
            
            acts = set( semName( x.name ) for x in store.get_acts(dset) if x.name in sem.activity_type_map )

            mp = { i : self.allFields.index(v) for i,v in enumerate(self.fields) }
            n = sum( 1 for _ in store.get_matrices(dset, days=[0]) ) # the days=[0] is a bug, but not really important now...
            
            self.mat = np.zeros( (n, len(self.allFields)), dtype = np.float32 )
            self.labels = [ None ] * n

            self.structVec = np.array([ int(a in self.fields) for a in self.allFields ], dtype = np.float32)
            self.actVec = np.array([ int(a in acts) for a in allActs ], dtype = np.float32)

            self.structVec = np.expand_dims(self.structVec, axis=0)


            tpIndex = { t: self.allFields.index(t) for t in sem.COSE.time_order().keys() }

            for row,(day,(t,lbl,d)) in tqdm(enumerate(store.get_matrices(dset, days=[0])), desc=dset): #
                for i,v in d.items():
                    self.mat[ row, mp[i] ] = v

                for tm, w in get_times(t):
                    self.mat[row, tpIndex[tm]] = w

                self.labels[ row ] = [ semName( x ) for x in lbl if x in sem.activity_type_map ]

                for l in self.labels[row]:
                    try:
                        self.index[l].add(row)
                    except KeyError:
                        self.index[l] = set([row])
        
        # Make sure the data is OK
        assert self.mat.shape[0] == len(self.labels)
    
    def cleanNames(self, lst):
        return [ x.replace("cose:","") for x in lst ]

    def balancedSample(self, n):
        useRows = set()
        
        for i,(l,rows) in enumerate(self.index.items()):
            k = int(n / (len(self.index) - i))
            if k >= len(rows):
                useRows |= rows
            else:
                useRows |= set( random.sample(list(rows), k) )
        
        return useRows


    def width(self):
        return self.mat.shape[1]

    def __getitem__(self, i):
        y = np.zeros( len(self.actIndex) )
        for l in self.labels[i]:
            y[ self.actIndex[l] ] = 1.0
        
        if i < self.seqlen:
            mb = np.zeros( (self.seqlen, self.mat.shape[1]) )
            mb[ self.seqlen - i :, :] = self.mat[:i,:]
        else:
            mb = self.mat[(i+1) - self.seqlen : i+1,:]

        sVec = np.repeat(self.structVec, mb.shape[0], axis=0)

        return np.hstack([mb, sVec]), self.actVec, y

    def __len__(self):
        return self.mat.shape[0]
    
    def split(self, percent = 10):
        n = int(len(self) * (percent/100))

        return Dataset(self.store, self, end=n), Dataset(self.store, self, start = n)
        

class MultiEnvData:

    def __init__(self, datasets):
        self.datasets = datasets
    
    def balancedSample(self, n_per_dataset):
        offset = 0
        samples = []
        for d in self.datasets:
            dSample = d.balancedSample(n_per_dataset)
            samples.extend( offset + r for r in dSample )
            offset += len(d)
        
        return samples

    def __len__(self):
        return sum(len(d) for d in self.datasets)
        
    def __getitem__(self, i):
        for d in self.datasets:
            if i >= len(d):
                i -= len(d)
            else:
                return d[i]

def collater(use_cuda):

    def collate_sequence( examples ):
        X = torch.from_numpy( np.stack([ ex[0] for ex in examples ], axis=1) ).type(torch.FloatTensor)
        actMask = torch.from_numpy(np.stack([ ex[1] for ex in examples ], axis=0)).type(torch.FloatTensor)
        label = torch.from_numpy(np.stack([ ex[2] for ex in examples ], axis=0)).type(torch.FloatTensor)

        if use_cuda:
            return (Variable(X.cuda()),
                    Variable(actMask.cuda()),
                    Variable(label.cuda()),)
        else:
            return (Variable(X),
                    Variable(actMask),
                    Variable(label),)
    
    return collate_sequence
        
class Trainer:

    def __init__(self, batch_size = 32, disable_cuda = False):
        self.use_cuda = torch.cuda.is_available() and not disable_cuda
        self.batch_size = batch_size
         
    def train(self, model, dataset, epochs = 100, sample=None, validate=None, log_to = "./", save_to = None, save_freq = 10):
        
        if self.use_cuda:
            model.cuda()
        else:
            model.cpu()

        collate_fn = collater(self.use_cuda)

        if validate:
            _msg = "Validate argument must be an integer percentage between 1 and 100"
            assert isinstance(validate, int) and 1 <= validate < 100, _msg
            training, validation = dataset.split( validate )
        else:
            training = dataset
            validation = None

        step = 0

        log = SummaryWriter( log_dir = os.path.join(log_to, datetime.now().isoformat("-")) )

        for epoch in range(1, 1+epochs):

            with seamr.BlockWrapper("Training for epoch %s" % epoch):
                try:

                    model.train()
                    #--------------------------------------
                    if sample is None:
                        trainingSampler = sampler.RandomSampler(training)
                    else:
                        trainingSampler = BalancedSampler(training, sample)

                    loader = DataLoader(training,
                                        sampler = trainingSampler,
                                        collate_fn = collate_fn,
                                        batch_size = self.batch_size )
                                        
                    with tqdm(total=len(trainingSampler), desc="Training epoch %s" % epoch) as progress:
                        for batch in loader:
                            #--------------------------------------

                            loss = model.update(batch)

                            if isinstance(loss, dict):
                                for name,val in loss.items():
                                    log.add_scalar(name, val, global_step = step)
                            else:
                                log.add_scalar("Loss", loss, global_step = step)

                            #--------------------------------------
                            progress.update( batch[-1].shape[0] )
                            step += 1

                finally:
                    model.eval()
            
            if validation is not None:
                try:
                    model.eval()
                    #--------------------------------------
                    with seamr.BlockWrapper("Validation for epoch %s" % epoch):
                        reals, preds = self.predict( model, validation )
                        for c,i in validation.actIndex.items():
                            log.add_pr_curve("PR_%s" % c.replace(" ","_"), reals[:,i], preds[:,i], global_step=step)

                finally:
                    model.train()
            
            if save_to and save_freq:
                if epoch % save_freq == 0:
                    model.save( save_to )

    def predict(self, model, dataset):
        #--------------------------------------
        if self.use_cuda:
            model.cuda()
        else:
            model.cpu()

        model.eval()

        loader = DataLoader(dataset,
                            sampler = sampler.SequentialSampler(dataset),
                            collate_fn = collater(self.use_cuda),
                            batch_size = self.batch_size )
        
        reals = []
        preds = []

        with tqdm(desc="Predicting %s" % dataset.dset, total = len(dataset)) as progress:
            for batch in loader:
                reals.append( batch[-1].data.cpu().numpy() )
                preds.append( model(batch[0], batch[1]).data.cpu().numpy() )
                progress.update(batch[0].shape[0])

        reals = np.vstack(reals)
        preds = np.vstack(preds)

        return reals, preds
        #--------------------------------------
