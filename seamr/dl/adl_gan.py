import os
import seamr
from seamr import core
from seamr import sem
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class Autoencoder(nn.Module):
    
    def __init__(self, insize = 100, units = 100, dims = 64, layers=2, bidirectional = True):
        super().__init__()

        self.insize = insize
        self.encoder = nn.LSTM( insize * 2,
                                units,
                                layers,
                                batch_first = False,
                                bidirectional = bidirectional )
        
        self.proj_size = units * layers * (int(bidirectional) + 1)

        self.proj = nn.Linear(self.proj_size, dims)
        self.unpack = nn.Linear(dims, insize)
        self.pMu      = nn.Linear(dims, dims)
        self.pLogVar  = nn.Linear(dims, dims)

        self.decoder = nn.LSTM(dims,
                               dims,
                               batch_first = False,
                               bidirectional = False)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, proj, n):
        outs = []
        input = proj.unsqueeze_(0)
        
        state = None
        for _ in range( n ):
            if state is None:
                o,state = self.decoder( input )
            else:
                o,state = self.decoder( input, state )

            outs.append( self.unpack(o) )
            input = o

        return torch.stack(list(reversed(outs)))

    def forward(self, seq):
    
        enc, (h_n, c_n) = self.encoder( seq )

        h_state = F.tanh( h_n.permute(1, 0, 2).contiguous().view( h_n.size(1), -1 ) )

        proj = self.proj( h_state )

        mu = F.relu( self.pMu( proj ) )
        logvar = self.pLogVar( proj )
        
        return self.reparameterize( mu, logvar )

class CGAN(nn.Module):

    def __init__(self, nClasses=0, insize=0, layers = []):
        super().__init__()
        self.layers = []
        self.insize = insize

        assert insize > 0

        w_in = insize
        for i,w in enumerate(layers):
            layer = nn.Linear( w_in + nClasses, w )
            self.layers.append( layer )
            setattr(self, "layer_%s" % i, layer)
            w_in = w
        
        self.output = nn.Linear( w_in + nClasses, insize )
        self.layers.append(self.output)

        self.pMu = nn.Linear( insize, insize )
        self.pLogVar = nn.Linear( insize, insize )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, condition):
        signal = Variable(torch.zeros(condition.size(0), self.insize).type_as(condition.data).normal_(), requires_grad =False)

        src = signal

        for l in self.layers:
            src = F.tanh( l(torch.cat([ condition, src ], dim=1)) )
        
        mu = self.pMu( src )
        logvar = self.pLogVar( src )
    
        rt = self.reparameterize(mu, logvar)

        return rt

class Classifier(nn.Module):
    def __init__(self, nClasses=0, insize=0, layers = []):
        super().__init__()

        assert nClasses > 0
        assert insize > 0

        self.nClasses = nClasses
        self.insize = insize
        self.layers = []

        for i,w in enumerate( layers ):
            layer = nn.Linear( insize + nClasses, w )
            self.layers.append( layer )
            setattr(self, "layer_%s" % i, layer)
            insize = w
        
        self.output = nn.Linear(insize + nClasses, nClasses)
    
    def forward(self, embedding, mask):

        input = embedding
        for l in self.layers:
            input = F.tanh( l( torch.cat([torch.squeeze(input), mask], dim=1) ) )
    
        return F.sigmoid( self.output(torch.cat([input, mask], dim=1)) )

class AEClassifier( nn.Module ):
    def __init__(self, autoencoder, classifier):
        super().__init__()
        self.step = 0
        self.autoencoder = core.build( autoencoder )
        self.classifier  = core.build( classifier )
        
        self.optim  = optim.Adam( self.parameters() )
    
    def save(self, model_dir):

        try:
            os.makedirs( model_dir )
        except:
            pass

        torch.save({'step': self.step,
                    'ae_state': self.autoencoder.state_dict(),
                    'cls_state': self.classifier.state_dict(),
                    'opt_state': self.optim.state_dict()
                },
                os.path.join( model_dir, "aecls.%d.tar" % self.step ))

    def load(self, from_path):
        if os.path.isdir(from_path):

            latest = max( os.listdir(from_path), key=lambda k: float(k.split(".")[1]) )

            file_path = os.path.join(from_path, latest)

            seamr.log.info("Loading checkpoint %s" % file_path)
            checkpoint = torch.load( file_path )
            
            self.step = checkpoint['step']
            
            self.autoencoder.load_state_dict(checkpoint['ae_state'])
            self.classifier.load_state_dict(checkpoint['cls_state'])
            self.optim.load_state_dict(checkpoint['opt_state'])
            
        else:
            seamr.log.info("No checkpoint found at '{}'".format(args.resume))


    def setPhase(self, ph):
        self.phase = ph

    def forward(self, seq, mask):
        return self.classifier( self.autoencoder(seq), mask )
    
    def update(self, batch):
        seq, mask, label = batch

        losses = {}

        try:
            self.optim.zero_grad()
            
            embedding = self.autoencoder(seq)

            pred = self.classifier( embedding, mask )

            loss = F.binary_cross_entropy( pred, label )
            
            ae_loss = F.mse_loss( self.autoencoder.decode(embedding, seq.size(0)), seq[ : , :, :self.autoencoder.insize ] )

            (loss + ae_loss).backward()
            self.optim.step()

            losses['Autoencoder'] = ae_loss.data.cpu().numpy()
            losses['Classification'] = loss.data.cpu().numpy()

            self.step += 1
        except RuntimeError:
            pass

        return losses
    
    def __str__(self):
        return "AECLS"

class ADLGAN( nn.Module ):
    
    COMPLETE = 0
    FITENCODER = 1

    def __init__(self, autoencoder, cgan, classifier):
        super().__init__()
        self.phase = self.COMPLETE
        self.step = 0

        self.autoencoder = core.build( autoencoder )
        self.classifier  = core.build( classifier )
        self.cgan        = core.build( cgan )

        self.ae_optim  = optim.Adam( self.autoencoder.parameters() )
        self.cls_optim = optim.Adam( self.classifier.parameters() )
        self.gen_optim = optim.Adam( self.cgan.parameters() )
    
    def __str__(self):
        return "ADLGAN"

    def save(self, model_dir):

        try:
            os.makedirs( model_dir )
        except:
            pass

        torch.save({'step': self.step,
                    'ae_state': self.autoencoder.state_dict(),
                    'cls_state': self.classifier.state_dict(),
                    'gen_state': self.cgan.state_dict(),
                    'ae_opt_state': self.ae_optim.state_dict(),
                    'cls_opt_state': self.cls_optim.state_dict(),
                    'gen_opt_state': self.gen_optim.state_dict() 
                },
                os.path.join( model_dir, "adlgan.%d.tar" % self.step ))

    def load(self, from_path):
        if os.path.isdir(from_path):

            latest = max( os.listdir(from_path), key=lambda k: float(k.split(".")[1]) )

            file_path = os.path.join(from_path, latest)

            seamr.log.info("Loading checkpoint %s" % file_path)
            checkpoint = torch.load( file_path )
            
            self.step = checkpoint['step']
            
            self.autoencoder.load_state_dict(checkpoint['ae_state'])
            self.ae_optim.load_state_dict(checkpoint['ae_opt_state'])

            self.classifier.load_state_dict(checkpoint['cls_state'])
            self.cls_optim.load_state_dict(checkpoint['cls_opt_state'])

            self.cgan.load_state_dict(checkpoint['gen_state'])
            self.gen_optim.load_state_dict(checkpoint['gen_opt_state'])
            
        else:
            seamr.log.info("No checkpoint found at '{}'".format(args.resume))


    def setPhase(self, ph):
        self.phase = ph

    def forward(self, seq, mask):
        return self.classifier( self.autoencoder(seq), mask )
    
    def update(self, batch):
        seq, mask, label = batch

        losses = {}

        self.ae_optim.zero_grad()
        embedding = self.autoencoder( seq )
        ae_loss = F.mse_loss( self.autoencoder.decode(embedding, seq.size(0)), seq[ : , :, :self.autoencoder.insize ] )
        ae_loss.backward()
        self.ae_optim.step()

        losses['Autoencoder'] = ae_loss.data.cpu().numpy()

        if self.phase is self.COMPLETE:
            self.cls_optim.zero_grad()
            self.gen_optim.zero_grad()

            # =======| Handle the Discriminator |=============

            fake_embedding = self.cgan( label )
            real_embedding = Variable(embedding.data, requires_grad = False)

            fake_pred = self.classifier( fake_embedding, mask )
            real_pred = self.classifier( real_embedding, mask )

            d_real_loss = F.binary_cross_entropy( real_pred, label )
            d_fake_loss = F.binary_cross_entropy( fake_pred, Variable(label.data.new(fake_pred.size()).zero_()) )

            total_loss = d_real_loss + d_fake_loss

            # =======| Handle the Generator |=============
            g_loss = F.binary_cross_entropy( fake_pred, label )
            total_loss += g_loss

            total_loss.backward()

            self.cls_optim.step()
            self.gen_optim.step()


            losses['D_real_loss'] = d_real_loss.data.cpu().numpy()
            losses['D_fake_loss'] = d_fake_loss.data.cpu().numpy()
            losses['Gen_loss'] = g_loss.data.cpu().numpy()
        
        self.step += 1

        return losses