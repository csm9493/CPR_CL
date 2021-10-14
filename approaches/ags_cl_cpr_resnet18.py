import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import copy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
args = get_args()

if 'omniglot' in args.experiment:
    from networks.conv_net_omniglot import Net
elif 'split_CUB200' in args.experiment:
    from networks.resnet18 import Net
else:
    from networks.conv_net import Net
    
class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None, log_name = None, split=False):
        self.model=model
        self.model_old=model
        self.omega=None
        self.log_name = log_name

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.lamb=args.lamb 
        self.initail_mu = args.mu
        self.mu = args.mu
        self.freeze = {}
        self.mask = {}
        self.args = args
        
        self.cpr = CPR()
        self.beta = args.cpr_beta
        
        for name, param in self.model.named_parameters():
            if ('conv' in name or 'fc' in name) and 'weight' in name:
                key =  '.'.join(name.split('.')[:-1])
                self.mask[key] = torch.zeros(param.shape[0], 1)

        self.update_frozen_model()

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):

        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        
        # Now, you can update self.t
        
        self.t = t

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()

            if 'split_CUB200' in args.experiment :
                xtrain_crop = crop_CUB200(xtrain)
                
                num_batch = xtrain_crop.size(0)
                self.train_epoch(t,xtrain_crop,ytrain,lr)

                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain_crop,ytrain)
                clock2=time.time()
                
            else:
                
                num_batch = xtrain.size(0)
                self.train_epoch(t,xtrain,ytrain,lr)

                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain,)
                clock2=time.time()
                
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            
            if 'split_CUB200' in args.experiment :
                xvalid_t=data[t]['valid']['x']
                yvalid_t=data[t]['valid']['y'].cuda()
            else:
                xvalid_t=data[t]['valid']['x'].cuda()
                yvalid_t=data[t]['valid']['y'].cuda()
                    
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            print(' lr : {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
            
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')

            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        test_loss, test_acc = self.eval(t, xvalid, yvalid)
        
        # Do not update self.t
        
        if 'split_CUB200' in args.experiment :
                xtrain = crop_CUB200(xtrain)

        self.omega_update(t, xtrain, ytrain, self.criterion, self.model, self.sbatch)
        self.reinitialization(input_size, taskcla)
        self.update_frozen_model()
        self.update_freeze()
        
        return

    def train_epoch(self,t,x,y,lr):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b]
            targets=y[b]

            # Forward current model
            
            outputs = self.model.forward(images)[t]
            loss=self.criterion(t,outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #Freeze the outgoing weights
            if t>0:
                for name, param in self.model.named_parameters():
                    if 'layer' not in name or 'bn' in name or 'downsample.1' in name:
                        continue
                    if ('conv' in name or 'fc' in name) and 'weight' in name:
                        key =  '.'.join(name.split('.')[:-1])
                    if 'downsample' in name:
                        key += 'downsample'
                    param.data = param.data*self.freeze[key]

        self.proxy_grad_descent(t,lr)
        
        return

    def eval(self,t,x,y):
        with torch.no_grad():
            total_loss=0
            total_acc=0
            total_num=0
            self.model.eval()
            
            if 'split_CUB200' in args.experiment :
                x = crop_CUB200(x, 'test')

            r = np.arange(x.size(0))
            r = torch.LongTensor(r).cuda()

            # Loop batches
            for i in range(0,len(r),self.sbatch): 
                if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
                else: b=r[i:]
                images=x[b]
                targets=y[b]

                # Forward
                
                output = self.model.forward(images)[t]

                loss=self.criterion(t,output,targets)
                _,pred=output.max(1)
                hits=(pred==targets).float()

                # Log
                total_loss+=loss.data.cpu().numpy()*len(b)
                total_acc+=hits.sum().data.cpu().numpy()
                total_num+=len(b)

            return total_loss/total_num,total_acc/total_num
    
    def cal_norm(self):
        norms_gs = {}
        norms_gp = {}
        for key in self.mask.keys():
            norms_gs[key] = torch.zeros_like(self.mask[key])
            norms_gp[key] = torch.zeros_like(self.mask[key])            
        
        key = None
        for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
            #Do not consider parameters in last layers
            if 'last' in name:
                continue
            #If 'conv' in name, change the key of norm dictionaries
            if ('conv' in name or 'fc' in name) and 'weight' in name:
                key =  '.'.join(name.split('.')[:-1])
            #resize
            param, param_old = [i.view(i.size(0), -1) for i in [param, param_old]]
            norms_gs[key] += param.norm(2, dim = -1, keepdim=True)**2
            norms_gp[key] += (param-param_old).norm(2, dim = -1, keepdim=True)**2
        for key in norms_gs.keys():
            norms_gs[key] = norms_gs[key].pow(1/2)
            norms_gp[key] = norms_gp[key].pow(1/2)
        
        return norms_gs, norms_gp
    
    def proxy_grad_descent(self, t, lr):

        with torch.no_grad():
            mu = self.mu
            norms_gs, norms_gp = self.cal_norm()
            key = None
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_fixed.named_parameters()):
                if 'last' in name:
                    continue
                    
                if ('conv' in name or 'fc' in name) and 'weight' in name:
                    key =  '.'.join(name.split('.')[:-1])

                norm_gs, norm_gp = norms_gs[key], norms_gp[key]
                data, data_old = [i.view(param.size(0), -1) for i in [param, param_old]]

                aux = F.threshold(norm_gs - mu * lr, 0, 0, False)
                alpha = aux/(aux+mu*lr)
                coeff = alpha * (1-self.mask[key])
                data_gs = data * coeff
                data_gp = 0

                if t>0:
                    aux = F.threshold(norm_gp - self.omega[key]*self.lamb*lr, 0, 0, False)
                    boonmo = lr*self.lamb*self.omega[key] + aux
                    alpha = (aux/boonmo)
                    alpha[alpha!=alpha] = 1 #remove nan elements

                    coeff_alpha = alpha * self.mask[key]
                    coeff_beta = (1-alpha) * self.mask[key]

                    data_gp = coeff_alpha * data + coeff_beta * data_old
                param.data = (data_gs + data_gp).reshape(param.size()) # squeeze for bias

        return


    def criterion(self,t,output,targets):
        return self.ce(output,targets) - self.beta * self.cpr(output)
    
    def cal_omega(self):
        # Init
        param_R = {}
        for name, param in self.model.named_parameters():
            if 'last' in name:
                continue
            if (('conv' in name) or ('fc' in name)) and 'weight' in name:
                key =  '.'.join(name.split('.')[:-1])
                param_R[key] = torch.zeros(param.shape[0], 1)

        # Compute
        self.model.train()
        for samples in tqdm(self.omega_iterator,desc='Omega update'):
            data, target = samples
            data, target = data.cuda(), target.cuda()

            # Forward and backward
            outputs = self.model.forward(data, True)[self.t]

            for idx, (act, key) in enumerate(zip(self.model.act, param_R.keys())):
                act = torch.mean(act, dim=0) # average N samples
                if len(act.size())>1:
                    act = torch.mean(act.view(act.size(0), -1), dim = 1).abs()
                param_R[key] += act.unsqueeze(-1).detach()*data.shape[0]

        with torch.no_grad():
            for key in param_R.keys():
                param_R[key]=(param_R[key]/len(self.train_iterator))
        return param_R

    
    def omega_update(self, t, x, y, criterion, model, sbatch):
        temp=self.cal_omega(t, x, y, criterion, model, sbatch)
        for n in temp.keys():
            if self.t>0:
                self.omega[n] = self.args.nu * self.omega[n]+temp[n] #+ t* temp[n]*mask      # Checked: it is better than the other option
            else:
                self.omega = temp
            self.mask[n] = (self.omega[n]>0).float()
            
    def cal_omega(self,t, x, y, criterion, model, sbatch=20):
        # Init
        param_R = {}
        for name, param in self.model.named_parameters():
            if 'last' in name:
                continue
            if (('conv' in name) or ('fc' in name)) and 'weight' in name:
                key =  '.'.join(name.split('.')[:-1])
                param_R[key] = torch.zeros(param.shape[0], 1)

        # Compute
        self.model.train()

        for i in range(0,x.size(0),sbatch):
            b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
            images=x[b]
            target=y[b]

            # Forward and backward
            outputs = model.forward(images, True)[t]

            for idx, (act, key) in enumerate(zip(self.model.act, param_R.keys())):
                act = torch.mean(act, dim=0) # average N samples
                if len(act.size())>1:
                    act = torch.mean(act.view(act.size(0), -1), dim = 1).abs()
                param_R[key] += act.unsqueeze(-1).detach()*sbatch

        with torch.no_grad():
            for key in param_R.keys():
                param_R[key]=(param_R[key]/x.size(0))
        return param_R


    
    def reinitialization(self, input_size, taskcla):
        t = self.t
        
        dummy = Net(input_size, taskcla).cuda()

        key = 0
        prekey = 0
        preprekey = 0
        for (name, param_dummy), (_, param) in zip(dummy.named_parameters(), self.model.named_parameters()):
            with torch.no_grad():
                if 'last' in name and str(t) not in name:
                    continue

                if ('conv' in name or 'fc' in name or 'last' in name) and 'weight' in name:
                    preprekey = prekey
                    prekey = key
                    key =  '.'.join(name.split('.')[:-1])

                # outgoing weight setting
                if prekey != 0 and not ('bn' in name or 'bias' in name) and 'downsample.1' not in name:
                    data = param.data
                    if 'downsample' in name:
                        mask = (self.omega[preprekey]>0).float().t()
                    else:
                        mask = (self.omega[prekey]>0).float().t()

                    if len(param.size()) > 2:
                        mask = mask.unsqueeze(-1).unsqueeze(-1)
                    elif len(param.size()) == 1:
                        mask = mask.squeeze()

                    if ('fc' in name or 'last' in name) and 'conv' in prekey: # conv -> linear
                        data = data.view(data.size(0), mask.size(1), -1)
                        mask = mask.unsqueeze(-1)
                        
                    data = data*mask
                    param.data = data.reshape(param.size())

                # incoming weight setting
                if 'last' in name:
                    continue
                data = param.view(param.size(0), -1)
                data_dummy = param_dummy.view(param_dummy.size(0), -1)

                norm = data.norm(2, dim = -1, keepdim=True)
                mask = (norm == 0).float()
                tmp = (1-mask)*data + mask*data_dummy
                param.data = tmp.reshape(param.size())
                
    def update_frozen_model(self):
        self.model.eval()
        self.model_fixed = copy.deepcopy(self.model)
        self.model_fixed.eval()
        for param in self.model_fixed.parameters():
            param.requires_grad = False
            
    def update_freeze(self):
        self.freeze = {}
        
        key = 0
        prekey = 0
        preprekey = 0
        for name, param in self.model.named_parameters():
            with torch.no_grad():
                if 'last' in name:
                    continue

                if ('conv' in name or 'fc' in name or 'last' in name) and 'weight' in name:
                    preprekey = prekey
                    prekey = key
                    key =  '.'.join(name.split('.')[:-1])

                # outgoing weight setting
                if prekey != 0 and not ('bn' in name or 'bias' in name) and 'downsample.1' not in name:
                    temp = torch.ones_like(param)
                    if 'downsample' in name:
                        temp[:, self.omega[preprekey].squeeze() == 0] = 0
                        temp[self.omega[key].squeeze() == 0] = 1
                        self.freeze[key+'downsample'] = temp
                    elif 'conv' in name:
                        temp[:, self.omega[prekey].squeeze() == 0] = 0
                        temp[self.omega[key].squeeze() == 0] = 1
                        self.freeze[key] = temp
                    else: 
                        temp = temp.reshape((temp.size(0), self.omega[prekey].size(0) , -1))
                        temp[:, self.omega[prekey].squeeze() == 0] = 0
                        temp[self.omega[key].squeeze() == 0] = 1
                        self.freeze[key] = temp.reshape(param.shape)

            


