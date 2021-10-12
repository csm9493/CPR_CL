import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms

args = get_args()
if args.conv_net:
    from networks.conv_net import Net
    
class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=5,clipgrad=100,args=None, log_name=None, split=False):
        self.model=model
        self.model_old=model
        self.fisher=None

        file_name = log_name
        
        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.split = split

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self.lamb=args.lamb
        self.cpr = CPR()
        self.beta = args.cpr_beta
        
        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr

        if args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr)
        if args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data, input_size, taskcla):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0=time.time()

            if 'split_CUB200' in args.experiment:
                xtrain_crop = crop_CUB200(xtrain)
                
                num_batch = xtrain_crop.size(0)
                self.train_epoch(t,xtrain_crop,ytrain, e)

                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain_crop,ytrain)
                clock2=time.time()
                
            else:
                
                num_batch = xtrain.size(0)
                self.train_epoch(t,xtrain,ytrain, e)

                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain)
                clock2=time.time()
                
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(
                e+1,1000*self.sbatch*(clock1-clock0)/num_batch,
                1000*self.sbatch*(clock2-clock1)/num_batch,train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')

            for task in range(t):
                if 'split_CUB200' in args.experiment:
                    xvalid_t=data[task]['valid']['x']
                    yvalid_t=data[task]['valid']['y'].cuda()
                else:
                    xvalid_t=data[task]['valid']['x'].cuda()
                    yvalid_t=data[task]['valid']['y'].cuda()

                valid_loss_t,valid_acc_t=self.eval(task,xvalid_t,yvalid_t)

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
                        if args.conv_net:
                            pass
#                             break
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.train()
        utils.freeze_model(self.model_old) # Freeze the weights
        
        if 'split_CUB200' in args.experiment:
                xtrain = crop_CUB200(xtrain)

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,deepcopy(self.model),self.criterion, split = self.split)
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option

        return

    def train_epoch(self,t,x,y, epoch):
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
            #torch.manual_seed(t*1000000+epoch*10000+i*2+1)
            if self.split:
                outputs = self.model.forward(images)[t]
            else:
                outputs = self.model.forward(images)
            loss=self.criterion(t,outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return

    def eval(self,t,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        
        if 'split_CUB200' in args.experiment:
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
            if self.split:
                output = self.model.forward(images)[t]
            else:
                output = self.model.forward(images)

            loss=self.criterion(t,output,targets)
            _,pred=output.max(1)
            hits=(pred==targets).float()

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

    def criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
    
        return self.ce(output,targets) + self.lamb*loss_reg - self.beta * self.cpr(output)

