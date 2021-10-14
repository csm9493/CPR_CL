import os,sys
import numpy as np
import random
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
from sklearn.feature_extraction import image
import torchvision.transforms.functional as tvF
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models.resnet import *
from arguments import get_args
import torch.nn.functional as F

args = get_args()

class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, lr_rho=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, param_name=None, lr_scale=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        self.param_name = param_name
        self.lr_rho = lr_rho
        self.lr_scale = lr_scale
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i,p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                n = self.param_name[i]

                if 'rho' in self.param_name[i]:
                    step_size = self.lr_rho * math.sqrt(bias_correction2) / bias_correction1
                else:
                    step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

#                 p.data.addcdiv_(-step_size, self.lr_scale[n] * exp_avg, denom)
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

def gs_cal(t, x, y, criterion, model, sbatch=20):
    
    # Init
    param_R = {}
    
    for name, param in model.named_parameters():
        if len(param.size()) <= 1:
            continue
        name = name.split('.')[:-1]
        name = '.'.join(name)
        param = param.view(param.size(0), -1)
        param_R['{}'.format(name)]=torch.zeros((param.size(0)))
    
    # Compute
    model.train()

    for i in range(0,x.size(0),sbatch):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=x[b]
        target=y[b]

        # Forward and backward
        outputs = model.forward(images, True)[t]
        cnt = 0
        
        for idx, j in enumerate(model.act):
            j = torch.mean(j, dim=0)
            if len(j.size())>1:
                j = torch.mean(j.view(j.size(0), -1), dim = 1).abs()
            model.act[idx] = j
            
        for name, param in model.named_parameters():
            if len(param.size()) <= 1 or 'last' in name or 'downsample' in name:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            param_R[name] += model.act[cnt].abs().detach()*sbatch
            cnt+=1 

    with torch.no_grad():
        for key in param_R.keys():
            param_R[key]=(param_R[key]/x.size(0))
    return param_R

def gs_cal_resnet18(t, x, y, criterion, model, sbatch=20):
    
    # Init
    param_R = {}
    
    for name, param in model.named_parameters():
        if 'last' in name:
            continue
        if (('conv' in name) or ('fc' in name)) and 'weight' in name:
            key =  '.'.join(name.split('.')[:-1])
            param_R[key] = torch.zeros(param.shape[0])
            print (param_R[key].shape)
    # Compute
    model.train()
    
    for i in range(0,x.size(0),sbatch):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=x[b]
        target=y[b]

        # Forward and backward
        outputs = model.forward(images, True)[t]
        cnt = 0
        
        for idx, j in enumerate(model.act):
            j = torch.mean(j, dim=0)
            if len(j.size())>1:
                j = torch.mean(j.view(j.size(0), -1), dim = 1).abs()
            model.act[idx] = j
            
        for name, param in model.named_parameters():
            if len(param.size()) <= 1 or 'last' in name or 'downsample' in name:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            param_R[name] += model.act[cnt].abs().detach()*sbatch
            cnt+=1 

    with torch.no_grad():
        for key in param_R.keys():
            param_R[key]=(param_R[key]/x.size(0))
    return param_R

########################################################################################################################
def crop(x, patch_size, mode = 'train'):
    cropped_image = []
    arr_len = len(x)
    if mode == 'train':
        for idx in range(arr_len):
            patch = image.extract_patches_2d(image = x[idx].data.cpu().numpy(),
                                            patch_size = (patch_size, patch_size), max_patches = 1)[0]

            # Random horizontal flipping
            if random.random() > 0.5:
                patch = np.fliplr(patch)
            # Random vertical flipping
            if random.random() > 0.5:
                patch = np.flipud(patch)
            # Corrupt source image
            patch = np.transpose(patch, (2,0,1))
            patch = tvF.to_tensor(patch.copy())
            cropped_image.append(patch)

    elif mode == 'valid' or mode == 'test':
        for idx in range(arr_len):
            patch = x[idx].data.cpu().numpy()
            H,W,C = patch.shape
            patch = patch[H//2-patch_size//2:H//2+patch_size//2, W//2-patch_size//2:W//2+patch_size//2,:]
            # Corrupt source image
            patch = np.transpose(patch, (2,0,1))
            patch = tvF.to_tensor(patch.copy())
            cropped_image.append(patch)

    image_tensor=torch.stack(cropped_image).view(-1,3,patch_size,patch_size).cuda()
    return image_tensor

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20, split = False):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=x[b]
        target=y[b]
        
        # Forward and backward
        model.zero_grad()
        if split:
            outputs = model.forward(images)[t]
        else:
            outputs=model.forward(images)
        loss= criterion(outputs, target)
      #  loss=criterion(t,outputs,target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n,_ in model.named_parameters():
            fisher[n]=fisher[n]/x.size(0)
    return fisher

########################################################################################################################

class CPR(nn.Module):
    def __init__(self):
        super(CPR, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        
        return b.mean()

def crop_CUB200(data, _type = 'train'):
    mean = torch.FloatTensor([[[123.77, 127.55, 110.25]]])
    std = torch.FloatTensor([[[59.16, 58.06, 67.99]]])
    size=[3,224,224]
    
    cropped_data = []
    
    for i in range(len(data)):
        
        if _type == 'train':

            img = data[i] * std + mean
            img = transforms.ToPILImage()(img.permute(2,0,1)).convert("RGB")

#             i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(224, 224))
#             cropped_patch = tvF.crop(img, i, j, h, w)
            
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
            cropped_patch = tvF.resized_crop(img, i, j, h, w, (224, 224))

            # Random horizontal flipping
            if random.random() > 0.5:
                cropped_patch = tvF.hflip(cropped_patch)

            # Random vertical flipping
            if random.random() > 0.5:
                cropped_patch = tvF.vflip(cropped_patch)
                
            cropped_data.append(transforms.Normalize((123.77, 127.55, 110.25), (59.16, 58.06, 67.99))(tvF.to_tensor(cropped_patch)*255.))
                
        else:

            c, h, w = data[i].shape
            
            if h == 224 and w == 224:
                
                return data
                
            else:
                img = data[i] * std + mean
                img = transforms.ToPILImage()(img.permute(2,0,1)).convert("RGB")
                img = tvF.center_crop(img, 224)
                
                cropped_data.append(transforms.Normalize((123.77, 127.55, 110.25), (59.16, 58.06, 67.99))(tvF.to_tensor(img)*255.))
            
    return torch.stack(cropped_data).view(-1,size[0],size[1],size[2]).cuda()
