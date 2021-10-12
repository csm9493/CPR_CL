import sys, os, time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

import pickle
import utils
import torch
from arguments import get_args
import random

tstart = time.time()

# Arguments

args = get_args()
if args.approach == 'si_cpr':
    log_name = '{}_{}_{}_{}_c_{}_lr_{}_cpr_beta_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed,
                                                                    args.c, args.lr, args.cpr_beta, args.batch_size, args.nepochs)
elif args.approach == 'finetuning' or args.approach == 'joint':
    log_name = '{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed, args.lr,
                                                                             args.batch_size, args.nepochs)

elif args.approach == 'ewc_cpr' or args.approach == 'mas_cpr' or args.approach == 'rwalk_cpr':
    log_name = '{}_{}_{}_{}_lamb_{}_lr_{}_cpr_beta_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed, 
                                                                       args.lamb, args.lr, args.cpr_beta,
                                                                             args.batch_size, args.nepochs)
    print ('seed : ', args.seed)
elif args.approach == 'ags_cl_cpr':
    log_name = '{}_{}_{}_{}_lamb_{}_mu_{}_lr_{}_cpr_beta_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach,args.seed, 
                                                                       args.lamb, args.mu, args.lr, args.cpr_beta,
                                                                             args.batch_size, args.nepochs)

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)


########################################################################################################################
# Split
args.split = False
split_experiment = [
    'split_cifar10',
    'split_cifar100',
    'split_cifar10_100',
    'split_cifar100_10',
    'split_cifar50_10_50',
    'split_CUB200_new',
    'omniglot',
]

conv_experiment = [
    'split_cifar10',
    'split_cifar100',
    'split_cifar10_100',
    'split_cifar100_10',
    'split_cifar50_10_50',
    'split_CUB200_new',
    'omniglot',
]

if args.experiment in split_experiment:
    args.split = True
if args.experiment in conv_experiment:
    args.conv = True
    log_name = log_name + '_conv'
elif args.model == 'Resnet18':
    log_name = log_name + '_Resnet18'

if args.output == '':
    args.output = './result_data/' + log_name + '.txt'
else:
    args.output = './result_data/' + log_name + args.output + '.txt'
    
# Seed
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print('[CUDA unavailable]'); sys.exit()

# Args -- Experiment
if args.experiment == 'split_cifar10':
    from dataloaders import split_cifar10 as dataloader
elif args.experiment == 'split_cifar100':
    from dataloaders import split_cifar100 as dataloader
elif args.experiment == 'split_cifar10_100':
    from dataloaders import split_cifar10_100 as dataloader
elif args.experiment == 'split_cifar100_10':
    from dataloaders import split_cifar100_10 as dataloader
elif args.experiment == 'split_cifar50_10_50':
    from dataloaders import split_cifar50_10_50 as dataloader
elif args.experiment == 'split_CUB200_new':
    from dataloaders import split_CUB200_new as dataloader
elif args.experiment == 'omniglot':
    from dataloaders import split_omniglot as dataloader

# Args -- Approach
if args.approach == 'finetuning':
    from approaches import finetuning as approach
elif args.approach == 'ags_cl_cpr':
    if args.model != 'Resnet18':
        from approaches import ags_cl_cpr as approach
    else:
        from approaches import ags_cl_cpr_resnet18 as approach
elif args.approach == 'ewc_cpr':
    from approaches import ewc_cpr as approach
elif args.approach == 'si_cpr':
    from approaches import si_cpr as approach
elif args.approach == 'rwalk_cpr':
    from approaches import rwalk_cpr as approach
elif args.approach == 'mas_cpr':
    from approaches import mas_cpr as approach
elif args.approach == 'joint':
    from approaches import joint as approach

if args.experiment == 'split_cifar100' or args.experiment == 'split_cifar10_100' or args.experiment == 'split_cifar100_10' or args.experiment == 'split_cifar50_10_50':
    
    from networks import conv_net as network

elif args.experiment == 'split_CUB200_new':
    
    if args.model =='Resnet18':
        from networks import resnet18 as network
        
elif args.experiment == 'omniglot':
    
    from networks import conv_net_omniglot as network

########################################################################################################################


# Load
print('Load data...')
data, taskcla, inputsize = dataloader.get(seed=args.seed, tasknum=args.tasknum)
print('\nInput size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = network.Net(inputsize, taskcla).cuda()
appr = approach.Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args,
                     log_name=log_name, split=args.split)

utils.print_model_report(net)
print('-' * 100)
relevance_set = {}
# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    
for t, ncla in taskcla:
    if t==args.tasknum:
        break

    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    if args.approach == 'joint':
        # Get data. We do not put it to GPU
        if t == 0:
            xtrain = data[t]['train']['x'].cuda()
            ytrain = data[t]['train']['y'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
            yvalid = data[t]['valid']['y'].cuda()
            task_t = t * torch.ones(xtrain.size(0)).int().cuda()
            task_v = t * torch.ones(xvalid.size(0)).int().cuda()
            task = [task_t, task_v]
        else:
            xtrain = torch.cat((xtrain, data[t]['train']['x'].cuda()))
            ytrain = torch.cat((ytrain, data[t]['train']['y'].cuda()))
            xvalid = torch.cat((xvalid, data[t]['valid']['x'].cuda()))
            yvalid = torch.cat((yvalid, data[t]['valid']['y'].cuda()))
            task_t = torch.cat((task_t, t * torch.ones(data[t]['train']['y'].size(0)).int().cuda()))
            task_v = torch.cat((task_v, t * torch.ones(data[t]['valid']['y'].size(0)).int().cuda()))
            task = [task_t, task_v]
    else:
        # Get data
        
        if 'split_CUB200' not in args.experiment:
            xtrain = data[t]['train']['x'].cuda()
            xvalid = data[t]['valid']['x'].cuda()
        else:
            xtrain = data[t]['train']['x']
            xvalid = data[t]['valid']['x']

        ytrain = data[t]['train']['y'].cuda()
        yvalid = data[t]['valid']['y'].cuda()
        task = t

    # Train
    if args.approach != 'joint':
        appr.train(task, xtrain, ytrain, xvalid, yvalid, data, inputsize, taskcla)
    else:
        appr.train(task, xtrain, ytrain, xvalid, yvalid)
    print('-' * 100)

    # Test
    for u in range(t + 1):
        if 'split_CUB200' not in args.experiment:
            xtest = data[u]['test']['x'].cuda()
        else:
            xtest = data[u]['test']['x']
            
        ytest = data[u]['test']['y'].cuda()

        print ('not in args.approach')
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss,
                                                                                  100 * test_acc))
        lss[t, u] = test_loss
        acc[t, u] = test_acc
        
    # Save

    print('Save at ' + args.output)
    np.savetxt(args.output, acc, '%.4f')
    torch.save(net.state_dict(), './weights/' + log_name + '_task_{}.pt'.format(t))
    
print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

if hasattr(appr, 'logs'):
    if appr.logs is not None:
        # save task names
        from copy import deepcopy

        appr.logs['task_name'] = {}
        appr.logs['test_acc'] = {}
        appr.logs['test_loss'] = {}
        for t, ncla in taskcla:
            appr.logs['task_name'][t] = deepcopy(data[t]['name'])
            appr.logs['test_acc'][t] = deepcopy(acc[t, :])
            appr.logs['test_loss'][t] = deepcopy(lss[t, :])
        # pickle
        import gzip
        import pickle

        with gzip.open(os.path.join(appr.logpath), 'wb') as output:
            pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)
########################################################################################################################
