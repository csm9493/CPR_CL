import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Continual')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--experiment', default='pmnist', type=str, required=False,
                        choices=['omniglot',
                                 'split_cifar10_100',
                                 'split_cifar100_10',
                                 'split_cifar50_10_50',
                                 'split_cifar100',
                                 'split_CUB200_new',],
                        help='(default=%(default)s)')
    parser.add_argument('--approach', default='lrp', type=str, required=False,
                        choices=['finetuning',
                                 'ewc_cpr',
                                 'si_cpr',
                                 'rwalk_cpr',
                                 'mas_cpr',
                                 'joint',
                                 'ags_cl_cpr',],
                        help='(default=%(default)s)')
    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD',
                                 'SGD_momentum_decay',
                                 'Adam'],
                        help='(default=%(default)s)')
    parser.add_argument('--model', default='Conv', type=str, required=False,
                    choices=[ 'Resnet18', 'Conv'],
                    help='(default=%(default)s)')
    
    parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--nepochs', default=100, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--batch-size', default=256, type=int, required=False, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.001, type=float, required=False, help='(default=%(default)f)')
    
    parser.add_argument('--lamb', default=1, type=float, help='(default=%(default)f)')
    parser.add_argument('--nu', default=0.9, type=float, help='(default=%(default)f)')
    parser.add_argument('--mu', default=1, type=float, help='groupsparse parameter')
    parser.add_argument('--c', default=1, type=float, help='(default=%(default)f)')
    parser.add_argument('--rho', type = float, default=-2.783, help='initial rho')
    parser.add_argument('--cpr-beta', default=0.0, type=float, help='(default=%(default)f)')
    
    parser.add_argument('--date', type=str, default='', help='(default=%(default)s)')
    parser.add_argument('--tasknum', default=10, type=int, help='(default=%(default)s)')
    parser.add_argument('--conv-net', action='store_true', default=False, help='Using convolution network')

    args=parser.parse_args()
    return args

