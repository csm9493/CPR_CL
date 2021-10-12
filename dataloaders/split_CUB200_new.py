import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
from scipy.misc import imread
from torch import Tensor


def split_cub200_loader(root):
    
    data = {}
    for i in range(10):
        data[i] = {}
        data[i]['name'] = 'split_cub200-{:d}'.format(i)
        data[i]['ncla'] = 20
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}
        data[i]['valid'] = {'x': [], 'y': []}

    folders = sorted(os.listdir(root+'/images'))
    with open(root+'/train_test_split.txt', 'r') as f:
        train_test_split = f.read().splitlines()
    
    mean = np.array([[[123.77, 127.55, 110.25]]])
    std = np.array([[[59.16, 58.06, 67.99]]])
    
    idx = -1
    label_true = -1
    
    for folder in folders:
        
        folder_path = os.path.join(root+'/images', folder)
        img_list = sorted(os.listdir(folder_path))
        label_true += 1
        
        img_len = len(img_list)
        tr_num = int(len(img_list) * 0.7)
        val_num = int(len(img_list) * 0.1)
        te_num = int(len(img_list) * 0.2)
        
        folder_img_idx = 0
        
        for ims in img_list:
            idx += 1
            img_path = os.path.join(folder_path, ims)
            img = imread(img_path)
            try:
                H,W,C = img.shape
                
                if H < 224 or W < 224:
                    continue
            except:
                continue
                
            if folder_img_idx < tr_num:
                s = 'train'
                print ('train', img.shape)
            elif folder_img_idx < tr_num + val_num:
                s = 'valid'
                print ('valid', img.shape)
            else:
                s = 'test'
                print ('test', img.shape)
                
            img = (img - mean) / std
            img_tensor = Tensor(img).float()
            task_idx = label_true // 20
            label = label_true % 20
            data[task_idx][s]['x'].append(img_tensor)
            data[task_idx][s]['y'].append(label) 
            
            folder_img_idx += 1
    
    return data

def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 10):
    
    data = {}
    taskcla = []
    size = [3, 214, 214]
    
    # Pre-load
    # mini_imagenet
    if not os.path.isdir('data/binary_split_cub200_new/'):
        os.makedirs('data/binary_split_cub200_new')
        root = os.path.dirname(__file__)
        data = split_cub200_loader(os.path.join(root, 'data/CUB_200_2011'))
        
        for i in range(10):
            for s in ['train', 'test', 'valid']:
#                 data[i][s]['x']=torch.stack(data[i][s]['x']).view(-1,size[0],size[1],size[2])
#                 data[i][s]['x']=torch.stack(data[i][s]['x'])
                data[i][s]['y']=torch.LongTensor(np.array(data[i][s]['y'],dtype=int)).view(-1)
                torch.save(data[i][s]['x'],os.path.join(os.path.expanduser('data/binary_split_cub200_new'),
                                                        'data' + str(i) + s + 'x.bin'))
                torch.save(data[i][s]['y'],os.path.join(os.path.expanduser('data/binary_split_cub200_new'),
                                                        'data' + str(i) + s + 'y.bin'))
    else:
        data[0] = dict.fromkeys(['name','ncla','train','test'])
        ids=list(shuffle(np.arange(10),random_state=seed))
        print('Task order =',ids)
        for i in range(10):
            data[i] = dict.fromkeys(['name','ncla','train','test'])
            for s in ['train','test', 'valid']:
                data[i][s]={'x':[],'y':[]}
                data[i][s]['x'] = torch.load(os.path.join(os.path.expanduser('data/binary_split_cub200_new'),
                                                          'data' + str(i) + s + 'x.bin'))
                data[i][s]['y'] = torch.load(os.path.join(os.path.expanduser('data/binary_split_cub200_new'),
                                                          'data' + str(i) + s + 'y.bin'))
            data[i]['ncla']=len(np.unique(data[i]['train']['y'].numpy()))
            data[i]['name']='split_cub200-'+str(ids[i-1])
        
    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data,taskcla, size