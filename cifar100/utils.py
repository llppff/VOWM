import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import torchvision
import math

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


#get and save model
def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


#get coefficients
def w1_count(lam,sim,process):
    if type(sim)==int:
        return (sim+(lam**process)*math.exp(sim))/(sim+math.exp(sim))
    else:
        return (sim.numpy().item()+(lam**process)*math.exp(sim.numpy().item()))/(sim.numpy().item()+math.exp(sim.numpy().item()))
def w2_count(w1,total,sim,flag):

    if type(sim)==int:
        return (1-w1)*(1-flag*pow((1+sim),-total))
    else:
        return (1-w1)*(1-flag*pow((1+sim.numpy().item()),-total))


#divide data into different tasks according to classes
def get_cifar100_by_cls_num(t_num=10):
    root = "../cifar100_ewc/data"
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    dat={}
    dat['train'] = torchvision.datasets.cifar.CIFAR100(root, train=True,
                                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean,std)]),
                                                     download=False)
    dat['test'] = torchvision.datasets.cifar.CIFAR100(root, train=False,
                                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean,std)]),
                                                     download=False)
    task_num = 100 // t_num
    data={}
    for i in range(task_num):
        data[i] = {'train':{'x':[],'y':[]},'test':{'x':[],'y':[]},'ncla':t_num,'name':'cifar100-'+str(i * t_num)+'---'+str((i + 1) * t_num - 1)}
        for s in ['train','test']:
            loader = torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
            for image,target in loader:
                label = target.numpy()[0]
                if label in range(i * t_num, (i + 1) * t_num):
                    data[i][s]['x'].append(image)
                    data[i][s]['y'].append(target)

            data[i][s]['x'] = torch.squeeze(torch.cat(data[i][s]['x'],0),1)
            data[i][s]['y'] = torch.cat(data[i][s]['y'],0)
    data[task_num] = {'train': {'x': [], 'y': []}, 'test': {'x': [], 'y': []}, 'ncla': 100}
    for i in range(task_num):
        for s in ['train', 'test']:
            if i != 0:
                data[task_num][s]['x'] = torch.cat((data[task_num][s]['x'], data[i][s]['x']), 0)
                data[task_num][s]['y'] = torch.cat((data[task_num][s]['y'], data[i][s]['y']), 0)
            else:
                data[task_num][s]['x'] = data[0][s]['x']
                data[task_num][s]['y'] = data[0][s]['y']

    taskcla = []
    n = 0
    for t in range(task_num):
        taskcla.append((t,data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    size = [3, 32, 32]
    return data,taskcla,size