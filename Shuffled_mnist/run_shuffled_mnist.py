import os,argparse
import sys

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from shuffled_mnist.VOWM import VOWM
dtype = torch.cuda.FloatTensor  # run on GPU

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


#initialize hyper-parameters by command-line arguments
parser=argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default= 30, help='(default=%(default)d)')
parser.add_argument('--epoch', default=30, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--lam', default=0.99, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--alpha', default=1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--beta', default=1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--gpu', default="1", type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--mode1', default="ps", type=str, required=False, help='choices=%(choices)s')
parser.add_argument('--mode2', default="ps", type=str, required=False, help='choices=%(choices)s')
parser.add_argument('--batch_size', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--regular_loss', default=1e-3, type=float,required=False, help='(default=%(default)d)')
parser.add_argument('--Task_num', default=10, type=int, required=False, help='(default=%(default)d)')

args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # use gpu

#output hyper-parameters
print(args)

#seed
seed_num = args.seed
np.random.seed(seed_num)
torch.manual_seed(seed_num)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_num)
else:
    print('[CUDA unavailable]')
    sys.exit()

#setting hyper-parameters
class_num = 10
num_epochs = args.epoch
batch_size = args.batch_size
learning_rate = args.lr
Task_num = args.Task_num

#get mnist data
train_dataset = datasets.MNIST(root='/home/liyanni/1307/ykc/mnist_owm/data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='/home/liyanni/1307/ykc/mnist_owm/data/', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


####################Preparation for training####################


#divide data into different tasks according to classes
def get_divide_data(train_loader,test_loader):
    data={}
    print("Prepare train data!")
    for task_index in range(Task_num):
        data[task_index]={}
        ss = np.arange(28*28)
        if task_index > 0:
            np.random.seed(task_index)
            np.random.shuffle(ss)
        for s in ['train','test']:
            data[task_index][s]={'x': [], 'y': []}
            if s=='train':
                for i, (images, labels) in enumerate(train_loader):
                    labels = Variable(labels)
                    images = Variable(images)
                    images = images.view(-1, 28 * 28)
                    numpy_data = images.data.cpu().numpy()
                    input = torch.from_numpy(numpy_data[:, ss])

                    data[task_index][s]['x'].append(input)
                    data[task_index][s]['y'].append(labels)
            else:
                for i, (images, labels) in enumerate(test_loader):
                    labels = Variable(labels)
                    images = Variable(images)
                    images = images.view(-1, 28 * 28)
                    numpy_data = images.data.cpu().numpy()
                    input = torch.from_numpy(numpy_data[:, ss])

                    data[task_index][s]['x'].append(input)
                    data[task_index][s]['y'].append(labels)
            data[task_index][s]['x']=torch.cat(data[task_index][s]['x'],dim=0)
            data[task_index][s]['y']=torch.cat(data[task_index][s]['y'],dim=0)
    return data



#calculate similarity
def get_cos_sim(data,tasknum,sample_num,mode='mean'):
    simlirity=[[0]*tasknum]*tasknum

    sample_data=[]
    for t in range(tasknum):
        samples = np.random.choice(
            data[t]['train']['x'].shape[0], size=(sample_num)
        )
        samples = torch.LongTensor(samples)
        cur=torch.cat([data[t]['train']['x'][samples[j]].view(1,-1) for j in range(sample_num)])
        if mode=='mean':
            cur=torch.mean(cur,dim=1)
        else:
            cur=cur.view(1,-1)
        sample_data.append(cur)

    for i in range(tasknum):
        for j in range(0,i):
            simlirity[i][j]=0.5*cos(sample_data[i],sample_data[j])+0.5
    return simlirity

#calculate cos
def cos(tensor_1, tensor_2):
    norm_tensor_1=tensor_1.norm(p=2,dim=-1, keepdim=True)
    norm_tensor_2=tensor_2.norm(p=2,dim=-1, keepdim=True)
    normalized_tensor_1 = tensor_1 / norm_tensor_1
    normalized_tensor_2 = tensor_2 / norm_tensor_2
    return (normalized_tensor_1*normalized_tensor_2).sum(dim=-1)



#preparations for network
def get_weight(shape, zeros=None):
    np.random.seed(seed_num)
    if zeros is None:
        w = np.random.normal(0, 1.0, shape)
        w = torch.from_numpy(w/(np.sqrt(sum(shape)/2.0)))
    else:
        w = np.zeros(shape)
        w = torch.from_numpy(w)
    return Variable(w.type(dtype), requires_grad=True)

def get_bias(shape):
    bias = 0.01 * np.random.rand(shape)
    bias = torch.from_numpy(bias)
    return Variable(bias.type(dtype), requires_grad=True)

def get_layer(shape, alpha=0, beta=0, zeros=None):
    w = get_weight(shape, zeros)
    return w, VOWM(shape, alpha, beta)



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


#model constructions
alpha = args.alpha
beta = args.beta
w1, layer1 = get_layer([28 * 28, 100], alpha=args.alpha, beta=args.beta)
b1 = get_bias(w1.size(1))
wo, layer_out = get_layer([100, class_num], alpha=args.alpha, beta=args.beta)
activationFunc = nn.ReLU().cuda()
CELoss = nn.CrossEntropyLoss().cuda()
regular_loss = args.regular_loss

#initialization
norm = 0
final_data=get_divide_data(train_loader,test_loader)
sim=get_cos_sim(final_data,Task_num,10*batch_size,mode='mean')
total_batches = len(train_dataset) // batch_size


#####################Train#####################
for t_id in range(Task_num):
    print("\n","="*20,"current task:{}".format(t_id),"="*20,"\n")

    cur_train_data=final_data[t_id]['train']['x']
    cur_train_label=final_data[t_id]['train']['y']
    train_data_size = np.arange(cur_train_data.size(0))
    if t_id > 0:
        np.random.shuffle(train_data_size)

    total_cur_batch=0

    #train_epoch
    for epoch in range(num_epochs):
        for cur_batch_for_epoch in range(0, len(train_data_size), batch_size):

            lambda1 = w1_count(args.lam, sim[t_id][t_id - 1], cur_batch_for_epoch / (len(train_data_size) / batch_size))
            lambda2 = w2_count(args.lam, Task_num, sim[t_id][t_id - 1], 1)

            #shuffle the mnist data and forward
            b = train_data_size[cur_batch_for_epoch:min(cur_batch_for_epoch + batch_size, len(train_data_size))]
            labels = Variable(cur_train_label[b]).cuda()
            inputs = Variable(cur_train_data[b]).cuda()
            out = activationFunc(inputs.mm(w1) + b1)

            y_pred = out.mm(wo)

            #backward the error and calculate the gradient
            loss = CELoss(y_pred, labels) + regular_loss * (torch.norm(w1) + torch.norm(wo))
            loss.backward()

            #modify the gradients and update the weights
            with torch.no_grad():
                layer1.VOWM_learn(w1, inputs, learning_rate, args.alpha, args.beta, lambda1, lambda2, args.mode1)
                layer_out.VOWM_learn(wo, out, learning_rate, args.alpha, args.beta, lambda1, lambda2, args.mode2)

            
            norm = torch.norm(wo).data[0]
            total_cur_batch+=1
            if ((total_cur_batch + 1) % (len(train_dataset) // batch_size)) == 0:
                print('Task [{:d}/{:d}], Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.3f} Norm: {:.3f}'
                      .format(t_id + 1, Task_num, epoch + 1, num_epochs, total_cur_batch + 1, total_batches,
                              loss.data[0], norm))

    #test all previous tasks after training current task
    all_accuracy = []
    for test_id in range(Task_num):
        cur_test_data = final_data[test_id]['test']['x']
        task_test_label = final_data[test_id]['test']['y']
        train_data_size = np.arange(cur_test_data.size(0))
        np.random.shuffle(train_data_size)

        correct_num = 0
        total_data_num = 0
        for cur_batch_for_epoch in range(0, len(train_data_size), batch_size):
            b = train_data_size[cur_batch_for_epoch:min(cur_batch_for_epoch + batch_size, len(train_data_size))]
            labels = Variable(task_test_label[b])
            inputs = Variable(cur_test_data[b]).cuda()
            out = activationFunc(inputs.mm(w1) + b1)

            y_pred = out.mm(wo)

            _, predictions = torch.max(y_pred.data, 1)
            total_data_num += labels.size(0)
            correct_num += (predictions.cpu() == labels).sum().numpy()
        all_accuracy.append((100 * correct_num / total_data_num))

    print(all_accuracy)
    print("after train for task {}，".format(t_id), end="")
    print("average Test Accuracy on All Tasks: {0:.2f} %".format(np.sum(all_accuracy) / len(all_accuracy)))

print("train finished")


#test all tasks after training
all_accuracy = []
for t_id in range(Task_num):
    cur_test_data=final_data[t_id]['test']['x']
    task_test_label=final_data[t_id]['test']['y']
    train_data_size = np.arange(cur_test_data.size(0))
    np.random.shuffle(train_data_size)

    correct_num = 0
    total_data_num = 0
    for cur_batch_for_epoch in range(0, len(train_data_size), batch_size):
        b = train_data_size[cur_batch_for_epoch:min(cur_batch_for_epoch + batch_size, len(train_data_size))]
        labels = Variable(task_test_label[b])
        inputs = Variable(cur_test_data[b]).cuda()
        out = activationFunc(inputs.mm(w1) + b1)

        y_pred = out.mm(wo)

        _, predictions = torch.max(y_pred.data, 1)
        total_data_num += labels.size(0)
        correct_num += (predictions.cpu() == labels).sum().numpy()
    all_accuracy.append((100 * correct_num / total_data_num))
    print('Test Accuracy of the model on the 10000 Shuffled_mnist images: %0.2f %%' % (100 * correct_num / total_data_num))

print(all_accuracy)
print("Aver Test Accuracy on All Tasks: {0:.2f} %".format(np.sum(all_accuracy) / len(all_accuracy)))

