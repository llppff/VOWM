import sys, argparse
import os
import numpy as np
import torch
import utils
import datetime
import random
import cifar as dataloader
import owm as approach
import cnn_owm as network


#initialize hyper-parameters by command-line arguments
parser=argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default= 30, help='(default=%(default)d)')
parser.add_argument('--experiment',default='cifar-100', type=str,required=False, help='(default=%(default)s)')
parser.add_argument('--approach', default='disjoint', type=str, required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs', default=30, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--batch-size', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.15, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--clr', default=0.15, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--mode', type=str, default='pds', help='(default=%(default)s)')
parser.add_argument('--cmode', type=str, default='ps', help='(default=%(default)s)')
parser.add_argument('--samples', default=500, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lam', default=0.99, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--clam', default=0.99, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--gpu', type=str, default='1', help='(default=%(default)s)')
parser.add_argument('--cls-num-per-task',type=int, default=50)
parser.add_argument('--simmode',type=str, default='cat')


args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # use gpu

#output hyper-parameters
print('\n')
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':', getattr(args,arg))
print('\n')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('\n')



#seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)



if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()




#get cifar100 data
print('Load data...')
data, taskcla, inputsize = utils.get_cifar100_by_cls_num(args.cls_num_per_task)
print('Input size =', inputsize, '\nTask info =', taskcla)
similarity=dataloader.get_cos_sim(data,len(taskcla),args.samples,mode=args.simmode)




#inits the model and the optimizer
print('Inits...')
net = network.Net(inputsize).cuda()
utils.print_model_report(net)

appr = approach.Appr(net, nepochs=args.nepochs, sbatch = args.batch_size, lr=args.lr, args=args)
utils.print_optimizer_config(appr.optimizer_conv)
utils.print_optimizer_config(appr.optimizer_fc)
print('-'*100)




#####################Train and test#####################
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

for t, ncla in taskcla:
    #t is the ID of tasks, ncla is the number of classes of task_t

    #output the ID of task_t and the name of classes it contains
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t+1, data[t]['name']))
    print('*'*100)

    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['test']['x'].cuda()
    yvalid = data[t]['test']['y'].cuda()

    #do train
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data, len(taskcla),similarity)
    print('-'*100)

    #test all previous tasks after training tasks_t
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss


#output the test accuracy after training
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100*acc[i, j]),end='')
    print()

#output the average acc
print('*'*100)
print("average acc:{}%".format(100*acc.mean(1)[-1]))
print('Done!')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print('='*100)