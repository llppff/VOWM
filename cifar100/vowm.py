import sys, time
import numpy as np
import torch

dtype = torch.cuda.FloatTensor  # run on GPU
import utils



class Appr(object):

    def __init__(self, model, nepochs=0, sbatch=64, lr=0,  clipgrad=10, args=None):
        self.model = model

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.clipgrad = clipgrad
        self.args=args
        self.optimizer_conv, self.optimizer_fc = self._get_disjoint_optimizer()
        self.optimizer=self._get_joint_optimizer()

        self.ce = torch.nn.CrossEntropyLoss()

        #the projection operators of A
        self.Pc1 = torch.autograd.Variable(torch.eye(3 * 2 * 2).type(dtype), volatile=True)
        self.Pc2 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype), volatile=True)
        self.Pc3 = torch.autograd.Variable(torch.eye(256 * 2 * 2).type(dtype), volatile=True)
        self.P1 = torch.autograd.Variable(torch.eye(512 * 4 * 4).type(dtype), volatile=True)
        self.P2 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)
        self.P3 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)

        #the orthogonal projection operators of Ω
        self.Qc1 = torch.autograd.Variable(torch.eye(3 * 2 * 2).type(dtype), volatile=True)
        self.Qc2 = torch.autograd.Variable(torch.eye(128 * 2 * 2).type(dtype), volatile=True)
        self.Qc3 = torch.autograd.Variable(torch.eye(256 * 2 * 2).type(dtype), volatile=True)
        self.Q1 = torch.autograd.Variable(torch.eye(512 * 4 * 4).type(dtype), volatile=True)
        self.Q2 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)
        self.Q3 = torch.autograd.Variable(torch.eye(1000).type(dtype), volatile=True)


        self.test_max = 0

        return

    #get optimizer if args.apporach is disjoint
    def _get_disjoint_optimizer(self, t=0, lr=None):
        lr = self.lr
        lr_owm = self.args.clr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
                             self.model.parameters())
        optimizer_conv = torch.optim.SGD([{'params': base_params}], lr=lr, momentum=0.9)

        optimizer_fc = torch.optim.SGD([{'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                        {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                        {'params': self.model.fc3.parameters(), 'lr': lr_owm}], lr=self.args.clr, momentum=0.9)
        return optimizer_conv,optimizer_fc

    #get optimizer if args.apporach is joint
    def _get_joint_optimizer(self, t=0, lr=None):
        lr = self.lr
        lr_owm = self.lr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        fc3_params = list(map(id, self.model.fc3.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params + fc3_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},{'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                          {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                          {'params': self.model.fc3.parameters(), 'lr': lr_owm}], lr=lr, momentum=0.9)
        return optimizer

    def train(self, t, xtrain, ytrain, xvalid, yvalid, data,all_data_pos,similarity):
        #save the best model
        best_model = utils.get_model(self.model)
        lr = self.lr

        if self.args.approach=='disjoint':
            self.optimizer_conv, self.optimizer_fc = self._get_disjoint_optimizer(t, lr)
        else:
            self.optimizer=self._get_joint_optimizer(t, lr)

        #train for epoch
        nepochs = self.nepochs

        try:
            for e in range(nepochs):
                #train
                self.train_epoch(xtrain, ytrain, total=len(similarity),cur_epoch=e, nepoch=nepochs,sim=similarity[t][t-1],t=t)
                train_loss, train_acc = self.eval(xtrain, ytrain)
                print('| [{:d}/5], Epoch {:d}/{:d}, | Train: loss={:.3f}, acc={:2.2f}% |'.format(t + 1, e + 1,
                                                                                                 nepochs, train_loss,
                                                                                                 100 * train_acc),
                      end='')
                #valid current task
                valid_loss, valid_acc = self.eval(xvalid, yvalid)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss, 100 * valid_acc), end='')
                print()

                #test current task
                xtest = data[all_data_pos]['test']['x'].cuda()
                ytest = data[all_data_pos]['test']['y'].cuda()

                _, test_acc = self.eval(xtest, ytest)

                #update the best model
                if test_acc>self.test_max:
                    self.test_max = max(self.test_max, test_acc)
                    best_model = utils.get_model(self.model)

                print('>>> Test on All Task:->>> Max_acc : {:2.2f}%  Curr_acc : {:2.2f}%<<<'.format(100 * self.test_max, 100 * test_acc))

        except KeyboardInterrupt:
            print()

        #restore best model
        utils.set_model_(self.model, best_model)
        return

    #train epoch
    def train_epoch(self, x, y, total=5,cur_epoch=0, nepoch=0,sim=0,t=0):
        self.model.train()
        r_len = np.arange(x.size(0))
        np.random.shuffle(r_len)
        r_len = torch.LongTensor(r_len).cuda()

        process=(cur_epoch)/nepoch

        #gradient of CNN is modified by Pds, gradient of full connected layer is modified by Ps

        #calculate coefficients of Pds
        lambda1=utils.w1_count(self.args.lam,total,sim,process)
        lambda2=utils.w2_count(lambda1,total,sim,-1)

        # calculate coefficients of Ps
        clambda1=utils.w1_count(self.args.clam,total,sim,process)
        clambda2=utils.w2_count(clambda1,total,sim,1)


        
        #train batch
        for i_batch in range(0, len(r_len), self.sbatch):
            b = r_len[i_batch:min(i_batch + self.sbatch, len(r_len))]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)

            #forward the input data
            output, h_list, x_list = self.model.forward(images)
            loss = self.ce(output, targets)

            #backward the error and calculate the gradient by standard BP method
            if self.args.approach=='disjoint':
                self.optimizer_conv.zero_grad()
                self.optimizer_fc.zero_grad()
            else:
                self.optimizer.zero_grad()
            loss.backward()
            decay = ((i_batch / len(r_len))/nepoch + (cur_epoch/nepoch))

            alpha_array = [1.0 * 0.0001 ** decay, 1.0 * 0.0001 ** decay, 1.0 * 0.0001 ** decay, 1.0 * 0.2 ** decay]
            beta_array = [10.0 * 0.0001 ** decay, 10.0 * 0.0001 ** decay, 10.0 * 0.0001 ** decay, 10.0 * 0.9 ** decay]

            #update the projection operators
            for n, w in self.model.named_parameters():
                #CNN layer
                if n == 'c1.weight':
                    self.pro_weight(self.Pc1, self.Qc1, x_list[0], w, alpha=alpha_array[0], beta=beta_array[0],stride=2,lambda1=lambda1,lambda2=lambda2,mode=self.args.mode)
                if n == 'c2.weight':
                    self.pro_weight(self.Pc2, self.Qc2, x_list[1], w, alpha=alpha_array[0], beta=beta_array[0], stride=2,lambda1=lambda1,lambda2=lambda2,mode=self.args.mode)
                if n == 'c3.weight':
                    self.pro_weight(self.Pc3, self.Qc3, x_list[2], w, alpha=alpha_array[0], beta=beta_array[0], stride=2,lambda1=lambda1,lambda2=lambda2,mode=self.args.mode)

                #fully connected layer
                if n == 'fc1.weight':
                    self.pro_weight(self.P1, self.Q1, h_list[0], w, alpha=alpha_array[1], beta=beta_array[1], cnn=False,lambda1=lambda1,lambda2=lambda2,mode=self.args.cmode,clambda1=clambda1,clambda2=clambda2)
                if n == 'fc2.weight':
                    self.pro_weight(self.P2, self.Q2, h_list[1], w, alpha=alpha_array[2], beta=beta_array[2], cnn=False,lambda1=lambda1,lambda2=lambda2,mode=self.args.cmode,clambda1=clambda1,clambda2=clambda2)
                if n == 'fc3.weight':
                    self.pro_weight(self.P3, self.Q3, h_list[2], w, alpha=alpha_array[3], beta=beta_array[3], cnn=False,lambda1=lambda1,lambda2=lambda2,mode=self.args.cmode,clambda1=clambda1,clambda2=clambda2)

            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)

            #update the weights
            if self.args.approach=='disjoint':
                self.optimizer_conv.step()
                self.optimizer_fc.step()
            else:
                self.optimizer.step()
        return

    #test or valid
    def eval(self, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        #train batch
        for i in range(0, len(r), self.sbatch):
            b = r[i:min(i + self.sbatch, len(r))]
            images = torch.autograd.Variable(x[b], volatile=True)
            targets = torch.autograd.Variable(y[b], volatile=True)

            #forward the input data, get the prediction, then get acc
            output,  _, _ = self.model.forward(images)
            loss = self.ce(output, targets)
            _, pred = output.max(1)
            hits = (pred % 100 == targets).float()

            #record the result
            total_loss += loss.data.cpu().numpy().item() * len(b)
            total_acc += hits.sum().data.cpu().numpy().item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    #specific implementation of VOWM
    def pro_weight(self,p, q, x, w, alpha=1.0, beta=1.0,cnn=True, stride=1,lambda1=0.5,lambda2=2,mode="pds",clambda1=0.2,clambda2=0.8):
        if cnn:
            #CNN
            # update P using RLS
            _, _, H, W = x.shape
            F, _, HH, WW = w.shape
            S = stride
            Ho = int(1 + (H - HH) / S)
            Wo = int(1 + (W - WW) / S)
            for i in range(Ho):
                for j in range(Wo):
                    # N*C*HH*WW, C*HH*WW = N*C*HH*WW, sum -> N*1
                    r = x[:, :, i * S: i * S + HH, j * S: j * S + WW].contiguous().view(1, -1)
                    k = torch.mm(p, torch.t(r))
                    p.sub_(
                        torch.mm(k, torch.t(k))
                        / (alpha +
                           torch.mm(r, k)))

            # update Q
            q = self.pro_weight_for_w(q,w,beta,cnn)
            if mode=="pds":
                ort_2 = lambda1*p+lambda2*torch.mm(p,q)
                w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(ort_2.data)).view_as(w)
            elif mode=="ps":
                I = torch.autograd.Variable(torch.eye(q.shape[0]).cuda())
                q_c = I.sub_(q)
                #the projection operator of Ω
                para = lambda1*p+lambda2*torch.mm(p,q_c)
                w.grad.data = torch.mm(w.grad.data.view(F, -1), torch.t(para.data)).view_as(w)
        else:
            #fully connected layer
            # update P using RLS
            r = x
            k = torch.mm(p, torch.t(r))
            p.sub_(torch.mm(k, torch.t(k)) / (alpha + torch.mm(r, k)))

            #update Q
            q = self.pro_weight_for_w(q,w,beta,cnn)
            if mode=="pds":
                ort_2 = clambda1*p+clambda2*torch.mm(p,q)
                w.grad.data = torch.mm(w.grad.data, torch.t(ort_2.data))
            elif mode=="ps":
                I = torch.autograd.Variable(torch.eye(q.shape[0]).cuda())
                q_c = I.sub_(q)
                # the projection operator of Ω
                para = clambda1*p+clambda2*torch.mm(p,q_c)
                w.grad.data = torch.mm(w.grad.data, torch.t(para.data))

    #update Q using RLS
    def pro_weight_for_w(self,q,w,beta=1.0,cnn=True):
        if cnn:
            #CNN
            out_ch, in_ch, HH, WW = w.shape
            for i in range(out_ch):
                rq=w[i:i+1,:,:,:].view(1,-1)
                kq = torch.mm(q,torch.t(rq))
                q.sub_(torch.mm(kq,torch.t(kq)) / (beta+torch.mm(rq,kq)))

        else:
            #fully connected layer
            out_ch,_=w.shape
            rq = torch.mean(w,0,True)
            kq = torch.mm(q,torch.t(rq))
            q.sub_(torch.mm(kq,torch.t(kq)) / (beta+torch.mm(rq,kq)))

        return q