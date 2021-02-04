
import torch
from torch.autograd import Variable
dtype = torch.cuda.FloatTensor  # run on GPU
import sys

class VOWM:

    def __init__(self,  shape, alpha=0, beta=0):

        self.input_size = shape[0]
        self.output_size = shape[1]
        self.alpha = alpha
        self.beta = beta

        #the projection operator of A
        self.P_owm = Variable((1.0 / self.alpha) * torch.eye(self.input_size).type(dtype), volatile=True)

        #the orthogonal projection operator of Ω
        self.Q_ort = Variable((1.0 / self.beta) * torch.eye(self.input_size).type(dtype), volatile=True)

    def VOWM_learn(self, w, input_, learning_rate, alpha=1.0, beta = 1.0, lambda1 = 0.2, lambda2 = 25, mode='ps'):

        #update P using RLS
        self.rp_owm = torch.mean(input_, 0, True)
        self.kp_owm = torch.mm(self.P_owm, torch.t(self.rp_owm))
        self.cp_owm = 1.0 / (alpha + torch.mm(self.rp_owm, self.kp_owm))
        self.P_owm = torch.sub(self.P_owm, self.cp_owm * torch.mm(self.kp_owm, torch.t(self.kp_owm)))

        #update Q using RLS
        out_ch,_ = w.shape
        w_ = w.view(out_ch,-1).detach().clone()
        rq_ort = torch.mean(w_,1,True)
        kq_ort = torch.mm(self.Q_ort, rq_ort)
        self.Q_ort = self.Q_ort - (torch.mm(kq_ort, torch.t(kq_ort)) / (beta + torch.mm(torch.t(rq_ort), kq_ort)))
        I = torch.eye(self.Q_ort.shape[0]).cuda()

        #the projection operator of Ω
        Q_para = torch.sub(I, self.Q_ort)


        if mode == 'ps':
            twdir = lambda1 * self.P_owm + lambda2 * torch.mm(self.P_owm, Q_para)
        elif mode == "pds":
            twdir = lambda1 * self.P_owm + lambda2 * torch.mm(self.P_owm, self.Q_ort)
        else:
            print("mode error")
            sys.exit()

        w.data -= learning_rate * torch.mm(twdir.data, w.grad.data)

        w.grad.data.zero_()


    def predit_lable(self, input_, w,):
        return torch.mm(input_, w)
