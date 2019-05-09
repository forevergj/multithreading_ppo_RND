import torch
import torch.nn as nn
import numpy as np
import threading, queue
import math
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
epsilongrnd = False
class Actor(nn.Module):
    def __init__(self,n_action,n_features=None,discret = True):
        super(Actor, self).__init__()
        self.n_features = n_features
        self.n_acion = n_action
        self.discret= discret#是否是离散动作
            #输入为图像
        if n_features==None:
            outlayer = 100
            if self.discret==True:

                self.actldicret = nn.Sequential(nn.Conv2d(3,6,kernel_size=(3,3)),
                                                nn.ELU(),
                                                nn.MaxPool2d(kernel_size=(2,2)),
                                                nn.Conv2d(6, 12, kernel_size=(3, 3)),
                                                nn.ELU(),
                                                nn.MaxPool2d(kernel_size=(2, 2)),
                                                )
                self.actsoftmax = nn.Sequential(nn.Linear(outlayer, self.n_acion),
                                                nn.Softmax(dim=1))
            else:
                self.actldicret = nn.Sequential(nn.Conv2d(3, 6, kernel_size=(3, 3)),
                                                nn.ELU(),
                                                nn.MaxPool2d(kernel_size=(2, 2)),
                                                nn.Conv2d(6, 12, kernel_size=(3, 3)),
                                                nn.ELU(),
                                                nn.MaxPool2d(kernel_size=(2, 2)),
                                                )
                self.mu =nn.Sequential(nn.Linear(outlayer,self.n_acion),
                                       nn.Tanh()
                                       )
                self.sigma = nn.Sequential(nn.Linear(outlayer,self.n_acion),
                                           nn.Softplus()
                                              )
            #输入为ram特征
        else:
            l1 = 64
            if self.discret == True:
                self.actldicret = nn.Sequential(nn.Linear(n_features,l1),
                                                nn.ReLU(),
                                                nn.Linear(l1,n_action),
                                                nn.Softmax(dim=1)
                                                )
            else:
                self.actldicret = nn.Sequential(nn.Linear(n_features, l1),
                                                nn.ReLU(),
                                                )
                self.mu = nn.Sequential(nn.Linear(l1,self.n_acion),
                                        nn.Tanh()
                                        )
                self.sigma = nn.Sequential(nn.Linear(l1, self.n_acion),
                                           nn.Softplus()
                                           )
    def forward(self,s):
        s = torch.FloatTensor(s)
        if self.n_features==None:
            if self.discret == True:
                s = self.actldicret(s)
                s = torch.flatten(s)
                out = self.actsoftmax(s)
                return out
            else:
                s = self.actl1(s)
                s = torch.flatten(s)
                mu = self.mu(s)
                sigma = self.sigma(s)
                Normal_list = torch.distributions.normal.Normal(loc=mu, scale=sigma)
                return Normal_list  # 返回连续动作的分布
        else:
            if self.discret == True:
                out = self.actldicret(s)
                return out
            else:
                s = self.actl1(s)#actor
                mu = self.mu(s)
                sigma = self.sigma(s)
                Normal_list =torch.distributions.normal.Normal(loc=mu,scale=sigma)
                return Normal_list#返回连续动作的分布
class Critic(nn.Module):
    def __init__(self,n_action,n_features=None,discret = True):
        super(Critic, self).__init__()
        self.n_features = n_features
        self.n_acion = n_action
        self.discret = discret  # 是否是离散动作
        if self.n_features==None:
            discret_outlayer1 =100
            discret_outlayer2=20
            self.criticl1 =nn.Sequential(nn.Conv2d(3, 6, kernel_size=(3, 3)),
                                         nn.ELU(),
                                         nn.MaxPool2d(kernel_size=(2, 2)),
                                         nn.Conv2d(6, 12, kernel_size=(3, 3)),
                                         nn.ELU(),
                                         nn.MaxPool2d(kernel_size=(2, 2)),
                                         )
            self.critic_value = nn.Sequential(nn.Linear(discret_outlayer1, discret_outlayer2),
                                            nn.ReLU(),
                                            nn.Linear(discret_outlayer2,1),
                                            )
        else:
            l1 = 64
            l2 = 32
            self.criticl1 = nn.Sequential(nn.Linear(n_features, l1),
                                            nn.ReLU(),
                                            nn.Linear(l1, l2),
                                            nn.ReLU(),
                                            nn.Linear(l2, 1),
                                            )

    def forward(self,s,r):
        s = torch.FloatTensor(s)
        r = torch.FloatTensor(r)
        if self.n_features==None:
            s = self.criticl1(s)
            s = s.view(s.size(0),-1)
            self.value = self.critic_value(s)
        else:
            self.value = self.criticl1(s)
        adv = r - self.value
        return adv
class PPO1:
    def __init__(self,epsilongrnd=True,IMG_W=None,IMG_H=None,n_features=None,n_action=None,
                 discret=True,drop_rate=0.1, reward_decay=0.95,):
        self.IMG_W = IMG_W
        self.IMG_H = IMG_H
        self.n_features = n_features
        self.n_action = n_action
        self.drop_rate = drop_rate
        self.gamma = reward_decay
        self.discret=discret
        self.epsilongrnd = epsilongrnd
        self.buffer_s=[]
        self.buffer_a=[]
        self.buffer_r=[]
        self.act_net = Actor(n_features=self.n_features,n_action=self.n_action,discret=self.discret)
        self.cri_net = Critic(n_features = self.n_features,n_action = self.n_action)
        self.old_act_net =Actor(n_features = self.n_features,n_action = self.n_action)
        self.METHOD = [
            dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
            dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
        ][1]
        # for m in net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, )  # initial weight
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # optimize all cnn parameters
        # crossentropy = nn.CrossEntropyLoss(reduction='none')
    def choose_action(self,s,rndreward=None):
        s = np.expand_dims(s,axis=0)
        if self.discret ==True:
            prob_weights= self.act_net.forward(s).data.numpy()
            if self.epsilongrnd==False:
                action = np.random.choice(range(prob_weights.shape[1]),
                                          p=prob_weights.ravel())  # select action w.r.t the actions prob

            else:#按照RND回报自适应抖动探索
                a =20#设计函数的超参数
                b =0.1
                rndreward = rndreward.detach().numpy()
                def fun(rndreward,a,b):
                    a = np.exp(-(a * rndreward + b / rndreward - 2 * math.pow(a * b, 1 / 2)))
                    return 1-a
                # print('end',rndreward)
                # print('sheji:',fun(rndreward=rndreward,a=a,b=b))
                if np.random.uniform() < fun(rndreward=rndreward,a=a,b=b):
                    action = np.random.choice(range(prob_weights.shape[1]),
                                              p=prob_weights.ravel())  # select action w.r.t the actions prob
                else:
                    action = np.random.randint(0, self.n_action)
            return action
        else:
            normal = self.act_net(s)
            constant_act = normal.sample()
            return constant_act
    def store_transition(self,state,reward,action):
        self.buffer_s.append(state)
        self.buffer_a.append(action)
        self.buffer_r.append((reward + 8) / 8)
    def get_v(self,s_):
        if s_.ndim < 2: s_ = s_[np.newaxis, :]
        return self.cri_net.criticl1(torch.FloatTensor(s_))
  
class RND(nn.Module):
    def __init__(self,n_features,n_action):
        super(RND,self).__init__()
        l1 = 10
        l2 = 5
        l3 = 16
        self.fixnet = nn.Sequential(nn.Linear(n_features,l1),
                                    nn.Linear(l1,l2),
                                    nn.ReLU())
        self.trainnet=nn.Sequential(nn.Linear(n_features,l1),
                                    nn.Linear(l1, l2),
                                    nn.ReLU())
        self.loss = nn.MSELoss()
    def forward(self, input):
        fix = self.fixnet(input)
        train = self.trainnet(input)
#         loss = torch.mean(torch.abs(fix-train))
        loss = fix-train#维数必须为[batch,1]
        return loss











