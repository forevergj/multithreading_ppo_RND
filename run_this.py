import numpy as np
import gym
from brain_md import *
import torch.nn as nn
import torch
import threading
import queue
# DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = True  # rendering wastes time

batchreward=False#是否选用RND批处理reward
epsilongrnd = False#是否添加epsilongrnd

EP_MAX = 100000
EP_LEN = 500
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0001               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 15            # loop update operation n-steps
EPSILON = 0.2
update_step=10
discret = True
multithreading = True       #如果用multithreding,不能render
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]
GAME=('CartPole-v1')
env = gym.make(GAME)

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
'''
#输入为图片
# img_w = env.observation_space.shape[0]
# img_h = env.observation_space.shape[1]
# img_chanel = env.observation_space.high.shape[2]
# n_action = env.action_space.n
'''
#输入为ram
n_features = env.observation_space.shape[0]
n_action = env.action_space.n
print(n_features)
print(n_action)

class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = global_ppo

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER,should_stop
        should_stop = False

        while should_stop==False:
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                if epsilongrnd==True:
                    s = torch.Tensor(s)
                    rndreward = rnd(s)
                    rndreward = torch.mean(rndreward)
                    # print('rndr:',rndreward)
                    s = s.data.numpy()
                    a = self.ppo.choose_action(s,rndreward)
                else:
                    a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                if done: r = -10
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r-1)  # 0 for not down, -11 for down. Reward engineering
                s = s_
                ep_r += r
                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:

                    if done:
                        v_s_ = 0  # end of episode
                    else:
                        v_s_ = self.ppo.get_v(s_)
                        v_s_ = v_s_.data.numpy()[0,0]
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:,None]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update
                    if GLOBAL_EP >= EP_MAX:
                        should_stop = True# stop training
                        break
                    if done: break
            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r)
def gather_nd(actprob,indices):
    num = actprob.shape[0]
    act = torch.zeros((num))
    for i in range(num):
        actnumber = int(indices[i,1])
        act[i] = actprob[i,actnumber]
    return act
def clip(x,min,max):
    for i in range(x.shape[0]):
        if x[i]<=min:
            x[i]=min
        if x[i]>=max:
            x[i]=max
    return x

def update():
    global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, should_stop
    while should_stop==False:
        if GLOBAL_EP < EP_MAX:
            UPDATE_EVENT.wait()
            global_ppo.old_act_net.load_state_dict(global_ppo.act_net.state_dict())
            data = [QUEUE.get() for _ in range(QUEUE.qsize())]
            data = np.vstack(data)
            s, a, r = data[:, :n_features], data[:, n_features: n_features + 1].ravel(), data[:, -1:]
            #RND training
            st = torch.Tensor(s)
            loss = rnd(st)

            def batch(exp_reward,batchsize = 5):
                exp_reward = np.expand_dims(exp_reward,axis=1)
                for i in range(exp_reward.shape[0]):
                    exp_reward[i]=np.mean(exp_reward[i:i+5])
                return exp_reward
            c = batch(np.mean(loss.data.numpy(), axis=1))
            if batchreward ==True:
                r = r+batch(np.mean(loss.data.numpy(),axis=1))
            else:
                r = r +np.mean(loss.data.numpy(),axis=1)
            RNDmean_loss = torch.mean(loss)
            # print('rndreward:',loss)
            RNDoptimizer = torch.optim.Adam(rnd.trainnet.parameters(), lr=0.001)
            RNDoptimizer.zero_grad()  # clear gradients for this training step
            RNDmean_loss.backward(retain_graph=True)  # backpropagation, compute gradients
            RNDoptimizer.step()
            adv = global_ppo.cri_net.forward(s, r)
            for i in range(update_step):
                actprob = global_ppo.act_net.forward(s)
                old_actprob = global_ppo.old_act_net.forward(s)
                a_indices = np.stack([np.arange(a.shape[0], dtype='int32'), a], axis=1)
                pi_prob = gather_nd(actprob, a_indices)
                oldpi_prob = gather_nd(old_actprob, a_indices)
                ratio = pi_prob / oldpi_prob
                # print(ratio)
                surr = torch.unsqueeze(ratio, dim=1) * adv  # surrogate loss
                # actor training
                if METHOD['name'] == 'kl_pen':
                    ratio =  actprob / old_actprob
                    # print(ratio)
                    surr = ratio * adv  # surrogate loss
                    def kl_divergence(p,q):
                        kl = 0.0
                        for i in range(10):
                            kl += p[i] * torch.log(p[i] / q[i])
                        kl_mean =torch.mean(kl)
                        aloss = -(torch.mean(surr -torch.Tensor([METHOD['lam']]) * kl))
                        return kl_mean,aloss
                    kl_mean,aloss=kl_divergence(pi_prob,oldpi_prob)
                    if kl_mean > 4 * METHOD['kl_target']:  # this in in google's paper
                        break
                    if kl_mean < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                        METHOD['lam'] /= 2
                    elif kl_mean > METHOD['kl_target'] * 1.5:
                        METHOD['lam'] *= 2
                    METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)  # sometimes explode, this clipping is my solution
                else:  # clipping method, find this is better

                    aloss = -torch.mean(torch.min(surr, clip(ratio, 1. - EPSILON, 1. + EPSILON) * adv))
                aoptimizer = torch.optim.Adam(global_ppo.act_net.parameters(), lr=0.0001)
                aoptimizer.zero_grad()  # clear gradients for this training step
                aloss.backward(retain_graph=True)  # backpropagation, compute gradients
                aoptimizer.step()
                #critic training
                closs = torch.mean(adv**2)
                coptimizer = torch.optim.Adam(global_ppo.cri_net.parameters(), lr=0.0001)
                coptimizer.zero_grad()  # clear gradients for this training step
                closs.backward(retain_graph=True)  # backpropagation, compute gradients
                coptimizer.step()
            UPDATE_EVENT.clear()  # updating finished
            GLOBAL_UPDATE_COUNTER = 0  # reset counter
            ROLLING_EVENT.set()

if __name__ == '__main__':
    global_ppo = PPO1(n_features=n_features,n_action=n_action,discret=discret,epsilongrnd=epsilongrnd)#输入为图片删除n_features即可
    rnd = RND(n_features,n_action)
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out

    workers = [Worker(wid=i) for i in range(N_WORKER)]
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    # for m in global_ppo.act_net.modules():
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    threads.append(threading.Thread(target=update))
    threads[-1].start()

    # plot reward change and test
    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    # env = gym.make('CartPole-v0')
    # while True:
    #     s = env.reset()
    #     for t in range(1000):
    #         env.render()
    #         s, r, done, info = env.step(global_ppo.choose_action(s))
    #         if done:
    #             break






