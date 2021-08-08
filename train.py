import torch
from TD3_net_spread import TD3
import ReplayBuffer
from environment import World
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--User_num", help="set the User Number", type=int,default=10)
    parser.add_argument("--uav_h", help="set the UAV height", type=float,default=0.95)
    parser.add_argument("--gamma", help="set the gamma", type=float,default=0.99)
    parser.add_argument("--buffer", help="set the buffer", type=int,default=100000)
    parser.add_argument("--net_width", help="set the net width",type=int,default=400)

    return parser


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

parser = create_parser()
args = parser.parse_args()
gamma = args.gamma
net_width = args.net_width
buffer = args.buffer
user_num = args.User_num
uav_h = args.uav_h
train_path = 'logs/exp'+str(user_num)+'/'
mkdir(train_path)
writer = SummaryWriter(train_path)
result_path = 'Result_'+str(user_num)+'/'
mkdir(result_path[:-1])
# train
np.random.seed(1)
torch.manual_seed(1)

uav_num = 1
T = 200
Length = 10    #1km,  100 m per 1
Width = 10
s_dim = uav_num*2 + user_num*2 + 1   # [b_1,...b_K,c_1,...,c_K,x,y,zeta]
a_dim = uav_num*2  # [theta, upsilon]
# In this paper, we consider the flight time slot in each step is a constant.
# In this case, the flight speed is completely equivalent to the flight distance variable.
# Thus, in the simulation, we use the flight distance as the action.
# You also can use the flight speed as the action directly.
dist_max = 0.5
max_action = np.array([math.pi,dist_max/2])  # -pi,pi; -25m, 25m
total_episode = 8000
sample_episode = 1000
memory_size = T * sample_episode  # The capacity of experience replay buffer
world = World(length=Length, width=Width, uav_num=uav_num, user_num=user_num, t=T, uav_h=uav_h,users_name='Users_'+str(user_num)+'.txt')


expl_noise = 0.6  # random noise std
t1 = time.time()

env_with_Dead = True
kwargs = {
    "env_with_Dead": env_with_Dead,
    "state_dim": s_dim,
    "action_dim": a_dim,
    "max_action": max_action,
    "user_num":user_num,
    "train_path":train_path,
    "gamma": gamma,
    "net_width": net_width,
    "a_lr": 1e-4,
    "c_lr": 1e-4,
    "Q_batchsize": 256,
}
model = TD3(**kwargs)
model_path = 'Model_'+str(user_num)+'/'
mkdir(model_path[:-1])
replay_buffer = ReplayBuffer.ReplayBuffer(s_dim, a_dim, max_size=int(buffer))

x = y = np.arange(0,Width,0.01)
x, y = np.meshgrid(x,y)


def draw_location(x_uav, y_uav,t, x_user, y_user, savepath,s):
    x_uav = np.transpose(x_uav)
    y_uav = np.transpose(y_uav)
    plt.figure(facecolor='w', figsize=(20, 20))
    for i in range(user_num):
        if s[user_num+i]==1.0:
            plt.scatter(x_user[i], y_user[i], c='red', marker='x', s=150, linewidths=4)
        else:
            plt.scatter(x_user[i], y_user[i], c='black', marker='x', s=150, linewidths=4)
    for i in range(uav_num):
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], c='blue', marker='.', linewidth=3.5, markersize=7.5)
        plt.plot(x_uav[i][t], y_uav[i][t], c='green', marker='.', markersize=12.5)
    for index in range(world.urban_world.Build_num):
        x1 = world.HeightMapMatrix[index][0]
        x2 = world.HeightMapMatrix[index][1]
        y1 = world.HeightMapMatrix[index][2]
        y2 = world.HeightMapMatrix[index][3]
        XList = [x1, x2, x2, x1, x1]
        YList = [y1, y1, y2, y2, y1]
        plt.plot(XList, YList, 'r-')
    plt.title('location', fontsize=30)
    plt.xlim((0, Length))
    plt.ylim((0, Width))
    plt.grid()
    plt.savefig(savepath)
    plt.close()


x0_user = np.zeros(world.user_num)
y0_user = np.zeros(world.user_num)
x0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
y0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
z0_uav = np.zeros([total_episode+1, T + 1, world.uav_num])
location = np.zeros([3, world.uav_num])


for episode in range(1, total_episode+1):
    t3 = time.time()
    s, t = world.reset()
    for i, user in enumerate(world.Users):
        x0_user[i] = user.x
        y0_user[i] = user.y
    for i, uav in enumerate(world.UAVs):
        location[0][i] = uav.x
        location[1][i] = uav.y
        location[2][i] = uav.h
    x0_uav[episode][0] = location[0]
    y0_uav[episode][0] = location[1]
    z0_uav[episode][0] = location[2]
    sum_reward = 0
    actions = np.zeros(2 * uav_num)
    done = False
    expl_noise *= 0.999
    while not done:
        a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=a_dim)
             ).clip(-max_action, max_action)  # obtain a new action
        s_, r, done,t,terminal = world.step_inside(a, s, t)
        sum_reward += r
        replay_buffer.add(s, a, r, s_, terminal)  # put a transition in buffer

        if replay_buffer.size > 2000: model.train(replay_buffer)

        for i, uav in enumerate(world.UAVs):
            location[0][i] = uav.x
            location[1][i] = uav.y
            location[2][i] = uav.h
        x0_uav[episode][t] = location[0]
        y0_uav[episode][t] = location[1]
        z0_uav[episode][t] = location[2]
        s = s_  # update state
        #if done and agent.pointer <= memory_size: # and episode % 5 == 0:
        #    print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'reward: ', r,'fa',world.fa,'raw_r',world.r,'num_fa2',num_fa2, 'Explore:%.2f'% epsino)
        #    print('sum_cover:',world.sum_cover)
        #if done and agent.pointer > memory_size: #and episode % 5 == 0:
    print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'Explore:%.2f'% expl_noise)
    #print('loss_a:', agent.loss_a, 'td_error:', agent.td_error)
    print('sum_cover:',world.sum_cover)
    writer.add_scalar('total_reward', sum_reward, episode)
  #  if agent.pointer > memory_size:
    #draw_location(x0_uav[episode], y0_uav[episode],t, x0_user, y0_user,
    #             result_path + 'UAVPath_Users_%s.png' % str(episode).zfill(2),s)
    if episode % 500 == 0 and episode >= 2000:
        model.save(episode,model_path)
       # torch.save(agent.Critic_eval.state_dict(), 'Critic_model' + str(episode) + '_' + str(user_num) + '.pkl')
    print(time.time()-t3)
print('Running time:', time.time() - t1)
