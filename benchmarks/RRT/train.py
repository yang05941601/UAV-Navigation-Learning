
from environment import World
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os
import argparse

def create_parser():
    """
    Parses each of the arguments from the command line
    :return ArgumentParser representing the command line arguments that were supplied to the command line:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--User_num", help="set the User Number", type=int)
    parser.add_argument("--uav_h", help="set the UAV height", type=float)
    return parser


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

parser = create_parser()
args = parser.parse_args()
user_num = args.User_num
uav_h = args.uav_h
#user_num = 10
train_path = 'logs/exp'+str(user_num)+'_'+str(uav_h)
mkdir(train_path)
result_path = 'Result_'+str(user_num)+'/'
mkdir(result_path[:-1])
# train
np.random.seed(1)

uav_num = 1
T = 200
Length = 10    #10km
Width = 10
cover_r = 2.5
s_dim = uav_num*2 + user_num*2 + 1   #无人机的位置和当前覆盖情况和总体的覆盖
a_dim = uav_num*2
dist_max = 0.5
max_action = np.array([math.pi,dist_max/2])
total_episode = 8000
sample_episode = 1000
memory_size = T * sample_episode
world = World(length=Length, width=Width, uav_num=uav_num, user_num=user_num, T=T, uav_h=uav_h, Cover_r=cover_r,users_name='Users_'+str(user_num)+'.txt')


expl_noise = 0.6
decay_rate = 0.9999
var = 0.6
epsino = 1
coverage_set = np.zeros([T + 1, user_num])
energy_set = np.zeros([T + 1, uav_num])
t1 = time.time()


x = y = np.arange(0,Width,0.01)
x, y = np.meshgrid(x,y)


def draw_location(x_uav, y_uav,t, x_user, y_user, savepath,s, gif=False, gif_dir='',
                  op=False, op_xpoint=0.0, op_ypoint=0.0):
    x_uav = np.transpose(x_uav)
    y_uav = np.transpose(y_uav)
    plt.figure(facecolor='w', figsize=(20, 20))
    for i in range(user_num):
        if s[user_num+i]==1.0:
            plt.scatter(x_user[i], y_user[i], c='red', marker='x', s=150, linewidths=4)
        else:
            plt.scatter(x_user[i], y_user[i], c='black', marker='x', s=150, linewidths=4)
  #  for uav in world.UAVs:
  #      plt.contour(x, y, (x - uav.x) ** 2 + (y - uav.y) ** 2, [cover_r ** 2])  # x**2 + y**2 = 9 的圆形
    for i in range(uav_num):
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], c='blue', marker='.', linewidth=3.5, markersize=7.5)
        plt.plot(x_uav[i][t], y_uav[i][t], c='green', marker='.', markersize=12.5)
    if op:
        plt.plot(op_xpoint, op_ypoint, c='magenta', marker='o', markersize=12.5)
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
    if gif:
        if not os.path.exists(gif_dir): os.mkdir(gif_dir)
        for i in range(1,T +1):
            gif_path = gif_dir + 'step_%s' % str(i).zfill(3)
            plt.figure(facecolor='w', figsize=(20, 20))
            plt.scatter(x_user, y_user, c='red', marker='x', s=150, linewidths=4)
            plt.plot(op_xpoint, op_ypoint, c='magenta', marker='o', markersize=12.5)
            plt.plot(location[0, :i + 1], location[1, :i + 1], c='blue', marker='o', markersize=4.0)
            plt.plot(location[0][i], location[1][i], c='blue', marker='o', markersize=15.0)
            plt.xlim((0, 100))
            plt.ylim((0, 100))
            plt.xticks(fontsize=25)
            plt.yticks(fontsize=25)
            location_r = np.array([location[0][i], location[1][i]])
            title =  'location'
            plt.title(title, fontsize=30)
            plt.savefig(gif_path)


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
    num_fa2 = 0
    sum_reward = 0
    actions = np.zeros(2 * uav_num)
    done = False
    while not done:
        coverage_set[t] = s[:user_num]

        s_, r, done,t,terminal = world.step_inside(a, s, t)
        sum_reward += r


        for i, uav in enumerate(world.UAVs):
            location[0][i] = uav.x
            location[1][i] = uav.y
            location[2][i] = uav.h
        x0_uav[episode][t] = location[0]
        y0_uav[episode][t] = location[1]
        z0_uav[episode][t] = location[2]
        s = s_
        #if done and agent.pointer <= memory_size: # and episode % 5 == 0:
        #    print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'reward: ', r,'fa',world.fa,'raw_r',world.r,'num_fa2',num_fa2, 'Explore:%.2f'% epsino)
        #    print('sum_cover:',world.sum_cover)
        #if done and agent.pointer > memory_size: #and episode % 5 == 0:
    print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'reward: ', r,'fa',world.fa,'raw_r',world.r,'num_fa2',num_fa2, 'Explore:%.2f'% expl_noise)
    #print('loss_a:', agent.loss_a, 'td_error:', agent.td_error)
    print('sum_cover:',world.sum_cover)
  #  if agent.pointer > memory_size:
    #draw_location(x0_uav[episode], y0_uav[episode],t, x0_user, y0_user,
    #             result_path + 'UAVPath_Users_%s.png' % str(episode).zfill(2),s)
    print(time.time()-t3)
print('Running time:', time.time() - t1)
