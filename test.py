import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import World
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import os
from torch.utils.tensorboard import SummaryWriter


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def draw_location_3d(x_uav, y_uav,z_uav,t, x_user, y_user,z_user, savepath,s):
    x_uav = np.transpose(x_uav)
    y_uav = np.transpose(y_uav)
    h_uav = np.transpose(z_uav)
    plt.figure(facecolor='w', figsize=(5, 5))
    for index in range(world.urban_world.Build_num):
        x1 = world.HeightMapMatrix[index][0]
        x2 = world.HeightMapMatrix[index][1]
        y1 = world.HeightMapMatrix[index][2]
        y2 = world.HeightMapMatrix[index][3]
        XList = [x1, x2, x2, x1, x1]
        YList = [y1, y1, y2, y2, y1]
        plt.plot(XList, YList, color='#708090')
    for i in range(user_num):
        if s[user_num+i]==1.0:
            plt.scatter(x_user[i], y_user[i], c='red', marker='^')
        else:
            plt.scatter(x_user[i], y_user[i], c='black', marker='^')
    for i in range(uav_num):
        plt.plot(x_uav[i][0:t+1], y_uav[i][0:t+1], c='blue', marker='.')
        plt.plot(x_uav[i][t], y_uav[i][t], c='green', marker='.')
    new_ticks = np.linspace(0, 10, 6)
    plt.xticks(new_ticks,['0','200','400','600','800','1000'])
    plt.yticks(new_ticks,['0','200','400','600','800','1000'])
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.savefig(savepath)
    #plt.show()
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for index in range(world.urban_world.Build_num):
        x1 = world.HeightMapMatrix[index][0]
        x2 = world.HeightMapMatrix[index][1]
        y1 = world.HeightMapMatrix[index][2]
        y2 = world.HeightMapMatrix[index][3]
        ax.bar3d(x1, y1, 0, world.urban_world.side, world.urban_world.side, world.HeightMapMatrix[index][4]*100, shade=True,
                 color='#708090')  #

      #  ax.text(x1, y1, world.HeightMapMatrix[index][4]*1000 + 2, str(np.floor(world.HeightMapMatrix[index][4]*1000)))
    for i in range(uav_num):
        ax.plot3D(x_uav[i][0:t + 1], y_uav[i][0:t + 1], h_uav[i][0:t + 1] * 100,
                  c='blue')  # , marker='.', linewidth=3.5, markersize=7.5)
        ax.scatter(x_uav[i][t], y_uav[i][t], h_uav[i][t] * 100, c='green', marker='.')  # , markersize=12.5)
    for i in range(user_num):
        if s[user_num + i] == 1.0:
            ax.scatter(x_user[i], y_user[i], z_user[i], c='red', marker='^')
        else:
            ax.scatter(x_user[i], y_user[i], z_user[i], c='black', marker='^')
    new_ticks = np.linspace(0, 10, 6)
    plt.xticks(new_ticks,['0','200','400','600','800','1000'])
    plt.yticks(new_ticks,['0','200','400','600','800','1000'])
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    ax.set_zlabel('Height(m)')
    ax.set_zlim(0, 135)
    #plt.show()
    plt.savefig(result_path + '3DUrban_' + savepath[8:])
    plt.close()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction,user_num):
        super(Actor, self).__init__()
        self.fc_t_sprend = nn.Linear(1, user_num)
        self.fc_loc_sprend = nn.Linear(2, user_num)
        self.l1 = nn.Linear(state_dim - 3 + user_num * 2, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)
        self.maxaction = maxaction
    def forward(self, state):
        t = self.fc_t_sprend(state[:, -1:])
        loc = self.fc_loc_sprend(state[:, -3:-1])
        a = torch.cat([t, loc, state[:, :-3]], 1)
        a = torch.tanh(self.l1(a))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a



# test
uav_num = 1
user_num = 10
uav_h = 0.95
result_path = 'Test_'+str(user_num)+'/'
mkdir(result_path[:-1])
T = 200
Length = 10    #1km
Width = 10
R = 5
cover_r = 2.5
s_dim = uav_num*2 + user_num*2 + 1
a_dim = uav_num*2
dist_max = 0.5
max_action = np.array([math.pi,dist_max/2])

test_episode = 25
world = World(length=Length, width=Width, uav_num=uav_num, user_num=user_num, t=T, uav_h=uav_h,users_name='Users_'+str(user_num)+'.txt')
model = Actor(s_dim, a_dim, 200, max_action,user_num)
model.load_state_dict(torch.load('Model_'+str(user_num)+'/td3_actor8000.pth',map_location={'cuda:1':'cuda:0'}))
model.eval()

coverage_set = np.zeros([T + 1, user_num])
energy_set = np.zeros([T + 1, uav_num])

success_rate = 0.0
Complete_time = np.zeros(test_episode)
fly_time = 0
hover_time = 0
Energy = np.zeros(test_episode)
Distance = np.zeros(test_episode)
Los = np.zeros(test_episode)
Max_cover = np.zeros(test_episode)
x0_user = np.zeros( world.user_num)
y0_user = np.zeros( world.user_num)
z0_user = np.zeros(world.user_num)
x0_uav = np.zeros([test_episode, T + 1, world.uav_num])
y0_uav = np.zeros([test_episode, T + 1, world.uav_num])
z0_uav = np.zeros([test_episode, T + 1, world.uav_num])
location = np.zeros([3, world.uav_num])

def power(v):
    P0 = 79.8563
    Pi = 88.6279
    U_tip = 120
    v0 = 4.03
    d1 = 0.6
    s = 0.05
    rho = 1.225
    A = 0.503
    P_h = P0 * (1+3*v**2/U_tip**2)+Pi*math.sqrt(math.sqrt(1+v**4/(4*v0**4))-v**2/(2*v0**2))+0.5*d1*rho*s*A*v**3
    return P_h

for episode in range(test_episode):
    s, t = world.reset()
    fly_dis = 0.0
    move_time = 0.0
    fly_energy = 0.0
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
        s_shape = torch.unsqueeze(torch.FloatTensor(s), 0)
        with torch.no_grad():
            actions = model(s_shape)[0].detach().numpy()
        fly_dis = fly_dis + (actions[1] + dist_max/2)
        fly_energy += power((actions[1] + dist_max/2)*100/2.5)*2.5
        s_, r, done, t,terminal = world.step_inside(actions, s, t)
        sum_reward += r
        move_time += 1.0
        s = s_
        for i, uav in enumerate(world.UAVs):
            location[0][i] = uav.x
            location[1][i] = uav.y
            location[2][i] = uav.h
            #print(uav.x,uav.y,uav.h)
        x0_uav[episode][t] = location[0]
        y0_uav[episode][t] = location[1]
        z0_uav[episode][t] = location[2]

        #writer.add_scalar('sum_reward%s'% str(episode),sum_reward, t)
        if done:  # and episode % 5 == 0:
            print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'reward: ', r, 'fa', world.fa,
                  'raw_r', world.r)
            print('sum_cover:', world.sum_cover,'hovering_time:',world.hovering_time)
            print(world.service_time)
            print(world.LoS_time)
            print('max_cover_time:',world.max_cover)
    #if episode == 6:
    draw_location_3d(x0_uav[episode], y0_uav[episode],z0_uav[episode], t, x0_user, y0_user,z0_user,
                result_path + 'Test_UAVPath_Users_%s.pdf' % str(episode).zfill(2), s)
    if world.terminal:
        success_rate += 1.0
    Complete_time[episode] = t*2.5 + world.hovering_time
    fly_time += t*2.5
    hover_time += world.hovering_time
    Energy[episode] = fly_energy + world.hovering_time * 168.4842
    Distance[episode] = fly_dis
    Los[episode] = world.LoS_time
    Max_cover[episode] = world.max_cover
    print(move_time)
# for i in range(1,T+1):
print('#', success_rate)
print('##', Complete_time)
print('###',Energy)
print('#', success_rate / test_episode)
print('##', sum(Complete_time)/test_episode)
print('###',sum(Energy)/test_episode)
print('Distance:', sum(Distance)/test_episode)
print('Los time:', sum(Los)/test_episode)
print('fly_time: ',fly_time/test_episode,'   hover_time: ',hover_time/test_episode)


