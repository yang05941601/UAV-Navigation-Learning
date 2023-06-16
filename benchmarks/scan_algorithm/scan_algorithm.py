from environment import World
from matplotlib import pyplot as plt
import numpy as np
import math
import os
result_path = 'Result/'

# test
uav_num = 1
user_num = 35
T = 200
Length = 10    #10km
Width = 10
uav_h = 0.95
cover_r = 2.5
s_dim = uav_num*2 + user_num*2   #无人机的位置和当前覆盖情况和总体的覆盖
a_dim = uav_num
dist_max = 0.5
max_action = [math.pi, dist_max / 2]
test_episode = 25
world = World(length=Length, width=Width, uav_num=uav_num, user_num=user_num, T=T, uav_h=uav_h, Cover_r=cover_r,users_name='Users_'+str(user_num)+'.txt')

x = y = np.arange(0,Width,0.01)
x, y = np.meshgrid(x,y)
np.random.seed(1)
# generate scan action pair
action_space = np.zeros([T,1])
for i in range(T):
    if 0<= i <= 18 or 41<=i<=58 or 81<=i<=98 or 121<=i<=138 or 161<=i<=178:
        action_space[i] = 0.0  #右
    elif 19<=i<=20 or 39<=i<=40 or 59<=i<=60 or 79<=i<=80 or 99<=i<=100 or 119<=i<=120 or 139<=i<=140 or 159<=i<=160 or 179<=i<=180:
        action_space[i] = 0.5  #上
    elif 21<=i<=38 or 61<=i<=78 or 101<=i<=118 or 141<=i<=158 or 181<=i<=199:
        action_space[i] = 1.0   #左



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
        plt.plot(x_uav[i][t], y_uav[i][t], c='green', marker='o', markersize=12.5)
        plt.plot(x_uav[i][0], y_uav[i][0], c='red', marker='o', markersize=12.5)
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
    plt.xlim((0, Length))
    plt.ylim((0, Width))
    plt.xlabel('x(km)',fontsize=30)
    plt.ylabel('y(km)',fontsize=30)
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


coverage_set = np.zeros([T + 1, user_num])
energy_set = np.zeros([T + 1, uav_num])

success_rate = 0.0
Complete_time = np.zeros(test_episode)
Energy = np.zeros(test_episode)
Distance = np.zeros(test_episode)
Los = np.zeros(test_episode)
Max_cover = np.zeros(test_episode)

x0_user = np.zeros([test_episode, world.user_num])
y0_user = np.zeros([test_episode, world.user_num])
z0_user = np.zeros([test_episode, world.user_num])
x0_uav = np.zeros([test_episode, T + 1, world.uav_num])
y0_uav = np.zeros([test_episode, T + 1, world.uav_num])
z0_uav = np.zeros([test_episode, T + 1, world.uav_num])
location = np.zeros([3, world.uav_num])

for episode in range(test_episode):
    s,t = world.reset()
    for i, user in enumerate(world.Users):
        x0_user[episode][i] = user.x
        y0_user[episode][i] = user.y
    for i, uav in enumerate(world.UAVs):
        location[0][i] = uav.x
        location[1][i] = uav.y
        location[2][i] = uav.h
    x0_uav[episode][0] = location[0]
    y0_uav[episode][0] = location[1]
    z0_uav[episode][0] = location[2]
    fly_dis = 0.0
    sum_reward = 0
    done = False
    while not done:
        actions = action_space[t]
        fly_dis = fly_dis + dist_max
        s_, r, done, t,terminal = world.step_inside(actions, s, t)
        sum_reward += r
        for i, uav in enumerate(world.UAVs):
            location[0][i] = uav.x
            location[1][i] = uav.y
            location[2][i] = uav.h
        x0_uav[episode][t] = location[0]
        y0_uav[episode][t] = location[1]
        z0_uav[episode][t] = location[2]
        s = s_
        if done:  # and episode % 5 == 0:
            print('Episode:', episode, 'Epoch:', t, 'total_Reward: %.4f' % sum_reward, 'reward: ', r, 'fa', world.fa,
                  'raw_r', world.r)
            print('sum_cover:', world.sum_cover)
    draw_location(x0_uav[episode], y0_uav[episode],t, x0_user[episode], y0_user[episode],
                  result_path + 'Random_UAVPath_Users_%s.png' % str(episode).zfill(2),s)
    if world.terminal:
        success_rate += 1.0
    print(world.service_time)
    print(fly_dis)
    print(world.LoS_time)
    print('hovering_time:',world.hovering_time)
    Distance[episode] = fly_dis
    Complete_time[episode] = fly_dis * 100 / 20 + world.hovering_time
    Energy[episode] = (fly_dis * 100 / 20) * 178.2958 + world.hovering_time * 168.4842
    Los[episode] = world.LoS_time
    Max_cover[episode] = world.max_cover
    #for i in range(1,T+1):
print('#', success_rate)
print('##', Complete_time)
print('###',Energy)
print('#', success_rate/test_episode)
print('##', sum(Complete_time)/test_episode)
print('###', sum(Energy)/test_episode)
print('Distance:',sum(Distance) / test_episode)
print('Los time:', sum(Los)/test_episode)
print('Max_cover:',Max_cover)




