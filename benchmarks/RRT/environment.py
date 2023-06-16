from entity import UAV, User
import numpy as np
import os
import sys
import math
import copy
from urban_world import Urban_world

class World(object):
    def __init__(self, length=2500, width=2500, uav_num=9, user_num=100, T=1000, uav_h=1.0, Cover_r=300,users_name='Users.txt'):
        self.length = length
        self.width = width
        self.uav_num = uav_num
        self.user_num = user_num
        self.users_path = users_name
        self.Users = []
        self.UAVs = []
        self.T = T
        self.t = 0
        self.max_x = length
        self.min_x = 0
        self.max_y = width
        self.min_y = 0
        self.uav_h = uav_h
        self.Cover_r = Cover_r
        self.eh = 1.0   # hovering energy
        self.er = 10.0 / 5.0 # max energy consumption / hovering
        self.dist_max = 0.5 # max flying distance m
        self.fa = 0.0
        self.r = 0.0
        self.sum_cover = 0.0
        self.terminal = False
        self.service_time = 0.0
        self.LoS_time = 0.0
        self.max_cover = 0.0
        # =============Define the GT Distribution=======================
        self.GT_loc = np.zeros([self.user_num,3])
        if os.path.exists(self.users_path):
            self.set_users()
            self.urban_world = Urban_world(self.GT_loc)
            self.HeightMapMatrix = self.urban_world.Buliding_construct()
        else:
            self.urban_world = Urban_world(self.GT_loc)
            self.HeightMapMatrix = self.urban_world.Buliding_construct()
            self.set_users()

    def reset(self, ):
 #       if test == 100:
 #           self.set_users()
 #       else:
 #           self.set_test_users(test)
        self.set_uavs()
        state = self.reset_state()
        self.t = 0
        self.service_time = 0.0
        self.LoS_time = 0.0
        self.max_cover = 0.0
        return state,self.t

    def set_users(self):
        self.Users =[]
        if os.path.exists(self.users_path):
            f = open(self.users_path, 'r')
            if f:self.Users = []
        if os.path.exists(self.users_path):
            f = open(self.users_path, 'r')
            if f:
                user_loc = f.readline()
                user_loc = user_loc.split(' ')
                self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                self.GT_loc[len(self.Users) - 1] = np.array(
                    [float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                while user_loc:
                    user_loc = f.readline()
                    if user_loc:
                        user_loc = user_loc.split(' ')
                        self.Users.append(User(float(user_loc[0]), float(user_loc[1]),float(user_loc[2])))
                        self.GT_loc[len(self.Users) - 1] = np.array([float(user_loc[0]), float(user_loc[1]), float(user_loc[2])])
                f.close()
        else:
            f = open(self.users_path, 'w')
            creat = False
            while not creat:
                x = np.random.uniform(1, self.length-1,1)[0]
                y = np.random.uniform(1, self.width-1,1)[0]
                z = 0.0
                count = 0
                for index in range(self.urban_world.Build_num):
                    x1 = self.HeightMapMatrix[index][0]
                    x2 = self.HeightMapMatrix[index][1]
                    y1 = self.HeightMapMatrix[index][2]
                    y2 = self.HeightMapMatrix[index][3]
                    if (x < x1 or x > x2) and (y < y1 or y > y2):
                        count += 1
                        continue
                if count == self.urban_world.Build_num:
                    self.Users.append(User(x, y, z))
                    self.GT_loc[len(self.Users)-1] = np.array([x,y,z])
                    f.writelines([str(x), ' ', str(y), ' ', str(z), '\n'])
                if len(self.Users) == self.user_num:
                    creat = True
            f.close()

    def set_uavs(self):
        self.UAVs = []
        x_set = np.random.uniform(1, self.length-1, self.uav_num)
        y_set = np.random.uniform(1, self.width-1, self.uav_num)
        for i in range(self.uav_num):
            x = x_set[i]
            y = y_set[i]
            h = self.uav_h
            self.UAVs.append(UAV(x, y, h))
        while not self.uav_connected():
            x_set = np.random.uniform(0.25, self.length - 0.25, self.uav_num)
            y_set = np.random.uniform(0.25, self.width - 0.25, self.uav_num)
            for i, uav in enumerate(self.UAVs):
                uav.x = x_set[i]
                uav.y = y_set[i]
                uav.h = 1.0

    def reset_state(self):
        s = np.zeros(self.user_num*2 + self.uav_num*2+1)
        #  state of user served by uav
        for j, uav in enumerate(self.UAVs):
            # 计算用户与无人机之间的数据速率
            uav_loc = [uav.x,uav.y,uav.h]
            cover_state,LoS_state,SNR_set = self.urban_world.getPointMiniOutage(uav_loc)
      #  print(cover_state, LoS_state)
        s[:self.user_num] = cover_state
        s[self.user_num:2*self.user_num] = s[:self.user_num]
        for i in range(len(cover_state)):
            if cover_state[i] == 1 and LoS_state[i]==True:
                self.LoS_time += 1.0

        if sum(cover_state) > 0.0:
            self.service_time += 1.0
        # state of energy consumption of uav
        for i,uav in enumerate(self.UAVs):
            s[2 * self.user_num+i*2] = uav.x
            s[2 * self.user_num+i*2+1] = uav.y
        s[-1] = 0.0 # 初始血量
        self.max_cover = sum(cover_state)
        return s

    def update_state(self, s,t,fa):
        s_ = np.zeros(self.user_num*2 + self.uav_num*2 + 1)
        cover_state = np.zeros(self.user_num)
        #  state of user served by uav
        for j, uav in enumerate(self.UAVs):
            # 计算用户与无人机之间的数据速率
            uav_loc = [uav.x,uav.y,uav.h]
            cover_state,LoS_state = self.urban_world.getPointMiniOutage(uav_loc)
      #  print(cover_state,LoS_state)
        s_[:self.user_num] = cover_state
        s_[self.user_num:self.user_num*2] = np.clip(s[self.user_num:self.user_num*2]+s_[:self.user_num],0,1)
        for i in range(len(cover_state)):
            if s[self.user_num+i]==0 and s_[self.user_num+i]==1 and LoS_state[i]==True:
                self.LoS_time += 1.0

        for i,uav in enumerate(self.UAVs):
            s_[2 * self.user_num+i*2] = uav.x
            s_[2 * self.user_num+i*2+1] = uav.y
        # 吃到的血包
        num_poi = sum(s_[self.user_num:self.user_num*2])-sum(s[self.user_num:2*self.user_num])
        if num_poi > 0.0:
            self.service_time += 1.0
        s_[-1] = s[-1]-1.0 + num_poi*10 - fa  #血量,回血药
        if self.max_cover < num_poi:
            self.max_cover = num_poi
        return s_

    def get_reward(self, state_):
        #reward = -1
        self.sum_cover = sum(state_[self.user_num:2*self.user_num])
        # 生存概率
        reward = 2/(1+np.exp(-state_[-1]/(self.user_num*10)))-1
        #reward = reward
        return reward

    def step_inside(self, actions, state, t):
        fa = 0.0
        v2 = 1.0
        fa2 = 0.0
        reward = 0.0
        self.t = t+1
        state_ = np.zeros(self.user_num*2 + self.uav_num*2 + 1)
        action_fake = copy.deepcopy(actions)
        uav_location_pre = np.zeros([self.uav_num, 2])  # 保存执行动作前的uav位置
        for i, uav in enumerate(self.UAVs):
            uav_location_pre[i][0] = uav.x
            uav_location_pre[i][1] = uav.y
        if len(actions) == self.uav_num * 2:
            for i, uav in enumerate(self.UAVs):
                uav.move_inside_test(actions[0],actions[1],self.dist_max)    # uav位置已经更新，执行完动作
            state_ = self.update_state(state,self.t, fa)
            for i, uav in enumerate(self.UAVs):
                penalty, bound = self.boundary_margin(uav, reward)
                fa += penalty
                if not bound:
                    uav.x = uav_location_pre[i][0]      # uav过界，取消动作，并且更新状态
                    uav.y = uav_location_pre[i][1]
                    state_ = self.update_state(state,self.t,fa)     # 更新状态
        reward = self.get_reward(state_)
        if sum(state_[self.user_num:2*self.user_num]) == float(self.user_num):
            print(state_[self.user_num:2*self.user_num])
            reward += 200-self.t
            self.terminal = True
            print("Complete task!!!")
        #elif state_[-1] <= 0.0:
        #    self.terminal = True # dead
        else:
            self.terminal = False
        done = False
        if self.terminal or self.t >= self.T:
            #reward += self.sum_cover
            done = True
        #self.fa = fa
        self.r = reward
        r = reward #- fa - fa2
        # print(reward, fa, fa2)
        return state_, r,done,self.t,self.terminal

    def boundary_margin(self, uav, reward):
        bound = True
        v1 = 1.0
       # alpha = 1 / (12500 * self.uav_num)
        alpha = 0.0
        #beta = 3 / self.uav_num - 9 / (25 * self.uav_num)
        beta = 1 / self.uav_num
        lx_plus = max(abs(uav.x - (self.max_x + self.min_x) / 2) - v1 * (self.max_x - self.min_x) / 2, 0.0)
        ly_plus = max(abs(uav.y - (self.max_y + self.min_y) / 2) - v1 * (self.max_y - self.min_y) / 2, 0.0)
        if lx_plus == 0.0 and ly_plus == 0.0:
            fa = 0.0
        else:
            fa = (alpha * (lx_plus ** 2 + ly_plus ** 2 ) + beta) * 1
            bound = False
        return fa, bound

    def uav_connected(self):
        connected = True
        MAX = sys.maxsize
        chararray_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G' ,'H' ,'I' ,'J' ,'K']  # UAV命名
        primgraph = [[0 for i in range(self.uav_num)] for j in range(self.uav_num)]
        graph_mess = [[0 for i in range(3)] for j in range(self.uav_num)]
        for i ,uav1 in enumerate(self.UAVs):
            for j ,uav2 in enumerate(self.UAVs):
                distance = math.sqrt((uav1.x - uav2.x) ** 2 + (
                        uav1.y - uav2.y) ** 2 )
                if i == j:
                    loss_ata = MAX
                    primgraph[i][j] = loss_ata
                    continue
                loss_ata = distance
                primgraph[i][j] = loss_ata     # 由空空信道的损失建立Graph
        # print(primgraph)
        chararray = chararray_list[:self.uav_num]
        charlist = []
        charlist.append(chararray[0])
        mid = []  # mid[i]表示生成树集合中与点i最近的点的编号
        lowcost = []
        # lowcost[i]表示生成树集合中与点i最近的点构成的边最小权值 ，-1表示i已经在生成树集合中
        lowcost.append(-1)
        mid.append(0)
        n = len(chararray)
        for i in range(1, n):  # 初始化mid数组和lowcost数组
            lowcost.append(primgraph[0][i])
            mid.append(0)
        sum = 0
        for k in range(1, n):  # 插入n-1个结点
            minid = 0
            min = MAX
            for j in range(1, n):  # 寻找每次插入生成树的权值最小的结点
                if (lowcost[j] != -1 and lowcost[j] < min):
                    minid = j
                    min = lowcost[j]
            charlist.append(chararray[minid])
            # print(chararray[mid[minid]] + '——' + chararray[minid] + '权值：' + str(lowcost[minid]))
            graph_mess[k] = [chararray[mid[minid]], chararray[minid], lowcost[minid]]
            sum += min
            lowcost[minid] = -1
            for j in range(1, n):  # 更新插入结点后lowcost数组和mid数组值
                if (lowcost[j] != -1 and lowcost[j] > primgraph[minid][j]):
                    lowcost[j] = primgraph[minid][j]
                    mid[j] = minid
        # print("sum=" + str(sum))
        # print("插入结点顺序：" + str(charlist))
        # print(graph_mess)
        for uav3 in range(1, self.uav_num):
            if graph_mess[uav3][2] > self.R:
                connected = False
        return connected
