from entity import UAV, User
import numpy as np
import os
import sys
import math
import copy
from urban_world import Urban_world

class World(object):
    def __init__(self, length=10, width=10, uav_num=1, user_num=20, t=200, uav_h=1.0,users_name='Users.txt'):
        self.length = length
        self.width = width
        self.uav_num = uav_num
        self.user_num = user_num
        self.users_path = users_name
        self.Users = []
        self.UAVs = []
        self.T = t
        self.t = 0
        self.max_x = length
        self.min_x = 0
        self.max_y = width
        self.min_y = 0
        self.uav_h = uav_h
        self.dist_max = 0.5 # max flying distance m
        self.fa = 0.0
        self.r = 0.0
        self.sum_cover = 0.0
        self.hovering_time = 0.0
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
        self.set_uavs()
        state = self.reset_state()
        self.t = 0
        self.hovering_time = 0.0
        self.service_time = 0.0
        self.LoS_time = 0.0
        self.max_cover = 0.0
        return state,self.t

    def set_users(self):
        self.Users =[]
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

    def reset_state(self):
        s = np.zeros(self.user_num*2 + self.uav_num*2+1)
        #  state of user served by uav
        for j, uav in enumerate(self.UAVs):
            # information between the UAV and users
            uav_loc = [uav.x,uav.y,uav.h]
            cover_state,LoS_state,SNR_set = self.urban_world.getPointMiniOutage(uav_loc)
        s[:self.user_num] = cover_state
        s[self.user_num:2*self.user_num] = s[:self.user_num]
        h_t = 0.0
        for i in range(len(cover_state)):
            if cover_state[i] == 1 and LoS_state[i]==True:
                self.LoS_time += 1.0
            if cover_state[i] == 1:
                current_ht = 10/(10*np.log2(1+SNR_set[i]))
                if h_t < current_ht:
                    h_t = current_ht
        self.hovering_time += h_t
        if sum(cover_state) > 0.0:
            self.service_time += 1.0
        # state of energy consumption of uav
        for i,uav in enumerate(self.UAVs):
            s[2 * self.user_num+i*2] = uav.x
            s[2 * self.user_num+i*2+1] = uav.y
        s[-1] = 0.0  # initial pheromone on the UAV
        self.max_cover = sum(cover_state)
        return s

    def update_state(self, s,t,fa):
        s_ = np.zeros(self.user_num*2 + self.uav_num*2 + 1)
        cover_state = np.zeros(self.user_num)
        #  state of user served by uav
        for j, uav in enumerate(self.UAVs):
            uav_loc = [uav.x,uav.y,uav.h]
            cover_state,LoS_state,SNR_set = self.urban_world.getPointMiniOutage(uav_loc)
        s_[:self.user_num] = cover_state
        s_[self.user_num:self.user_num*2] = np.clip(s[self.user_num:self.user_num*2]+s_[:self.user_num],0,1)
        h_t = 0.0
        for i in range(len(cover_state)):
            if s[self.user_num+i]==0 and s_[self.user_num+i]==1 and LoS_state[i]==True:
                self.LoS_time += 1.0
            if s[self.user_num+i]==0 and s_[self.user_num+i]==1:
                current_ht = 10 / (10 * np.log2(1 + SNR_set[i]))
                if h_t < current_ht:
                    h_t = current_ht
        self.hovering_time += h_t
        for i,uav in enumerate(self.UAVs):
            s_[2 * self.user_num+i*2] = uav.x
            s_[2 * self.user_num+i*2+1] = uav.y
        # the number of the covered users in the current step
        num_poi = sum(s_[self.user_num:self.user_num*2])-sum(s[self.user_num:2*self.user_num])
        if num_poi > 0.0:
            self.service_time += 1.0
        s_[-1] = s[-1]-1.0 + num_poi*10 - fa  # update the pheromone
        if self.max_cover < num_poi:
            self.max_cover = num_poi
        return s_

    def get_reward(self, state_):
        self.sum_cover = sum(state_[self.user_num:2*self.user_num])
        reward = 2/(1+np.exp(-state_[-1]/(self.user_num*10)))-1
        return reward

    def step_inside(self, actions, state, t):
        fa = 0.0
        reward = 0.0
        self.t = t+1
        state_ = np.zeros(self.user_num*2 + self.uav_num*2 + 1)
        uav_location_pre = np.zeros([self.uav_num, 2])  # make a copy of the uav's location
        for i, uav in enumerate(self.UAVs):
            uav_location_pre[i][0] = uav.x
            uav_location_pre[i][1] = uav.y
        if len(actions) == self.uav_num * 2:
            for i, uav in enumerate(self.UAVs):
                uav.move_inside_test(actions[0],actions[1],self.dist_max)    # execute the action
            state_ = self.update_state(state,self.t, fa)
            for i, uav in enumerate(self.UAVs):
                penalty, bound = self.boundary_margin(uav)
                fa += penalty
                if not bound:
                    uav.x = uav_location_pre[i][0]      # the uav break the boundary constraint, the action is cancelled
                    uav.y = uav_location_pre[i][1]
                    state_ = self.update_state(state,self.t,fa)     # update the state again.
        reward = self.get_reward(state_)
        if sum(state_[self.user_num:2*self.user_num]) == float(self.user_num):
            print(state_[self.user_num:2*self.user_num])
            reward += 200-self.t
            self.terminal = True
            print("Complete task!!!")
        else:
            self.terminal = False
        done = False
        if self.terminal or self.t >= self.T:
            done = True
        self.r = reward
        return state_, self.r,done,self.t,self.terminal

    def boundary_margin(self, uav):
        bound = True
        v1 = 1.0
        alpha = 0.0
        beta = 1 / self.uav_num
        lx_plus = max(abs(uav.x - (self.max_x + self.min_x) / 2) - v1 * (self.max_x - self.min_x) / 2, 0.0)
        ly_plus = max(abs(uav.y - (self.max_y + self.min_y) / 2) - v1 * (self.max_y - self.min_y) / 2, 0.0)
        if lx_plus == 0.0 and ly_plus == 0.0:
            fa = 0.0
        else:
            fa = (alpha * (lx_plus ** 2 + ly_plus ** 2 ) + beta) * 1
            bound = False
        return fa, bound