"""
RRT_2D
@author: huiming zhou
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from environment import World
import env, plotting
import copy

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")


uav_num = 1
user_num = 5
uav_h = 0.95
T = 200
Length = 10  # 10km
Width = 10
cover_r = 2.5
world = World(length=Length, width=Width, uav_num=uav_num, user_num=user_num, T=T, uav_h=uav_h, Cover_r=cover_r,
                  users_name='Users_' + str(user_num) + '.txt')
num = 0.0
service_num = 0.0
hovering_time = 0.0
los_time = 0.0
np.random.seed(2)
# TSP_path = [2.0, 17.0, 0.0, 11.0, 6.0, 12.0, 13.0, 39.0, 9.0, 18.0, 34.0, 37.0, 35.0, 36.0, 27.0, 1.0, 10.0, 23.0, 16.0,
#             7.0, 5.0, 24.0, 21.0, 32.0, 19.0, 22.0, 25.0, 26.0, 8.0, 38.0, 30.0, 20.0, 15.0, 3.0, 29.0, 31.0, 14.0, 28.0,
#             4.0, 33.0] # 40

#TSP_path = [29.0, 3.0, 15.0, 31.0, 20.0, 30.0, 8.0, 26.0, 22.0, 25.0, 32.0, 19.0, 24.0, 21.0, 16.0, 5.0, 7.0, 23.0, 10.0,
#            1.0, 27.0,34.0, 9.0, 18.0, 11.0, 0.0, 17.0, 2.0, 6.0, 12.0, 13.0, 28.0, 14.0, 4.0, 33.0] # 35
#TSP_path = [7.0, 5.0, 16.0, 23.0, 1.0, 10.0, 27.0, 24.0, 19.0, 21.0, 25.0, 22.0, 26.0, 8.0, 20.0, 15.0, 3.0, 29.0, 9.0,
#            18.0, 11.0, 0.0, 17.0, 2.0, 6.0, 12.0, 13.0, 28.0, 14.0, 4.0] # 30

#TSP_path = [ 2.0, 0.0, 17.0, 12.0, 6.0, 11.0, 18.0, 9.0, 10.0, 1.0, 23.0, 5.0, 16.0, 7.0, 21.0, 24.0, 19.0, 22.0, 8.0,
#                 20.0, 15.0, 3.0, 13.0, 14.0, 4.0] # 25

#TSP_path = [16.0, 7.0, 5.0, 1.0, 10.0, 19.0, 8.0, 3.0, 15.0, 13.0, 12.0, 6.0, 9.0, 18.0, 11.0, 0.0, 17.0, 2.0, 14.0, 4.0] # 20

#TSP_path = [11.0, 5.0, 10.0, 7.0, 6.0, 14.0, 8.0, 0.0, 2.0, 3.0, 12.0, 13.0, 4.0, 9.0, 1.0] # 15

#TSP_path = [7.0, 5.0, 1.0, 9.0, 8.0, 3.0, 6.0, 0.0, 2.0, 4.0] # 10

TSP_path = [1.0, 3.0, 0.0, 2.0, 4.0] # 5

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env.Env()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary


    def planning(self, TSP_path, state,num,service_num,los_time,hovering_time):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new:
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)
                uav_loc = [node_new.x, node_new.y, 0.95]
                cover_state, LoS_state, SNR_set = world.urban_world.getPointMiniOutage(uav_loc)
                pop_temp = []
                if cover_state[int(TSP_path[-1])] == 1.0:
                    self.new_state(node_new, self.s_goal)
                    h_t = 0.0
                    for i in range(len(TSP_path)):
                        if cover_state[int(TSP_path[i])] == 1.0:
                            if LoS_state[int(TSP_path[i])] == True:
                                los_time += 1.0
                            current_ht = 10/(10*np.log2(1+SNR_set[int(TSP_path[i])]))
                            if h_t < current_ht:
                                h_t = current_ht
                            num = num + 1.0
                            print('goal!!!:', num)
                            pop_temp.append(i)
                    for i in range(len(pop_temp)):
                        TSP_path.pop(pop_temp[len(pop_temp)-1-i])
                    service_num += 1.0
                    hovering_time += h_t
                    return self.extract_path(node_new), TSP_path, uav_loc,num,service_num,los_time,hovering_time

        return None

    def generate_random_node(self, goal_sample_rate):
        delta = 0.05

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(node_end.x,node_end.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def plot_path(path):
    if len(path) != 0:
        plt.plot([x[0] for x in path], [x[1] for x in path], '-r', linewidth=2)
        plt.pause(0.01)
    plt.show()


state = world.reset()
for uav in world.UAVs:
    x = uav.x
    y = uav.y
x_start = (0, 0.5)  # Starting node
X_goal_set = []
path_total = [x_start]
while TSP_path:
    x_goal = (world.Users[int(TSP_path[-1])].x, world.Users[int(TSP_path[-1])].y)  # Goal node
    X_goal_set.append(x_goal)
    rrt = Rrt(x_start, x_goal, 0.1, 0.05, 10000)
    print(TSP_path)
    path,TSP_path, UAV_loc,num,service_num,los_time,hovering_time = rrt.planning(TSP_path, state,num,service_num,los_time,hovering_time)
    x_start = (UAV_loc[0],UAV_loc[1])
    path_total = path[:-1]+path_total
#rrt2 = Rrt(x_goal, x_goal2, 0.5, 0.05, 10000)

#path2 = rrt2.planning()
plot_path(path_total)
if path_total:
    plotting.Plotting(x_start, X_goal_set).animation(rrt.vertex, path_total, "RRT", True)
else:
    print("No Path Found!")
dis = 0.0
for i in range(len(path_total)-1):
    dis += np.sqrt((path_total[i+1][0]-path_total[i][0])**2+(path_total[i+1][1]-path_total[i][1])**2)

print("Distance_ori:", dis)
print('service_time:',service_num)
Complete_time = dis * 100 / 20 + hovering_time
Energy = (dis * 100 / 20) * 178.2958 + hovering_time * 168.4842
print('Complete_time:',Complete_time)
print('Energy:',Energy)
print('los_time:',los_time)
print('Max_cover:',max_cover)


