import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from environment import World

uav_num = 1
user_num = 40
dis_set = [15.31,21.87,27.27,26.37,29.14,32.05,34.9,39.24]
dis = dis_set[7]
uav_h = 0.95
T = 200
Length = 10  # 10km
Width = 10
cover_r = 2.5

world = World(length=Length, width=Width, uav_num=uav_num, user_num=user_num, T=T, uav_h=uav_h, Cover_r=cover_r,users_name='Users_' + str(user_num) + '.txt')
hovering_time = 0.0
for i, user in enumerate(world.Users):
    # 计算用户与无人机之间的数据速率
    uav_loc = [user.x,user.y,uav_h]
    cover_state,LoS_state,SNR_set = world.urban_world.getPointMiniOutage(uav_loc)
    hovering_time += 10/(10*np.log2(1+SNR_set[i]))
print('hovering_time:',hovering_time)

#TSP_path = [2.0, 17.0, 0.0, 11.0, 6.0, 12.0, 13.0, 39.0, 9.0, 18.0, 34.0, 37.0, 35.0, 36.0, 27.0, 1.0, 10.0, 23.0, 16.0,
#            7.0, 5.0, 24.0, 21.0, 32.0, 19.0, 22.0, 25.0, 26.0, 8.0, 38.0, 30.0, 20.0, 15.0, 3.0, 29.0, 31.0, 14.0, 28.0,
#            4.0, 33.0,40] # 40

#TSP_path = [29.0, 3.0, 15.0, 31.0, 20.0, 30.0, 8.0, 26.0, 22.0, 25.0, 32.0, 19.0, 24.0, 21.0, 16.0, 5.0, 7.0, 23.0, 10.0,
#            1.0, 27.0,34.0, 9.0, 18.0, 11.0, 0.0, 17.0, 2.0, 6.0, 12.0, 13.0, 28.0, 14.0, 4.0, 33.0,35] # 35
#TSP_path = [7.0, 5.0, 16.0, 23.0, 1.0, 10.0, 27.0, 24.0, 19.0, 21.0, 25.0, 22.0, 26.0, 8.0, 20.0, 15.0, 3.0, 29.0, 9.0,
#            18.0, 11.0, 0.0, 17.0, 2.0, 6.0, 12.0, 13.0, 28.0, 14.0, 4.0,30] # 30

#TSP_path = [ 2.0, 0.0, 17.0, 12.0, 6.0, 11.0, 18.0, 9.0, 10.0, 1.0, 23.0, 5.0, 16.0, 7.0, 21.0, 24.0, 19.0, 22.0, 8.0,
#                 20.0, 15.0, 3.0, 13.0, 14.0, 4.0,25] # 25

#TSP_path = [16.0, 7.0, 5.0, 1.0, 10.0, 19.0, 8.0, 3.0, 15.0, 13.0, 12.0, 6.0, 9.0, 18.0, 11.0, 0.0, 17.0, 2.0, 14.0, 4.0,20] # 20

#TSP_path = [11.0, 5.0, 10.0, 7.0, 6.0, 14.0, 8.0, 0.0, 2.0, 3.0, 12.0, 13.0, 4.0, 9.0, 1.0,15] # 15

#TSP_path = [7.0, 5.0, 1.0, 9.0, 8.0, 3.0, 6.0, 0.0, 2.0, 4.0,10] # 10

#TSP_path = [1.0, 3.0, 0.0, 2.0, 4.0,5] # 5



Complete_time = dis * 100 / 20 + hovering_time
Energy = (dis * 100 / 20) * 178.2958 + hovering_time * 168.4842
print('Complete_time:',Complete_time)
print('Energy:',Energy)
