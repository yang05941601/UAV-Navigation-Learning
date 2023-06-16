import pandas as pd
import numpy as np
import math
import time

#dataframe = pd.read_csv("./data/TSP10cities.tsp",sep=" ",header=None)
dataframe = pd.read_csv("./data/TSP5users.tsp",sep=" ",header=None)
v = dataframe.iloc[:,1:3]
train_v= np.array(v)
path = [5,4,3,1,2,0]
dist = 0
for i in range(len(path)-1):
    dist_2 = math.sqrt((train_v[path[i]][0]-train_v[path[i+1]][0])**2+(train_v[path[i]][1]-train_v[path[i+1]][1])**2)
    print(train_v[path[i]],train_v[path[i+1]],dist_2)
    dist += dist_2
print(dist)
