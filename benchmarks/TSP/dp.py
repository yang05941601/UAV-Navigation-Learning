# -*- coding: utf-8 -*-
"""
动态规划法
name:JCH
date:6.8
"""
import pandas as pd
import numpy as np
import math
import time

#dataframe = pd.read_csv("./data/TSP10cities.tsp",sep=" ",header=None)
dataframe = pd.read_csv("./data/TSP10users.tsp",sep=" ",header=None)
v = dataframe.iloc[:,1:3]

train_v= np.array(v)
train_d=train_v
dist = np.zeros((train_v.shape[0]+1,train_d.shape[0]+1))

#计算距离矩阵
for i in range(train_v.shape[0]):
    for j in range(train_d.shape[0]):
        dist[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))
    dist[i,j+1] = 0
for index in range(train_v.shape[0]+1):
    dist[train_v.shape[0],index] = 0
#print(dist[5,8])
#print(dist[1,7])

"""
N:城市数
s:二进制表示，遍历过得城市对应位为1，未遍历为0
dp:动态规划的距离数组
dist：城市间距离矩阵
sumpath:目前的最小路径总长度
Dtemp：当前最小距离
path:记录下一个应该到达的城市
"""

N=train_v.shape[0]+1
path = np.ones((2**(N+1),N))
dp = np.ones((2**(train_v.shape[0]+1+1),train_d.shape[0]+1))*-1

def TSP(s,init,num):
    if dp[s][init] !=-1 :
        return dp[s][init]
    if s==(1<<(N)):
        return dist[0][init]
    sumpath=1000000000
    for i in range(N):
        if s&(1<<i):
            m=TSP(s&(~(1<<i)),i,num+1)+dist[i][init]
            if m<sumpath:
                sumpath=m
                path[s][init]=i
    dp[s][init]=sumpath
    return dp[s][init]

if __name__ == "__main__":
    init_point=0
    s=0
    for i in range(1,N+1):
        s=s|(1<<i)
    start = time.time()
    distance=TSP(s,init_point,0)
    end = time.time()
    s=0b111111111110    # 10个的时候
  #  s = 0b11111111111111110  # 15个的时候
    init=0
    num=0
    print(distance)
    while True:
        print(path[s][init])
        init=int(path[s][init])
        s=s&(~(1<<init))
        num+=1
        if num>9:    #10个的时候小于9， 15个的时候：14
            break
    print("程序的运行时间是：%s"%(end-start))
"""
结果：
10127.5521435
4.0
6.0
7.0
1.0
3.0
2.0
5.0
8.0
9.0
程序的运行时间是：0.025878614787075094
"""