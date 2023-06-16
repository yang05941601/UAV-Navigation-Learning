# -*- coding: utf-8 -*-
"""
分支限界法
name:JCH
date:6.8
"""
import pandas as pd
import numpy as np
import math
from queue import Queue
import time

dataframe = pd.read_csv("./data/TSP10cities.tsp",sep=" ",header=None)
v = dataframe.iloc[:,1:3]

train_v= np.array(v)
train_d=train_v
dist = np.zeros((train_v.shape[0],train_d.shape[0]))

#计算距离矩阵
for i in range(train_v.shape[0]):
    for j in range(train_d.shape[0]):
        dist[i,j] = math.sqrt(np.sum((train_v[i,:]-train_d[j,:])**2))

INF = 10000000
n = train_v.shape[0]
class Node:
    def __init__(self):
        self.visited=[False]*n
        self.s=1
        self.e=1
        self.k=1
        self.sumv=0
        self.lb=0
        self.listc=[]


pq = Queue() #创建一个优先队列
low=0 #下界
up=0#上界（使用贪心算法得出）
dfs_visited=[False]*n
dfs_visited[0]=True
def dfs(u,k,l):
    if k==n-1 :
        return (l+dist[u][0])
    minlen=INF
    p=0
    for i in range(n):
        if dfs_visited[i]==False and minlen>dist[u][i]:
            minlen=dist[u][i]
            p=i
    dfs_visited[p]=True
    return dfs(p,k+1,l+minlen)

def get_up():
    global up
    up=dfs(0,0,0)

def get_low():
    global low
    for i in range(n):
        temp=dist[i].copy()
        temp.sort()
        #print("%s"%(temp[0]))
        low=low+temp[0]+temp[1]
    low=low/2

def get_lb(p):
    ret=p.sumv*2
    min1=INF #起点和终点连出来的边
    min2=INF
    #从起点到最近未遍历城市的距离 
    for i in range(n):
        if p.visited[i]==False and min1>dist[i][p.s]:
            min1=dist[i][p.s]
    ret=ret+min1
    
    #从终点到最近未遍历城市的距离
    for j in range(n):
        if p.visited[j]==False and min2>dist[p.e][j]:
            min2=dist[p.e][j]
    #进入并离开每个未遍历城市的最小成本
    for i in range(n):
        if p.visited[i]==False:
            min1=min2=INF
            for j in range(n):
                min1=dist[i][j] if min1 > dist[i][j] else min1
            for m in range(n):
                min2=dist[i][m] if min2 > dist[m][i] else min2
            ret=ret+min1+min2
    return (ret+1)/2


def solve():
    global up
    get_up()
    get_low() #获得下界
    node=Node()
    node.s=0 #起始点从1开始
    node.e=0 #结束点到1结束(当前路径的结束点)
    node.k=1 #遍历过得点数，初始1个
    node.visited=[False]*n #是否遍历过
    node.listc.append(0)
    for i in range(n):
        node.visited[i]==False
    node.visited[0]=True
    node.sumv=0 #目前路径的距离和
    node.lb=low #初始目标值等于下界
    ret=INF #ret是问题的最终解
    pq.put(node) #将起点加入队列
    while pq.qsize()!=0: #如果已经走过了n-1个点
        tmp=pq.get()
        if tmp.k==n-1:
            p=0 #最后一个没有走的点
            for i in range(n):
                if tmp.visited[i]==False:
                    p=i
                    break
            ans=tmp.sumv+dist[tmp.s][p]+dist[p][tmp.e] #总的路径消耗
            #如果当前的路径和比所有的目标函数值都小则跳出
            #否则继续求其他可能的路径和，并更新上界
            if ans <= tmp.lb:
                ret=min(ans,ret)
                break
            else:
                up=min(ans,up)#上界更新为更接近目标的ans值
                ret=min(ret,ans)
                continue
        #当前点可以向下扩展的点入优先级队列
        
        for i in range(n):
            if tmp.visited[i]==False:
                next_node=Node()
                next_node.s=tmp.s #沿着tmp走到next，起点不变 
                next_node.sumv=tmp.sumv+dist[tmp.e][i]
                next_node.e=i #更新最后一个点 
                next_node.k=tmp.k+1
                next_node.listc=tmp.listc.copy()
                next_node.listc.append(i)
                #print(tmp.k)
                #tmp经过的点也是next经过的点
                next_node.visited=tmp.visited.copy()
                next_node.visited[i] = True;
                next_node.lb = get_lb(next_node);#求目标函数
                if next_node.lb>=up:
                    continue
                pq.put(next_node)
    tmp.listc.append(4)
    return ret,tmp

if __name__ == "__main__":
    for i in range(n):
        dist[i][i]=INF
    start = time.clock()
    sumpath,node=solve()
    end = time.clock()
    print("结果：")
    print(sumpath)
    list1=node.listc.copy()
    for i in list1:
        print(i)
    print("程序的运行时间是：%s"%(end-start))
    

"""
结果：
10127.5521435
0
9
8
6
7
1
3
2
5
4
程序的运行时间是：0.7683305954415118
"""