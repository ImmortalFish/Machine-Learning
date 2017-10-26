# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 22:16:51 2017

@author: Administrator
"""

'''
算法：k-means
数据集：西瓜数据集4.0
簇数(k)：3
'''

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

#首先还是要读取数据
def ReadData(path):
    data = pd.DataFrame(pd.read_csv(path))
    return data

#随机选择k个样本作为初始均值向量
def RandomSelectVec(frame):
    '''
    frame:数据集
    return:均值向量dataframe
    '''
    #将随机选择的k个初始向量存储到init_vec_list中
    list_select = random.sample(range(len(frame)),k)
    return frame.ix[list_select].reset_index(drop = True)
    
#计算欧氏距离并找出最短距离的簇标记
def MinEuclidean(frame_1,frame_2,j):
    '''
    j:第j个样本int
    frame_1:数据集
    frame_2:均值向量dataframe
    返回：样本j选择的距离最近的簇的编号，[0,1,2]中的一个
    '''
    func = lambda x : np.square(x)
    sum_squre_ij = 0.0
    list_euc = []
    #求距离
    for i in range(len(frame_2)):
        sum_squre_ij = sum((frame_1.ix[j] - frame_2.ix[i]).apply(func))
        list_euc.append(np.sqrt(sum_squre_ij))
    return list_euc.index(min(list_euc))

#根据距离最近的均值把该样本的编号归类到簇中
def JoinCluster(frame_1,frame_2):
    '''
    frame_1:数据集
    frame_2：均值向量frame
    return:簇分类结果list
    '''
    #创建一个有k个空值的簇列表
    cluster_all = []
    for i in range(k):
        cluster_all.append([])
    #把样本一次添加进相应的簇
    for j in range(len(frame_1)):
        cluster_all[MinEuclidean(frame_1,frame_2,j)].append(list(frame_1.index)[j])
    return cluster_all

#计算新的均值向量
def NewVec(cluster_list,frame):
    '''
    cluster_list:簇分类结果
    frame:数据集
    return：新的均值向量
    '''
    #创建一个空的frame，存储新的均值向量
    new = pd.DataFrame()
    #计算新的均值向量并存储
    for i in range(k):
        new = new.append(frame.ix[cluster_list[i]].mean(),ignore_index = True)
    return new

#判断分簇是否与以前的是否一样
def Judge(cluster_1,cluster_2):
    '''
    cluster_1:之前的cluster
    cluster_2:新的cluster
    return:bool
    '''
    #如果簇分类就过没有变化就返回True
    if cluster_1 == cluster_2:
        return True
    else:
        #否则返回False
        return False

#把聚类之后的结果画出来
def Plot(frame_1,frame_2,cluster_list,x):
    '''
    frame_1:数据集
    frame_2:最终的均值向量
    cluster_list:最终的分类结果
    x:要分几个簇
    '''
    #如果数据集不是二维的就划不出来的
    if frame_1.shape[1] != 2:
        
        print('不是二维数据画不出来图！')
        return None
    
    color_1 = ['oy', 'ob', 'og', 'ok', 'or', 'oc', 'om', '+r', '+g', '+c'] 
    if x > len(color_1):
        print('分的簇太多，颜色不够画了~ ~ ~ ~ ~')
        return None
    
    for i in frame_1.index:
        for j in range(len(cluster_list)):
            if i in cluster_list[j]:
                cluster_index = j
        plt.plot(data.ix[i][0],data.ix[i][1],color_1[cluster_index])
    color_2 = ['^y', '^b', '^g', '^k', '^r', '^c', '^m', '^r', '^g', '^c']    
    for i in frame_2.index:
        plt.plot(frame_2.ix[i][0],frame_2.ix[i][1],color_2[i],markersize = 12)
    
    plt.show()
            
if __name__ == '__main__':
    csv_path = input('请输入您的数据集文件的绝对路径：')
    print('\n正在进行聚类，请稍等。。。。。。。\n')
    data = ReadData(csv_path)
    frame_test = data.ix[[5,11,23]].reset_index(drop = True)  #西瓜书的初始均值向量，用来测试
    #自定义簇的大小
    k = 3
    #初始选择的均值向量frame
    init_vec_frame = RandomSelectVec(data)
    #根据初始的均值向量的分簇结果，存储的编号
    cluster = JoinCluster(data,init_vec_frame)
    #新的均值向量frame
    i = 1
    #用来标记簇是否改变了
    cluster_diff = True
    #开始迭代
    while cluster_diff:
        #新的均值向量frame
        new_frame = NewVec(cluster,data)
        #新分簇结果，存储的编号
        cluster_new = JoinCluster(data,new_frame)
        if Judge(cluster,cluster_new):
            #新分的簇与上一次的簇相同
            cluster_diff = False
            print('下面是聚类的结果，编号代表数据集中样本的索引：\n')
            for i in range(k):
                print('簇%d:\t' % (i + 1),cluster_new[i])
                print('\n')
        else:
            cluster_diff = True
            cluster = cluster_new.copy()         
        i = i + 1
    Plot(data,new_frame,cluster_new,k)  