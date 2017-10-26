# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 13:13:43 2017

@author: Administrator
"""

'''
算法：高斯混合聚类
数据集：西瓜数据集4.0
簇数(k)：3
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random

#读取数据并将其转为矩阵存储
def ReadData(path):
    '''
    path:数据集所在的绝对路径
    return:返回数据集的矩阵mat
    '''
    mat = np.mat(pd.DataFrame(pd.read_csv(path)).values)
    return mat

#高斯混合分布  GaussiaMixedDistribution:GMD
def GMD(alpha,vec,miu,cov_mat):
    '''
    alpha:选择第i个高斯分布的概率
    vec:数据集中的一个样本
    miu:均值向量
    cov_mat:nxn的协方差矩阵
    return:返回概率
    '''
    n = len(vec)
    down = pow(2 * math.pi,n / 2) * pow(np.linalg.det(cov_mat),0.5) #公式中的分母
    up = pow(math.e,-0.5 * float((vec - miu) * cov_mat.I * (vec - miu).T))
    p = up / down
    return alpha * p

#计算第j个样本x由第i个高斯混合分布生成的后验概率即E步  后验概率PosteriorProbability:pp
def EStep(vec,miu_list,alpha_list,cov_mat_list,k):
    '''
    vec:数据集中的一个样本
    miu_list:存储miu值的list
    alpha_list:存储alpha值的list
    cov_mat_list：存储协方差矩阵的list
    k:有k个高斯分布
    return:返回样本在这k个高斯分布下的后验概率的list
    '''
    up = []
    down = 0.0
    pp = []
    for i in range(k):
        up.append(GMD(alpha_list[i],vec,miu_list[i],cov_mat_list[i]))
        down += GMD(alpha_list[i],vec,miu_list[i],cov_mat_list[i])
    for i in range(k):
        pp.append(up[i] / down)
    return pp

#确定新的模型参数
def MStep(data,pp_mat,k):
    '''
    data:数据集
    pp_mat:向量j对应在第i个高斯分布下的后验概率的矩阵
    k:有k个高斯分布
    return:各个新的参数
    '''
    mat = data.copy()
    mat_1 = np.mat(np.zeros([2,2]))
    new_miu_list = []  
    new_alpha_list = []
    new_cov_mat_list = []
    for i in range(k):
        down = pp_mat.sum(axis = 0)[0,i]
        for j in range(len(data)):
            mat[j] = pp_mat[j,i] * data[j]
        up = mat.sum(axis = 0)
        new_miu_list.append(up / down)
        new_alpha_list.append(down / len(data))
        
    for i in range(k):
        down = pp_mat.sum(axis = 0)[0,i]
        for j in range(len(data)):
           mat_1 += pp_mat[j,i] * (data[j] - new_miu_list[i]).T * (data[j] - new_miu_list[i])
        new_cov_mat_list.append(mat_1 / down)
        mat_1 = np.mat(np.zeros([2,2]))
    return new_miu_list,new_alpha_list,new_cov_mat_list

#根据后验概率选择簇
def JoinCluster(mat,pp_mat,k):
    '''
    mat:数据集mat
    pp_mat:向量j对应在第i个高斯分布下的后验概率的矩阵
    k:有k个高斯分布
    return:分簇的情况list
    '''
    #创建一个有k个空值的簇列表
    cluster = []
    for i in range(k):
        cluster.append([])
    for j in range(len(mat)):
        cluster[int(sum(pp_mat[j].tolist(),[]).index(pp_mat[j].max()))].append(j)
    return cluster

#把聚类之后的结果画出来
def Plot(data,cluster,miu_list,k):
    '''
    data:数据集
    cluster:包含簇分类的数据集
    miu_list:最终的均值向量
    cluster_list:最终的分类结果
    k:要分几个簇
    '''
    #如果数据集不是二维的就划不出来的
    if data.shape[1] != 2:  
        print('不是二维数据画不出来图！')
        return None
    
    color_1 = ['oy', 'ob', 'og', 'ok', 'or', 'oc', 'om', '+r', '+g', '+c'] 
    if k > len(color_1):
        print('分的簇太多，颜色不够画了~ ~ ~ ~ ~')
        return None

    for j in range(len(data)):
        for i in range(len(cluster)):
            if j in cluster[i]:
                cluster_index = i
        plt.plot(data[j,0],data[j,1],color_1[cluster_index])
        
    color_2 = ['^y', '^b', '^g', '^k', '^r', '^c', '^m', '^r', '^g', '^c']    
    for i in range(len(miu_list)):
        plt.plot(miu_list[i][0,0],miu_list[i][0,1],color_2[i],markersize = 12)
        
    plt.show()
    
#初始化模型参数
def InitPara(data):
    '''
    data:数据集
    return:各个初始的参数
    '''
    k = 3
    data_cluster_start = []
    for i in range(k):
        data_cluster_start.append([])
    miu_list = list(data[[5,21,26]])
#    list_select = random.sample(range(len(data)),k)
#    miu_list = list(data[list_select])
    cov_1 = np.mat([[0.1,0.0],[0.0,0.1]])
    cov_mat_list = []
    alpha_list = [0.333,0.333,0.333]
    for i in range(k):
        cov_mat_list.append(cov_1)
    pp = np.mat(np.zeros([len(data),k]))
    
    return k,miu_list,alpha_list,cov_mat_list,data_cluster_start,pp

#主函数
if __name__ == '__main__':
    csv_path = input('请输入您的数据集文件的绝对路径：')
    print('\n正在进行聚类，请稍等。。。。。。。\n')
    data_mat = ReadData(csv_path)
    k,miu_lists,alpha_lists,cov_mat_lists,cluster_start,pp_ji = InitPara(data_mat)
    i = 1
     
    while i <= 50:
        for j in range(len(data_mat)):
            pp_ji[j] = EStep(data_mat[j],miu_lists,alpha_lists,cov_mat_lists,k)
        miu_lists,alpha_lists,cov_mat_lists = MStep(data_mat,pp_ji,k)
        i += 1  
    cluster_new = JoinCluster(data_mat,pp_ji,k)
    Plot(data_mat,cluster_new,miu_lists,k)    