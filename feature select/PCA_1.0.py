# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:38:33 2017

@author: Administrator
"""

'''

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#读取数据
def ReadData():
    '''
    return:数据集dataframe
    '''
    path = input('请输入数据集的绝对路径：')
    return pd.read_csv(path)

#求列平均并用样本减去列平均即去中心化
def Aver(frame):
    '''
    frame:数据集
    return:减去了列平均的数据集
    '''
    f = lambda x : x - frame.mean()
    return frame.apply(f,axis = 1)

#求协方差矩阵
def CovMat(frame):
    '''
    frame:减去了平均值的数据集
    return:协方差矩阵
    '''
    mean = frame.mean()
    n = frame.shape[1]
    mat = np.mat(np.zeros([n,n]))
    for i in range(n):
        for j in range(n):
            mat[i,j] = np.mat(frame.ix[:,i].apply(lambda x : x - mean[i])) * np.mat(frame.ix[:,j].apply(lambda x : x - mean[j])).T / (len(frame) - 1)
    return mat

#求特征值和特征向量
def EigenValuesVec(frame):
    '''
    frame:协方差矩阵
    return:特征值与特征向量
    '''
    return np.linalg.eig(frame)

#选择k个特征值最大的特征向量组成矩阵并返回
def SelectVec(array_1,array_2,k):
    '''
    array_1:特征值
    array_2:特征值向量
    k:选择多少个特征向量
    return:选择的特征向量组成的矩阵
    '''
    empty_frame = pd.DataFrame()
    sort_array_1 = sorted(array_1,reverse = True)
    for i in range(k):
        index = array_1.tolist().index(sort_array_1[i])
        empty_frame = pd.concat([empty_frame,pd.DataFrame(array_2[:,index])],axis = 1)
    return np.mat(empty_frame)

#将样本点投影到选取的特征向量上
def Projection(frame,mat):
    '''
    frame:减去了平均值的数据集
    mat:选择的特征向量组成的矩阵
    '''
    return np.dot(np.mat(frame),mat)

#PCA主函数
def Pca(frame,k):
    '''
    frame:数据集
    k:选择几个特征向量
    return:降维后的矩阵
    '''
    aver_frame = Aver(frame)
    cov_mat = CovMat(aver_frame)
    eig_value,eig_vec = EigenValuesVec(cov_mat)
    eig_vec_mat = SelectVec(eig_value,eig_vec,k)
    return Projection(aver_frame,eig_vec_mat)

#画图函数
def Plot(frame,mat,vec):
    '''
    frame:去中心化后的数据集
    mat:降维后的矩阵
    vec:特征向量
    ''' 
    colors = ['yellow','blue']
    for i in range(len(frame)):
        if mat.shape[1] == 1:
            for j in range(len(vec)):
                plt.plot([vec[:,j][0],0],[vec[:,j][1],0],colors[j])
            plt.plot(frame.ix[i][0],frame.ix[i][1],'og')
            plt.plot(mat[i,0],0,'^r')
        else:
            for j in range(len(vec)):
                plt.plot([vec[:,j][0],0],[vec[:,j][1],0],colors[j])
            plt.plot(frame.ix[i][0],frame.ix[i][1],'og')
            plt.plot(mat[i,0],mat[i,1],'^r')
    plt.show()
    
#对应的样本值减去
if __name__ == '__main__':
    data = ReadData()  #D:\文档\暑期培训\07--PCA J48\数据集\test.csv
    N = int(input('请输入降维的维数：'))
    result = Pca(data,N)
    if data.shape[1] != 2:
        print('投影矩阵为：',result)
    else:
        Plot(Aver(data),result,EigenValuesVec(CovMat(Aver(data)))[1])
        print('实线为特征向量；红色的点为投影矩阵；绿色的点为去中心化后的数据集')