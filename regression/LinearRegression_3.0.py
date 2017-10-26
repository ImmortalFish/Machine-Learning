# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:56:14 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#矩阵转置
def Transpose(matrix):
		#获取矩阵的行数和列数
    rows,cols = matrix.shape
    #创建一个新的矩阵，用来存储转置矩阵
    #行数和列数为原矩阵的列数和行数，因为矩阵转置后大小就变换了
    tran = np.mat(np.zeros((cols,rows)))
    #循环交换对应位置的值
    for row in range(rows):
        for col in range(cols):
            tran[col,row] = matrix[row,col]
    return tran

#求行列式
def Det(matrix):
		#展开第一行递归调用本方法求得行列式
	  #获取行列式的列数
    cols = matrix.shape[1]
    #初始化行列式的值
    det = 0.0
    #如果行列式为空了则结束
    if matrix.shape[0] <= 0:
        return None
    #如果行列式的大小为1，则该行列式的值就为该元素值
    #这是递归调用的最后一层
    elif matrix.shape[0] == 1:
        return matrix[0,0]
    else: 
        for col in range(cols):  
        		#下面这串代码是在获取第一行元素的代数余子式
            minor = matrix[np.array(list(range(0)) + list(range(1,cols)))[:,np.newaxis], np.array(list(range(col)) + list(range(col + 1,cols)))]
            #下面这串代码是在循环求行列式了
            det += (-1) ** (col) * matrix[0,col] * Det(minor)
        return det

#求伴随矩阵  
def Adjoint(matrix):  
		#创建一个新的矩阵，用来存储转置矩阵
		#矩阵的大小与原来的相同  
    new = np.mat(np.zeros(matrix.shape))
    rows,cols = new.shape
    #循环求代数余子式
    for row in range(rows):  
        for col in range(cols):  
            minor = matrix[np.array(list(range(row)) + list(range(row + 1,rows)))[:,np.newaxis],  
                           np.array(list(range(col)) + list(range(col + 1,cols)))]  
            new[row, col] = (-1)**(row+col) * Det(minor)  
    return Transpose(new)

#求逆矩阵
def Inverse(matrix):
    return Adjoint(matrix) / Det(matrix)

#线性回归求结果
#传入一个列表，该列表是一个属性值，且是最后一个数为1.0
#关于传入值，后面主函数有例子，不用在这儿太纠结
def LinearRegression(arr):
		#把列表转为矩阵
    mat_xi = np.mat([arr])
    #计算预测值
    fxi = float(mat_xi * Inverse((Transpose(X) * X)) * Transpose(X) * label)
    return fxi

#画散点图
def Plot(frame,x_label,y_label,i,attrs):
    y = frame[attrs[len(attrs) - 1]]
    plt.figure(figsize=(6,4))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('%s -- reslut' % x_label)
    x = frame[attr[i]]
    T = np.arctan2(x,y)
    plt.scatter(x,y,c  =T,s = 25,alpha = 1.0,marker = 'o')
    plt.show()

if __name__ == '__main__':
	  #path存储列表数据集的绝对路径
    path = 'this is your dateset path'
    #读取数据集
    data = pd.DataFrame(pd.read_csv(path))
    #把数据集转为矩阵
    data_mat = np.mat(data.values)
    #特征名称
    attr = list(data.columns.values)
    #标签数据单独保存
    label = data_mat[:,data_mat.shape[1] - 1]
    #生成一个全是1的列向量
    one = np.linspace(1,1,data_mat.shape[0],dtype = int)[:,np.newaxis]
    #生成公式中的X矩阵
    X = np.hstack((data_mat[:,:data_mat.shape[1] - 1],one))
    #属性数即公式中的d
    num_attr = data_mat.shape[1] - 1
    #这是测试数据，即LinearRegression方法中的传入值
    xi_test= [0.230372,1.0]
    predict = LinearRegression(np.array(xi_test))
    new_row = xi_test.copy()
    new_row[len(new_row) -1 ] = predict
    #把新的一列添加进去
    data_after_predict = data.append(pd.DataFrame(np.mat([new_row]),columns = attr),ignore_index=True)
    print('预测的结果为：',LinearRegression(np.array(xi_test)))
    
    #画出散点图
    if num_attr == 1:
        Plot(data_after_predict,attr[0],attr[num_attr],0,attr)