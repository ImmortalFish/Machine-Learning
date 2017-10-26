# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 11:40:19 2017

@author: Administrator
"""

'''
随机森林算法
'''

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree

def LoadData():
    '''
    读取数据集
    return: dataframe
    '''
    path = 'this is your dataset path'
    return pd.read_csv(path)

def SplitData(frame):
    '''
    把数据集划分为测试集和训练集，同时标签列全部拿出
    frame: 数据集
    return: 训练集，测试集，训练集标签，测试集标签
    '''
    coulumns = frame.shape[1]
    x = frame.ix[:,0:coulumns - 1]
    y = frame.ix[:,coulumns - 1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test

def RandomSelect(frame, series):
    '''
    随机抽样数据集
    frame: 训练集
    series: 训练集的标签列
    return: 有放回随机抽样后的数据集和对应的标签列
    '''
    sample = frame.sample(frac = 1.0, replace = True)
    return sample, series[list(sample.index)]

def Tree(frame, series):
    '''
    训练决策树，选择的特征数为总特征数的平方根，不限制最大深度
    frame: 训练集
    series: 训练集的标签列
    return: 决策树
    '''
    clf = tree.DecisionTreeClassifier(max_features = 'sqrt')
    clf.fit(frame, series)
    return clf

def RandomForest(frame, series, num):
    '''
    生成随机森林
    frame: 训练集
    series: 训练集的标签列
    num: 森林中树的数量
    return: 决策树的集合即森林
    '''
    frame_list = []
    series_list = []
    clf_list = []
    
    for i in range(num):
       x, y = RandomSelect(frame, series)
       frame_list.append(x)
       series_list.append(y)
       
    for i in range(num):
       clf_list.append(Tree(frame_list[i], series_list[i]))
       
    return clf_list 


def Predict(frame, forest):
    '''
    随机森林的测试
    frame: 测试集
    forest: 随机森林
    return: 预测的结果
    '''
    num = len(forest)
    result_frame = pd.DataFrame(columns = range(num))
    
    for j in range(len(forest)):
        result_frame[j] = forest[j].predict(frame)
    
    return result_frame.mode(axis = 1)[0]
    

def Performance(a_list, series):
    '''
    查看该随机森林的性能
    a_list: 预测结果的列表
    series: 测试集的标签列
    return: 预测对了的个数
    '''
    right_num = 0
    for i in range(len(a_list)):
        if a_list[i] == series[list(series.index)[i]]:
            right_num = right_num + 1
            
    return right_num

def RandomForestMain():
    data = LoadData()
    tree_num = int(input('请输入森林中树的数量:'))
    data_train, data_test, label_train, label_test = SplitData(data)
    random_forest = RandomForest(data_train, label_train, tree_num)
    predict_result = Predict(data_test, random_forest)
    right_num = Performance(predict_result, label_test)
    print('\n测试集总数是%d个, 预测对了%d个, 随机森林的准确率为%.2f%% \n' % (len(label_test), right_num, right_num / len(label_test) * 100))
    tree_result = tree.DecisionTreeClassifier().fit(data_train, label_train).predict(data_test)
    right_tree = Performance(tree_result, label_test)
    print('决策树模型的准确率为%.2f%% \n' % (right_tree / len(label_test) * 100))
    
if __name__ == '__main__':
    RandomForestMain()