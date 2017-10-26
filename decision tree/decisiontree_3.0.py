# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 00:46:30 2017

@author: Administrator
"""

import pandas as pd
import math
import numpy as np

#读取数据的函数
def ReadData(path):
    data_frame = pd.DataFrame(pd.read_csv(path))
    return data_frame

path = 'D:\\文档\\暑期培训\\04--有监督学习\\数据\\watermelon2.0.csv'
data = ReadData(path)

#属性的list
attr = list(data.columns.values[:data.shape[1] - 1])
attr_copy = attr.copy()
#样本的总数
num_sample = data.shape[0]
#标签类型,此处为[1,0]
label = list(data.iloc[:,-1].drop_duplicates())

#计算当前样本集合的熵
def Entropy(frame):
    #样本的数量
    num_frame = frame.shape[0]
    #创建一个value为全0的字典，此处为{0: 0, 1: 0}
    label_dict = dict.fromkeys(label,0)
    #数一下各个类所占的个数
    for i in range(frame.shape[0]):
        label_dict[frame.ix[i][-1]] += 1 
    #初始化熵值
    ENT = 0.0
    for key in label_dict:
        pk = label_dict[key] / num_frame
        if pk == 0:
            ENT -= pk
        else:
            ENT -= pk * math.log(pk,2)
    return ENT

#计算根据属性a划分的信息增益
def InfoGain(frame,a):
    gain = 0.0
    #首先要统计属性a有哪些取值
    a_attr = list(frame[a].drop_duplicates())
    #接着根据每个可能的值划分样本并计算其熵
    for i in a_attr:
        frame_a_i = frame.ix[frame[a] == i].reset_index(drop = True)
        #分支节点的权重
        weight_i = frame_a_i.shape[0] / frame.shape[0]
        ent_i = Entropy(frame_a_i) #对了的，验证了的
        gain += weight_i * ent_i
    return Entropy(frame) - gain  
    
#计算属性a的增益率
def GainRate(frame,a):
    IV = 0.0
    a_attr = list(frame[a].drop_duplicates())
    for i in a_attr:
        frame_a_i = frame.ix[frame[a] == i].reset_index(drop = True)
        weight_i = frame_a_i.shape[0] / frame.shape[0]
        if weight_i == 0:
            IV -= weight_i
        else:
            IV -= weight_i * math.log(weight_i,2)
    return InfoGain(frame,a) / IV

#选择最优属性，返回属性
def ChooseBestAttr(frame):
    #属性名的list
    attr = list(frame.columns.values[:data.shape[1] - 1])
    #属性的个数
    num_attr = frame.shape[1] - 1
    #初始化各属性的信息增益率的字典
    gain_dict = dict.fromkeys(attr,0)
    sum_gain = 0.0
    for i in attr:
        gain_dict[i] = InfoGain(frame,i)
        sum_gain += InfoGain(frame,i)
    #所有属性的平均信息增益    
    aver_gain = sum_gain / num_attr
    #选择信息增益高于平均水平的属性
    gain_big_aver_dict = gain_dict.copy()
    for key,value in gain_dict.items():
        if value <= aver_gain:
            gain_big_aver_dict.pop(key)
    #再选择增益率最高的属性
    gain_rate = gain_big_aver_dict.copy()
    for key,value in gain_big_aver_dict.items():
        gain_rate[key] = GainRate(frame,key)
    return {v:k for k,v in gain_rate.items()}[max(gain_rate.values())]

#判断当前的样本集合中的所有样本是否全是一个类别    
def SameLable(frame):
    attr_current = list(frame.iloc[:,-1].drop_duplicates())
    #如果是则返回True
    if len(attr_current) == 1:
        return True
    #如果不是则返回False
    else:
        return False
    
#判断属性集是否为空
def AttrSetIsNull(attr_list):
    if len(attr_list) == 0:
        return True
    else:
        return False

#划分样本子集
def SplitData(frame,a_attr,value):
    frame_new = frame[frame[a_attr].isin([value])]
    return frame_new.reset_index(drop = True)
    
#找出数据集中样本数最多的类并返回该类别
def MostSample(frame):
    label_list = list(frame.iloc[:,-1])
    label_current = list(frame.iloc[:,-1].drop_duplicates())
    label_count_dict = dict.fromkeys(label_current,0)
    for i in label_current:
        label_count_dict[i] = label_list.count(i)
    return {v:k for k,v in label_count_dict.items()}[max(label_count_dict.values())]

#判断数据集在属性集上的取值是否相同
def IsSameValue(frame,attrs):
    if frame.shape[0] == 1:
        return True
    elif list(frame[attrs].values[0]) == list(frame[attrs].values[1]):
        return True
    else:
        return False
    
def CraetTree(frame,attrs):
    #标签集合
    label_list = list(frame.iloc[:,-1])
    #如果样本集合全是一个类别，则直接返回该类别
    if SameLable(frame):
        return label_list[0]
    #如果属性集为空话则返回类别次数最多的类
    if len(label_list) == 1 or IsSameValue(frame,attrs):
        return MostSample(frame)
    #找到最优属性
    best_atrr = ChooseBestAttr(frame)
    best_attr_set = set(frame[best_atrr])
    #创建节点
    my_tree = {best_atrr:{}}
		#attr.remove(best_atrr)
    for value in best_attr_set:
        Dv = SplitData(frame,best_atrr,value)
        if Dv is None:
            return MostSample(frame)
        else:
            left_labels = attrs[:]
            my_tree[best_atrr][value] = CraetTree(Dv,left_labels)
    return my_tree

#对新来的数据进行决策树分类
def Desicion(tree,attrs,test_data):
    root_node = list(tree.keys())[0]
    second_node = tree[root_node]
    node_index = attrs.index(root_node)
    for key in second_node.keys():
        if test_data[node_index] == key:
            if type(second_node[key]).__name__ =='dict':
                label_desicion = Desicion(second_node[key],attrs,test_data)
            else:
                label_desicion = second_node[key]
    return label_desicion
    
if __name__ == '__main__':    
    print('所生成的树的字典结构如下：',CraetTree(data,attr))
    test_data = np.array(data.drop(data.columns.values[-1],axis = 1))
    label_desicion_list = []
    for i in test_data:
        label_desicion_list.append(Desicion(CraetTree(data,attr),attr,list(i)))
    print('分类的结果：',label_desicion_list)
