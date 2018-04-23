# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagle

分别读取每个类下的所有文件，用flow对每类图片分别处理
输出图像分类到对应类文件夹内
batch_size<图片总数时，无法保证每张图片数量相同


"""

import database_IO as io
from PIL import Image
import numpy as np
from sklearn.utils import shuffle


# %%
path_orig = "../data/set/"
incl_agmt = True

'''
数据读取
读取耗时: 0.5329769272005365 s
原样本数: 486  增强样本数: 0
读取耗时: 10.377855655516214 s
原样本数: 486  增强样本数: 9711
样本尺寸: (224, 224, 3)
类别: ['C1', 'C2', 'C3', 'C4', 'C5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'P1', 'P2', 'P3', 'P4', 'P5']
'''
imgs_orig, imgs_name_orig, labels_orig, imgs_agmt, imgs_name_agmt, labels_agmt, classes = io.load_all_data(
    path_orig, incl_agmt)

now_splits = 3
n_splits = 3

'''
数据分割 训练集-测试集
分割耗时: 0.14749014473832744 s
训练集数: 324  测试集数: 162
分割耗时: 2.208010385859467 s
训练集数: 6480  测试集数: 162
中共 3 份 第 2 份
'''
imgs_train, imgs_test, labels_train, labels_test = io.data_split(
    imgs_orig, labels_orig, imgs_agmt, labels_agmt, n_splits, now_splits, incl_agmt)

imgs_train, labels_train = shuffle(imgs_train, labels_train)

#del imgs_orig, imgs_name_orig, labels_orig, imgs_agmt, imgs_name_agmt, labels_agmt

'''
数据分割 训练集-验证集-测试集
'''
imgs_vld, imgs_test, labels_vld, labels_test = io.train_test_split(
    imgs_test, labels_test, test_size=0.5)

'''
将标签中字符映射为序号
'''
labels_train = np.vectorize(classes.get)(labels_train)
labels_vld = np.vectorize(classes.get)(labels_vld)
labels_test = np.vectorize(classes.get)(labels_test)
