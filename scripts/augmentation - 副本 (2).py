# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 11:53:24 2018

@author: Sandiagle

不分别读取每个类下的所有文件，用flow_from_directory
直接对总文件夹处理
输出图像无法分类到对应类文件夹内
batch_size<图片总数时，无法保证每张图片数量相同

"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import glob
import os
import numpy as np
from scipy import misc
from preprocessing_funciton import random_crop_with_noise
import time
# %%

path_origin = '../Data_Origin/'
path_agument = '../Data_Agument/'
agument_time = 10  # 一张图成倍增长数量

data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     zca_whitening=True,
                     zca_epsilon=1e-6,
                     rotation_range=30.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     brightness_range=None,
                     shear_range=0.2,
                     zoom_range=0.2,
                     channel_shift_range=1.,
                     cval=0.,
                     horizontal_flip=True,
                     preprocessing_function=random_crop_with_noise)

# %%
start = time.clock()

train_datagen = ImageDataGenerator(**data_gen_args)

g = os.walk(path_origin)

for path, dir_list, file_list in g:
    # dir_list:类别的list，例如['cats', 'dogs']
    for dir_name in dir_list:
        # dir_name:类别的名称，例如cats
        print(dir_name)
        is_exists = os.path.exists(path_agument+dir_name)
        if not is_exists:
            os.makedirs(path_agument+dir_name)

        # 设置生成流
        train_generator = train_datagen.flow_from_directory(
            path_origin+dir_name,
            target_size=(150, 150),
            batch_size=32,
            shuffle=None,
            save_to_dir=path_agument+dir_name,
            save_format='jpg')

        # 循环生成
        for i in range(agument_time):
            train_generator.next()

end = time.clock()
print(end-start)
# 原始2592×1728 1.55MB 101张 耗时40s
# 第三方缩小到400×400耗时5s augmentation耗时10s