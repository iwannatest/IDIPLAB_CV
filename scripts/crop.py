# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:54:23 2018

@author: Sandiagle
"""

import glob
import os
from skimage.io import imread, imshow, imsave

path_origin = '../data/set 400×400/'
g = os.walk(path_origin)

for path, dir_list, file_list in g:
    # dir_list:类别的list，例如['cats', 'dogs']
    for dir_name in dir_list:
        # dir_name:类别的名称，例如cats

        for filename in glob.glob(path_origin+dir_name+'/*.jpg'):
            #  filename:带有路径的图片完整地址，例如../Data_Origin/cats\rezero_icon_10.jpg
            img = imread(filename)  # 这是一个PIL图像

            # 设置图像尺寸
            height, width = img.shape[:2]
            x = int((width-height)/2)
            w = height
            img = img[:, x:w+x, :]
            imsave(filename, img)
