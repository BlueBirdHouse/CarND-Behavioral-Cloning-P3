# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 09:32:44 2017
这个文件为撰写文档准备相关的数据资料
@author: Bird
"""
#%%包引用功能区
from keras.models import load_model
from keras.utils import plot_model


#%%功能区
model_Path = './Model/Accept3.h5'
model = load_model(model_Path)

plot_model(model, to_file='./Figs/Dmodel.png',show_shapes=True,show_layer_names=False)

