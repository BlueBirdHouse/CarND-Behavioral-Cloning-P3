# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:25:11 2017

@author: Bird
这个文件完成模型的训练过程
"""
#%%包文件导入区
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Conv2D, Cropping2D

#%%函数定义区
def GetFromPickle(FileName):
    '''
    调入存储的数据文件
    '''
    pickle_file = './Data/Pretreatment' + FileName
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        Images = pickle_data['Images']
        steering_angle = pickle_data['steering_angle']
      
    return Images, steering_angle

def RandomZeroData(angle_In):
    '''
    针对训练数据里的零数据，通过增加噪声额的方法避免为0
    '''  
    angle_Out = []
    for Counter in range(len(angle_In)):
        A_steering_angle = angle_In[Counter]
        if A_steering_angle == 0:
            angle_Out.append(A_steering_angle + 0.01 * np.random.randn())
        else:
            angle_Out.append(A_steering_angle)
    return angle_Out

def generator(Images, steering_angle, batch_size=128):
    #利用生成器为每一个训练集增加不同的噪声
    num_samples = len(steering_angle)
    while 1: # Loop forever so the generator never terminates
        Images, steering_angle = sklearn.utils.shuffle(Images, steering_angle)
        for offset in range(0, num_samples, batch_size):
            X_train = Images[offset:offset+batch_size]
            y_train = steering_angle[offset:offset+batch_size]

            y_train = y_train + 0.01 * np.random.randn(len(y_train))

            yield X_train, y_train


#%%调入训练数据
Left_Images, Left_steering_angle = GetFromPickle('Left')
Left_Images_Flip, Left_steering_angle_Flip = GetFromPickle('Left_Flip')
Right_Images, Right_steering_angle = GetFromPickle('Right')
Right_Images_Flip, Right_steering_angle_Flip = GetFromPickle('Right_Flip')

Images = np.concatenate((Left_Images,Left_Images_Flip,Right_Images,Right_Images_Flip),axis=0)
steering_angle = np.concatenate((Left_steering_angle,Left_steering_angle_Flip,Right_steering_angle,Right_steering_angle_Flip),axis=0)

del Left_Images,Left_Images_Flip,Right_Images,Right_Images_Flip
del Left_steering_angle,Left_steering_angle_Flip,Right_steering_angle,Right_steering_angle_Flip

#%%生成训练网络
model = Sequential()
#标准化层(接受浮点输入)
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
#model.add(Lambda(lambda x: x/255.0, input_shape = (160,320,3)))
#图像裁剪层
model.add(Cropping2D(cropping=((70,25),(0,0))))
#卷积1
model.add(Conv2D(24, 5, strides=(2, 2)))
model.add(Activation('relu'))
#卷积2
model.add(Conv2D(36, 5, strides=(2, 2)))
model.add(Activation('relu'))
#卷积3
model.add(Conv2D(48, 5, strides=(2, 2)))
model.add(Activation('relu'))
#卷积4
#model.add(Conv2D(64, 3, strides=(1, 1)))
#model.add(Activation('relu'))
#卷积4
#model.add(Conv2D(64, 3, strides=(1, 1)))
#model.add(Activation('relu'))
#进入分类层
model.add(Flatten())
#model.add(Dense(1164))
#model.add(Activation('sigmoid'))
#model.add(Dense(100))
#model.add(Activation('sigmoid'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.add(Dense(1))

#模型生成
model.compile(optimizer='Adam',loss='mean_squared_error')

#%%训练模型
#model.fit(x = Images, y = steering_angle, batch_size = 128, epochs=1,validation_split=0.2,shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(Images,steering_angle, test_size=0.2, shuffle = True)
train_generator = generator(X_train,y_train)
validation_generator = generator(X_test,y_test)

model.fit_generator(train_generator, steps_per_epoch=len(y_train)//128,epochs=3, validation_data=validation_generator, validation_steps=len(y_test)//128)


#%%保存模型
#model.save(filepath='./Model/Accept3.h5',overwrite=True,include_optimizer = True)














        
