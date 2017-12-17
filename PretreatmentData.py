# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:25:11 2017
这个文件完成数据预处理过程。所有可以前期处理的逻辑都放在这里。
最后处理的好的数据被存储。
@author: Bird
"""
#%%包文件导入区
import pickle
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda
from keras.layers import Conv2D, Cropping2D

#%%函数定义区
def GetFromPickle(FileName):
    '''
    调入存储的数据文件
    '''
    pickle_file = './Data/TrainData_' + FileName
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        Images = pickle_data['Images']
        steering_angle = pickle_data['steering_angle']
        steering_angle = np.asarray(steering_angle,dtype = np.float32)
        
    return Images, steering_angle

def GetUnZeroData(Images_In, angle_In):
    '''
    这个函数找到不为0的角度信息和图片，以便做训练
    '''
    Images = np.zeros([0,160,320,3],dtype = np.uint8)
    steering_angle = []
    
    for Counter in range(len(angle_In)):
        A_steering_angle = angle_In[Counter]
        if A_steering_angle != 0:
            steering_angle.append(A_steering_angle)
            
            AFig = Images_In[Counter,:,:,:]
            AFig = AFig[np.newaxis,:,:,:]
            Images = np.row_stack((Images,AFig))
    
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

def FlipAugmentation(Images_In, angle_In):
    '''
    利用翻转图片来增加训练数量
    angle_In需要是矩阵
    '''
    Images = np.zeros([0,160,320,3],dtype = np.uint8)
    steering_angle = []
    for Counter in range(len(angle_In)):
        AFig = Images_In[Counter,:,:,:]
        AFig = np.fliplr(AFig)
        AFig = AFig[np.newaxis,:,:,:]
        Images = np.row_stack((Images,AFig))
        print(Counter)
    steering_angle = -angle_In
    return Images,steering_angle

def AddToPickle(Images,steering_angle,FileName):
    '''
    多次调用在相同Pickle文件中累加数据
    '''
    pickle_file = './Data/Pretreatment' + FileName
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                    {
                        'Images': Images,
                        'steering_angle': steering_angle
                    },
            pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise


#%%调入训练数据
Left_Images, Left_steering_angle = GetFromPickle('Left')
Right_Images, Right_steering_angle = GetFromPickle('Right')

#找到不为0的训练数据
#Left_Images, Left_steering_angle = GetUnZeroData(Left_Images, Left_steering_angle)
#Right_Images, Right_steering_angle = GetUnZeroData(Right_Images, Right_steering_angle)
#随机化数据
#Left_steering_angle = RandomZeroData(Left_steering_angle)
#Right_steering_angle = RandomZeroData(Right_steering_angle)
#反转数据
Left_Images_Flip, Left_steering_angle_Flip = FlipAugmentation(Left_Images, Left_steering_angle)
Right_Images_Flip, Right_steering_angle_Flip = FlipAugmentation(Right_Images, Right_steering_angle)


#%%数据存储过程
AddToPickle(Left_Images,Left_steering_angle,'Left')
AddToPickle(Right_Images,Right_steering_angle,'Right')
AddToPickle(Left_Images_Flip,Left_steering_angle_Flip,'Left_Flip')
AddToPickle(Right_Images_Flip,Right_steering_angle_Flip,'Right_Flip')



        
