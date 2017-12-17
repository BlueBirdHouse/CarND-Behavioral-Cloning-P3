# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:48:11 2017
这个文件利用游戏大师FuCaihong产生的训练数据训练神经网络
从模拟器生成的ZIP文档中取出数据，然后转换为pickle格式
这些训练数据非常重要，所以建立这个专门的转换文件
@author: Bird
"""
#%%包文件导入区
import rarfile
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

#%%函数定义区
def LoadFiles(FileName,KeyWord):
    '''
        这个函数将训练数据导入内存。
    '''
    Images = np.zeros([0,160,320,3],dtype = np.uint8)
    steering_angle = []
    
    rarf = rarfile.RarFile(FileName)
    csv_file = rarf.open(KeyWord + '/driving_log.csv')
    counter = 0
    for Aline in csv_file:
        Aline = Aline.decode('utf8')
        Aline = Aline.split(',')
        ImgFileName = Aline[0].split('\\')[-1]
        ImgFilePath = KeyWord + '/IMG/' + ImgFileName
        
        #找到对应的图像文件
        Img_file = rarf.open(ImgFilePath)
        AImg = plt.imread(Img_file)
        AImg = AImg[np.newaxis,:,:,:]
        Images = np.row_stack((Images,AImg))
        
        #找打对应的角度指令
        A_angle = np.float32(float(Aline[3]))
        steering_angle.append(A_angle)
        
        counter = counter + 1
        print(counter)
        
    return Images,steering_angle

def AddToPickle(Images,steering_angle,FileName):
    '''
    多次调用在相同Pickle文件中累加数据
    '''
    pickle_file = './Data/TrainData_' + FileName
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

    
#%%数据文件导入区
Left_Images, Left_steering_angle = LoadFiles('./Data/Left.rar','Left')
AddToPickle(Left_Images,Left_steering_angle,'Left')
Right_Images, Right_steering_angle = LoadFiles('./Data/Right.rar','Right')
AddToPickle(Right_Images,Right_steering_angle,'Right')


#%%临时保存数据


