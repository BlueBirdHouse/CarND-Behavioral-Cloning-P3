'''
利用PID控制器控制速度，利用传回的图像计算角度，然后将计算结果传回模拟器。
'''

#%%包引用功能区
#通过对源码做ASCII编码，利用文本协议传输二进制内容
import base64

import numpy as np

#网络传输端口
import socketio

#可以实现对网络的并发访问
import eventlet
import eventlet.wsgi

#PIL不能在Python3中运行!!!
from PIL import Image

#可以快速生成一个Web应用程序。可以根据访问的目录不同返回不同的网页。
from flask import Flask

from io import BytesIO

from keras.models import load_model

import matplotlib.pyplot as plt

import time

#%%类定义区
class SimplePIController:
#一个PI控制器
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        #设置目标点
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

#%%功能实现区
#在本地开一个服务器，然后启动一个Web应用程序并起名为'root'
sio = socketio.Server()
app = Flask(__name__)
#生成一个速度控制器，并顶点到速度9上
controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)

model_Path = './Model/Accept3.h5'
model = load_model(model_Path)

#%%事件定义区
#当发生遥测事件的时候
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        #data应该是从服务器传来的信息
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        #利用带缓冲的方法，将b64de编码以后的图像解码出来
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        '''
        plt.imshow(image_array)
        plt.show()
        time.sleep(0.1)
        '''
        steering_angle = float(model.predict(image_array[np.newaxis,:,:,:], batch_size=1))

        #用油门PID控制器计算的当前的油门指令
        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)
        print('由于没有数据传入，这次控制被取消。')


@sio.on('connect')
#一旦客户端连接，就立即发送停车信息
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    #发送转交和油门信息
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

# wrap Flask application with engineio's middleware
#服务其和Web应用程序捆绑起来
app = socketio.Middleware(sio, app)
# deploy as an eventlet WSGI server
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)