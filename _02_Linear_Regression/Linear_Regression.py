# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y = read_data()
    lambd = 2e-5
    weight = np.dot(np.linalg.inv(np.dot(np.dot(x.T,x)+np.dot(lambd,np.eye(6)))),np.dot(x.T,y))
    return weight @ data
    
def lasso(data):
    label = 2e-5
    x,Y = read_data()
    weight = np.ones([6,6])
    y = np.dot(weight,x)
    loss = (np.sum(y - Y)**2) / 6
    rate = 1e-10
    for i in range(int(1e9)):
        y = np.dot(weight, x)
        loss = (np.sum(y - Y) ** 2) / 6
        if loss < label:
            break
        dw = np.dot((y - Y),x.T)
        weight = weight - dw * rate
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
