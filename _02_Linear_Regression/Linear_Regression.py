# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y = read_data()
    lambd = -0.1
    weight = np.dot(np.linalg.inv((np.dot(x.T,x)+np.dot(lambd,np.eye(6)))),np.dot(x.T,y))
    return weight @ data
    
def lasso(data):
    label = 2e-8
    m = 6
    x,Y = read_data()
    #weight = np.ones([1,6])
    weight = np.zeros(6)
    y = np.dot(weight,x.T)
    lambd = 0.5
    loss = (np.sum(y - Y)**2) / 6 + lambd * np.linalg.norm(y-Y,ord = 1) / 12
    dw = np.dot((y - Y),x)
    #dw = np.dot(x,np.dot(x.T,weight) - y) / n + lambd * np.sign(weight)
    rate = 1e-10
    for i in range(int(2e5)):
        y = np.dot(weight, x.T)
        loss = (np.sum(y - Y) ** 2) / 6 + lambd * np.linalg.norm(y-Y,ord = 1) / 12
        if abs(loss) < label:
            break
        dw = np.dot((y - Y),x)
        #dw = np.dot(x,np.dot(x.T,weight) - y) / n + lambd * np.sign(weight)
        #weight = weight * ( 1 - (rate * lambd / 6)) - dw * rate
        weight = weight - rate * dw
    return weight @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
