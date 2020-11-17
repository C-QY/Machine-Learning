import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def raw_data(path):
    data=pd.read_csv(path,names=['size','bedroom','price'])
    return data


# 数据归一化
def normalize_data(data):
    data2=(data-data.mean())/(data.max()-data.min())
    return data2


def cost_function(theta,x,y):
    j=np.sum(np.power(x.dot(theta)-y,2))/(2*x.shape[0])
    return j


def gradient_descent(theta,x,y):
    a=0.01
    epoch=1000
    cost=[]
    for i in range(epoch):
        theta=theta-(((x.dot(theta)-y).ravel()).dot(x))*a/x.shape[0]
        cost.append(cost_function(theta,x,y))
    return theta,cost


def draw_iteration(cost,epoch=1000):
    plt.plot(np.arange(0,epoch),cost)
    plt.title('iteration and cost')
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()


def normal_equation(theta,x,y):
    theta=((np.linalg.inv((x.T).dot(x))).dot(x.T)).dot(y)
    return theta


def main():
    rawdata=raw_data('ex1data2.txt')
    normaldata=normalize_data(rawdata)
    # print(normaldata.head())
    x1=normaldata['size']
    x2=normaldata['bedroom']
    y=normaldata['price']
    x=np.c_[np.ones(x1.shape[0]),x1,x2]
    theta=np.ones(x.shape[1])
    j=cost_function(theta,x,y)
    # print(j)
    theta,cost=gradient_descent(theta,x,y)
    draw_iteration(cost)
    # 梯度下降的代价
    print(cost_function(theta,x,y))
    theta=normal_equation(theta,x,y)
    # 正规方程的代价
    print(cost_function(theta,x,y))
	#0.0156070718918562
	#0.007274047883954535

main()
