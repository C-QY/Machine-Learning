import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from  matplotlib import cm
import pandas as pd
import seaborn as sns
#导入数据
dataFile = 'C:/Users\CQY\Desktop\LogisticRegression_Data1.txt'
data = pd.read_csv(dataFile,names=['exam1','exam2','admitted'])
data.describe()
#可视化测试数据
sns.set(context="notebook",style="darkgrid",palette=sns.color_palette("RdBu",2))
sns.lmplot('exam1','exam2',hue='admitted',data=data,
           size=6,
           fit_reg=False,
           scatter_kws={"s":50}
           )
plt.show()
#Sigmoid函数
def Sigmoid(z):
    return 1/(1+np.exp(z))
#假设函数：h(Θ,x)=σ(x^T,Θ)
def h(Theta,x):
    return Sigmoid(np.dot(x.T,Theta))
#成本函数
def J(Theta, X,y):
    J1 = np.dot(np.log(h(Theta, X)).T,y)
    J2 = np.dot(1-np.log(h(Theta, X)).T, 1-y)
    return (J1 + J2).len(y)
#梯度计算公式
def GD(Theta, X,y):
    gd = np.dot(X,h(Theta, X)-y)
    return  gd/len(y)
#梯度下降法
def GDA(X, y, learningRate = 0.2, precision=1e-4, maxIteration = 10000):
    Theta0 = np.zeros((X.shape[0],1)).reshape(-1,1)
    J0 = J(Theta0, X, y)
    Theta1 = np.zeros((X.shape[0],1)).reshape(-1,1)

    listTheta = []
    listJ = []
    listIter = []
    listTheta.append(Theta0)
    listJ.append(J0)
    listIter.append(0)

    for k in range(1,maxIteration):
        gd= GD(Theta0, X, y)
        Theta1= Theta0 -learningRate*gd
        J1 = J(Theta1, X, y)
        if (np.linalg.norm(gd,ord=2)<precision) and (np.linalg.norm(J1-J0)<precision):
            break

        Theta0 = Theta1
        J0=J1
        listTheta.append(Theta0)
        listJ.append(J0)
        listIter.append(k)
    return  Theta1,np.array(listTheta), np.array(listJ),np.array(listIter)


