import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from  matplotlib import cm
import pandas as pd
import seaborn as sns

m=100
pa,pb=-3,3
X = np.random.uniform(pa,pb,size=m).reshape(-1,1)
plt.scatter(X[:,-1], y)
plt.axis([pa,pb,5*pa+3,5*pb+3])
plt.show()


#假设函数：h(Θ,x)=σ(x^T,Θ)
def h(theta,X):
    return theta[0] +theta[1]*X
#损失函数
def L(theta,X,y):
    return (h(theta,X)-y)**2
#成本函数
def J(theta, X,y):
    m = len(y)
    jSum = 0
    for i in range(m):
        jSum += L(theta,X[i],y[i])
    return jSum / (2*m)

#梯度计算函数
def GD(theta, X,y):
    m = len(y)
    gd = np.zeros((2,1),dtype=float).reshape(-1,1)
    for i in range(m):
        gd[0] += h(theta,X[i])-y[i]
        gd[1] += (h(theta,X[i]) - y[i])*X[i]
    return  gd/m

#梯度下降法
def GDA(X, y, thetaInit = [0,0], learningRate = 0.2, precision=1e-4, maxIteration = 10000):
    theta0 = np.array(thetaInit).reshape(-1,1)
    J0 = J(theta0, X, y)
    theta1 = np.zeros((2,1)).reshape(-1,1)

    listTheta = []
    listJ = []
    listIter = []
    listTheta.append(theta0)
    listJ.append(J0)
    listIter.append(0)

    for k in range(1,maxIteration):
        gd = GD(theta0,X,y)
        theta1 = theta0 -learningRate*gd
        J1 = J(theta1, X, y)
        if(np.linalg.norm(gd,ord=1) < precision) and (np.linalg.norm(J1 -J0) < precision):
            break
        theta0 = theta1
        j0 = J1
        listTheta.append(theta0)
        listJ.append(J0)
        listIter.append(k)
    return theta1,np.array(listTheta),np.array(listJ),np.array(listIter)

coef, thetaList, JList, iterList = GDA(X,y,learningRate=0.1)
thetaList.reshape(-1,2)
JList.reshape(-1,1)
iterList.reshape(-1,1)

fig = plt.figure()

fig.add_subplot(2,2,1)
plt.scatter(X[:,-1], y)
plt.axis([pa,pb,5*pa+3,5*pb+3])
X_plot = np.linspace(pa,pb,100)
y_plot = coef[0] + coef[1]*X_plot
plt.plot(X_plot,y_plot,color='red')

fig.add_subplot(2,2,2)
plt.plot(iterList,JList)

fig.add_subplot(2,2,3)
plt.plot(thetaList[:,0],thetaList[:,1])

theta0 = np.linspace(abs(np.min(thetaList[:, 0]))-1, abs(np.max(thetaList[:,0])+1), 50)
theta1 = np.linspace(abs(np.min(thetaList[:, 1]))-1, abs(np.max(thetaList[:,1])+1), 50)
XLen = len(theta0)
YLen = len(theta1)
XX, YY =np.meshgrid(theta0,theta1)
JPlotList = np.zeros(XLen,YLen)
for i in range(XLen):
    for j in range(YLen):
        JPlotList[i,j] = J(theta0[i],theta1[j],X, y)

ax = fig.add_subplot(2,2,4,projection='3d')
ax.plot_surface(XX,YY,JPlotList,cmap=cm.rainbow)
ax.contour(XX,YY,JPlotList,zdir = 'z',offset = 2,cmap =plt.get_cmap('rainbow'))
ax.scatter(thetaList[:,0],thetaList[:,1],JList,color='black',marker='o')
plt.show()