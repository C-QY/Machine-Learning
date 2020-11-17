import numpy as np
from matplotlib import pyplot as plt

with open('LogisticRegression_Data1.txt') as object_file:
    lines = object_file.readlines()
ex2data1 = np.zeros(shape=(len(lines), 3))
m = len(lines)
i = 0
for line in lines:
    line = line.strip('\n')
    line = line.split(',')
    line = [float(x) for x in line]
    ex2data1[i] = line
    ex2data1_r3 = ex2data1[:, 2]
    ex2data1_r2 = ex2data1[:, 1]
    ex2data1_r1 = ex2data1[:, 0]
    if ex2data1_r3[i] == 0:
        plt.scatter(ex2data1_r1[i], ex2data1_r2[i], c='r', marker='o')
    else:
        plt.scatter(ex2data1_r1[i], ex2data1_r2[i], c='b', marker='x')
    i += 1


def hx(theta, X):
    z = np.dot(X, theta.T)
    hx = 1 / (1 + np.exp(-z))
    return hx


J = 0
theta = [1, 1, 1]
theta = np.array(theta)
ex2data1 = np.array(ex2data1)
k = 0
alpha = 0.1
X1 = ex2data1[:, :2]
X = np.ones((100, 3))
X[:, 1:3] = X1
Y = ex2data1[:, -1]
"""def J(theta,X,y):
    hx2=hx(theta,X)
    try:
        return -np.sum(y*np.log(hx2)+(1-y)*np.log(1-hx2))/len(y)
    except:
        return float("inf")"""
while k < 1000000:
    hx1 = hx(theta, X)
    gradient_J = np.dot(hx1 - Y, X) / m
    last_theta = theta
    theta = theta - alpha * gradient_J
    k += 1
    """if abs(J(theta,X,Y)-J(last_theta,X,Y))<0.0001:
        break"""

print(theta)
x = np.linspace(0, 100, 1000)
plt.plot(x, -theta[0] / theta[2] - x * theta[1] / theta[2])
plt.show()