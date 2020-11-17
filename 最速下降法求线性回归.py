import matplotlib.pyplot as plt
import numpy as np
#从.txt文件中读取数据
def loadData(flieName):
    file = open(flieName, 'r')
    #定义两个空list，存放文件中的数据
    x = []
    y = []
    for line in file:
        trainingSet = line.split(',')  #对于每一行，按','把数据分开
        x.append(trainingSet[0])  #文件中的第一列数据逐一添加到 X 中
        y.append(trainingSet[1])  #文件中的第二列数据逐一添加到 y 中
    return (x, y)  #x,y组成一个元组，通过函数一次性返回

def displayData(x,y):
    plt.scatter(x,y,20)
    plt.title("training data")
    plt.xlabel("population")
    plt.ylabel("profit")
    #设置坐标的最小值和最大值
    plt.axis([4.0,23.0,-5.0,25.0])
    #plt.show()

def displayFit(x,y,theta):
    thetalist=theta.tolist()
    plt.scatter(x, y,c='#ee8640',s=20)
    plt.title("gradientDescent")
    plt.xlabel("population")
    plt.ylabel("profit")
    # 设置坐标的最小值和最大值
    plt.axis([4.0, 23.0, -5.0, 25.0])
    x1=np.arange(4.5,25,0.01)
    h=thetalist[0][0]+x1*thetalist[0][1]
    plt.plot(x1,h)
    plt.show()

def displayNormal(x,y,theta):
    list=theta.tolist()
    plt.scatter(x, y,c='#ee8640',s=20)
    plt.title("NormalEquation")
    plt.xlabel("population")
    plt.ylabel("profit")
    # 设置坐标的最小值和最大值
    plt.axis([4.0, 23.0, -5.0, 25.0])
    x1 = np.arange(4.5, 25, 0.01)
    h = list[0][0] + x1 * list[1][0]
    plt.plot(x1, h)
    plt.show()

#计算代价函数j
def computerCostJ(X,Y,thea):
    sum=np.sum(np.power(X*thea.T-Y,2))
    j=sum/(2*len(X))
    return j

#梯度下降法求theta矩阵
def gradientDescent(X,Y,theta):
    #设置学习速率alpha以及迭代次数epoch,每次的代价j放入cost中
    alpha=0.01
    epoch=1000
    cost=[]
    m=len(X)
    for i in range(epoch):
        temp=theta-(alpha/m)*(X*(theta.T)-Y).T*X
        theta=temp
        cost.append(computerCostJ(X,Y,theta))
    return theta,cost

#正规方程法
def normalEquation(X,Y,theta):
    theta=np.linalg.inv(X.T*X)*X.T*Y
    return theta

def main():
    (lx,ly)=loadData("C:/Users\CQY\Desktop\CQY\学习\大三\机器学习\ex1data1.txt")
    lx=[float(x) for x in lx]
    ly=[float(x) for x in ly]
    cost=[]
    displayData(lx,ly)
    #将list转换为矩阵
    matx=np.mat(lx).T
    Y=np.mat(ly).T
    #向训练集中加一列1,生成训练集X
    matz=np.ones(len(lx),int).T
    X=np.c_[matz,matx]
    #初始化theta
    theta=np.mat([[0,0]])
    #检查维度,(97, 2) (1, 2)
    #print(X.shape,theta.shape)
    #计算初始代价函数(初始化theta矩阵时),32.072733877455676
    j=computerCostJ(X,Y,theta)
    #梯度下降
    (theta,cost)=gradientDescent(X,Y,theta)
    displayFit(lx,ly,theta)
    #正规方程
    theta=normalEquation(X,Y,theta)
    displayNormal(lx,ly,theta)
    '''
    #查看列表中对最大值和最小值，设置坐标轴的最小值和最大值
    slx=sorted(lx)
    sly=sorted(ly)
    print(slx[0],slx[-1])
    print(sly[0],sly[-1])
    '''

main()


