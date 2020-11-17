import numpy as np
from matplotlib import pyplot

theta = []  # 存储theta0和theta1的中间结果
area = [150, 200, 250, 300, 350, 400, 600]  # 数据
price = [6450, 7450, 8450, 9450, 11450, 15450, 18450]


def BGDSolve():  # 批量梯度下降
    alpha = 0.00000001  # 步长
    kec = 0.00001  # 终止条件
    theta0 = 7  # 初始值
    theta1 = 7
    m = len(area)  # 数据个数
    theta.append((theta0, theta1))
    while True:
        sum0 = sum1 = 0
        # 计算求和求导过的损失函数
        for i in range(m):
            sum0 = sum0 + theta0 + theta1 * area[i] - price[i]
            sum1 = sum1 + (theta0 + theta1 * area[i] - price[i]) * area[i]
        theta0 = theta0 - sum0 / m * alpha  # 公式上是 alpha/m * sum0
        theta1 = theta1 - sum1 / m * alpha
        print(theta0, theta1)
        theta.append((theta0, theta1))  # 保存迭代结果
        # 迭代终止条件，变化量小于kec就终止
        if abs(sum0 / m * alpha) < kec and abs(sum1 / m * alpha) < kec:
            return theta0, theta1


def Plot():  # 绘图函数
    theta0, theta1 = BGDSolve()
    pyplot.scatter(area, price)
    x = np.arange(100, 700, 100)
    y = theta0 + theta1 * x
    pyplot.plot(x, y)
    pyplot.xlabel('area')
    pyplot.ylabel('price')
    pyplot.show()


if __name__ == '__main__':
    theta0, theta1 = BGDSolve()
    Plot()
    print(len(theta))

