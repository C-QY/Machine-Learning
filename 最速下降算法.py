
# 目标函数：f(x) = x^2 + 1
# 参数为自变量，标量
# 返回目标函数值，标量
def objectiveFunction1D(theta):
    return theta**2 + 1


# 梯度函数：f'(x) = 2x
# 参数为自变量，标量
# 返回函数在自变量处的梯度
def gradientFunction1D(theta):
    return 2 * theta


# 最速下降法
# objFun：目标函数
# gradFun：梯度函数
# theta0：算法初始点，默认值为0
# learningRate：学习率，默认值为1e-3
# precision：精度，默认为1e-4
# maxIteration：最大迭代数，默认为1e4
# 返回目标函数的局部最小解，标量
def gradientDescent(objFun, gradFun, theta0=0, learningRate=1e-3, precision=1e-4, maxIteration=int(1e4)):
    theta = theta0
    for i in range(maxIteration):
        gradValue = gradFun(theta)
        theta = theta - learningRate * gradValue
        if abs(gradValue) < precision:
            break
        print('\n{}次迭代后，局部最优解为：{}，函数值为{}。'.format(i+1, theta, objFun(theta)))


if __name__ == '__main__':
    gradientDescent(objectiveFunction1D, gradientFunction1D, 1)