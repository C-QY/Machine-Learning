import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x + +2 + np.random.normal(0, 1, size=100)
plt.scatter(x, y)
plt.show()

np.random.seed(666)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=10)


def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)), #构造特征
        ("std_scaler", StandardScaler()),#数据归一化
        ("lin_reg", LinearRegression())#线性回归
    ])


def plot_model(model):  # 绘制图像的代码封装，为了方便使用，直接传入模型即可
    x_plot = np.linspace(-3, 3, 100).reshape(100, 1)
    y_plot = model.predict(x_plot)
    plt.scatter(x, y)
    plt.plot(x_plot[:, 0], y_plot, color='r')
    plt.axis([-3, 3, 0, 6])
    plt.show()


poly_reg = PolynomialRegression(degree=50)
poly_reg.fit(X, y)
plot_model(poly_reg)


def RridgeRegression(degree, alpha):  # 修改管道
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),#构造特征
        ("std_scaler", StandardScaler()),# 数据归一化
        ("lin_reg", Ridge(alpha=alpha))  # 加入L2回归，防止过拟合
    ])

ridge1 = RridgeRegression(20, 0.0001)
# 修改alpha值，分别为1,100,100000。可以看出线条逐渐变得平滑，当alpha很大的时候，为了使目标函数小，所以会使系数趋近于0，因此会得出几乎平行的一条直线。
ridge1.fit(x_train, y_train)
y_test = ridge1.predict(x_test)
# mean_squared_error(y_test,x_test)
plot_model(ridge1)
