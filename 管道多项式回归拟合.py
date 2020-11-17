import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error  # 引入均方误差用来测试拟合的分数
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(666)  # 为了反复测试，这里将随机种子固定
x = np.random.uniform(-3., 3., size=100)
X = x.reshape(-1, 1)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

def PolynomialRegression(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
poly_reg = PolynomialRegression(degree=10)
poly_reg.fit(X,y)
y_predict = poly_reg.predict(X)
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],color='r')
plt.show()
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=10)

def plot_learning_curve(algo,x_train,x_test,y_train,y_test): #algo代表不同算法
    train_score = []
    test_score = []
    for i in range(1,len(x_train)+1):
        algo.fit(x_train[:i],y_train[:i])
        y_train_predict = algo.predict(x_train[:i])
        y_test_predict = algo.predict(x_test)
        train_score.append(mean_squared_error(y_train_predict,y_train[:i]))
        test_score.append(mean_squared_error(y_test_predict,y_test))
    plt.plot([i for i in range(1,76)],np.sqrt(test_score),label='train')
    plt.plot([i for i in range(1,76)],np.sqrt(train_score),label='test')
    plt.legend()
    plt.axis([0,len(x_train)+1,0,4])#限定坐标显示的范围，因为主要还是比较t_train和t_test相近的地方。
    plt.show()
def PolynomialRegression(degree):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std_scaler",StandardScaler()),
        ("lin_reg",LinearRegression())
    ])
poly2_reg = PolynomialRegression(degree=2)
plot_learning_curve(poly2_reg,x_train,x_test,y_train,y_test) #可以看到稳定的值比线性回归低




