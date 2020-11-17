import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
# 读取原始数据
def raw_data(path):
    data=pd.read_csv(path,names=['population','profit'])
    return data


def main():
    data=raw_data('ex1data1.txt')

    X=data['population']
    y=data['profit']
    X=np.c_[np.ones(X.size),X]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    #调用LinearRegression
    clf = linear_model.LinearRegression()
    clf.fit(X, y)
    print(clf.coef_)
    print(clf.intercept_)


    #调用Ridge
    model = Ridge(alpha=0.01, normalize=True)
    model.fit(X_train, y_train)
    train_score_R = model.score(X_train, y_train)  # 模型对训练样本得准确性
    test_score_R = model.score(X_test, y_test)  # 模型对测试集的准确性
    print(train_score_R)
    print(test_score_R)
    # 调用RidgeCV
    model = RidgeCV(alphas=[1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001], normalize=True)
    model.fit(X_train, y_train)
    print(model.alpha_)
    # 调用 LassoCV
    lscv = LassoCV(alphas=(1.0, 0.1, 0.01, 0.001, 0.005, 0.0025, 0.001, 0.00025), normalize=True)
    lscv.fit(X,y)
    print('Lasso optimal alpha: %.3f' % lscv.alpha_)
    #ElasticNetCV
    encv = ElasticNetCV(alphas=(0.1, 0.01, 0.005, 0.0025, 0.001), l1_ratio=(0.1, 0.25, 0.5, 0.75, 0.8), normalize=True)
    encv.fit(X,y)
    print('ElasticNet optimal alpha: %.3f  L1 ratio: %.4f' % (encv.alpha_, encv.l1_ratio_))



main()
