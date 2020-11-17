
import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
# 加载数据集
boston =datasets.load_boston()
# 提取特征以及价格
boston_X = boston.data
boston_y = boston.target
# 将特征以及价格存入DataFrame
boston_data = pd.DataFrame(boston_X)
boston_data.columns = boston.feature_names  # 设置表头
boston_data['price'] = boston_y  # 增加价格数据项
# 选取输入特征和结果，这里13项全选
data_X=boston_data[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
data_y=boston_data[['price']]
# 拆分训练/测试数据集
X_train,X_test,y_train,y_test = train_test_split(
    data_X, data_y, test_size = 0.1, random_state=12
)
# 实例化线性回归模型
model=linear_model.LinearRegression()
# 将训练数据传入模型开始训练
model.fit(X_train,y_train)
print('模型权重为：', model.coef_)     # 系数
print('模型偏置为：', model.intercept_) # 截距
print('训练数据准确率为：', model.score(X_train, y_train))
print('测试数据准确率为：', model.score(X_test, y_test))
