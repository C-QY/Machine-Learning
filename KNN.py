from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing
# 加载iris数据集
iris = load_wine()
# 提取特征和标签
# X, y = iris.data[:, :2], iris.target
X, y = iris.data, iris.target
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
# 数据预处理
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# 声明并训练模型
KNNClassfiler = neighbors.KNeighborsClassifier(n_neighbors=5)
KNNClassfiler.fit(X_train, y_train)
# 评估模型（方法1）
accuracy = KNNClassfiler.score(X_test, y_test)
print('方法1预测准确率为：{:.2f}%'.format(accuracy*100))
# 预测评估（方法2）
y_pred = KNNClassfiler.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('方法2预测准确率为：{:.2f}%'.format(accuracy*100))
