import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 导入测试数据
iris = load_iris()
# 数据行
rows = iris.data
# 获取类别数据，这里注意的是已经经过了处理，target里0、1、2分别代表三种类别
label = iris.feature_names
# 获取类别名字（相当于列名）
target = iris.target
data1 = pd.DataFrame(data= np.c_[iris.data, iris.target],
                     columns= iris.feature_names + ['Species'])
print(data1)
# 分类器子函数
# 根据“近朱者赤”原则，找最近距离的k个训练样本，取其标签作为预测样本的标签
def getPredictLabel(trainData, testFeature, k):
    distanceList = []
    closerlist = []
    for idx, row in trainData.iterrows():
        # 训练样本特征
        trainFeature = row[label].values
        # 计算距离
        distance = euclidean(trainFeature, testFeature)
        distanceList.append(distance)
    # 最短距离
    for i in range(k):
        pos = np.argmin(distanceList)
        predLabel = trainData.iloc[pos][['Species']].values
        closerlist.append(str(predLabel))
        distanceList[pos] = 500
    return closerlist
# 结果判定子函数
# 若正确标签在k近邻中占最多，则判定为对
def right(predLabel, trueLabel):
    count = {'[0.]' : 0, '[1.]' : 0, '[2.]' : 0}
    for i in predLabel:
        count[i] = count[i] + 1
    for value in count.values():
        if count[trueLabel] < value:
            return False
    return True
# 分类器主函数
def main():
    # 加载数据
    k = 8
    irisData = data1
    print(type(irisData))
    # 分离数据
    trainData, testData = train_test_split(irisData, test_size=0.50, random_state=12)
    # 分离器，同时计算准确率
    accurateAccount = 0
    for idx, row in testData.iterrows():
        # 获取特征
        testFeature = row[label].values
        # 预测标签
        predLabel = getPredictLabel(trainData, testFeature, k)
        # 实际标签
        trueLabel = row[['Species']].values
        # 输出计算
        if right(predLabel, str(trueLabel)):
            accurateAccount = accurateAccount + 1
        print("标签预测值{}，真实值{}".format(predLabel, trueLabel))
    # 输出准确率
    testSampleNumbers = testData.shape[0]
    print("预测准确率为：{:.2f}%".format(accurateAccount / testSampleNumbers * 100))
# 测试
if __name__ == "__main__":
    main()
