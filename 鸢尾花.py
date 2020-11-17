import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataFile = 'C:/Users\CQY\Desktop/iris.data'

Species = ['Iris-setosa',
           'Iris-versicolor',
           'Iris-virginica'
           ]

featureColumns = ['SepalLengthInCm', 'SepalWidthInCm', 'PetalLengthInCm', 'PetalWidthInCm']
specialColums = ['Species']


# 预测方法
# 参数：
# trainData：训练数据集
# testFeature：测试特征
# predNum：需要预测的数量(距离最近的标签数)
def getPredictLabel(trainData, testFeature, predNum):
    distanceList = []
    for idx, row in trainData.iterrows():
        # 训练样本特征
        trainFeature = row[featureColumns].values
        # 计算距离
        distance = euclidean(trainFeature, testFeature)
        distanceList.append(distance)
    # 最短距离
    # 遍历predNum次，每次找出最小距离的下标，然后将它的值改为inf，这样下一次找最小值就可以得到第二小
    predLabel = []
    for _ in range(predNum):
        pos = np.argmin(distanceList)  # 找出最小值的下标
        distanceList[pos] = np.inf  # 将当前最小值下标改为inf
        predLabel.append(str(trainData.iloc[pos][specialColums].values))  # 将预测标签存入数组
    # 返回数据
    return predLabel


# 分类器主函数
# 参数：predNum：需要预测的数量(距离最近的标签数)
def main(predNum):
    # 加载数据
    irisData = pd.read_csv(dataFile)
    # 分离数据
    trainData, testData = train_test_split(irisData, test_size=0.50, random_state=12)
    # 分离器，同时计算准确率
    accurateAccount = 0
    for idx, row in testData.iterrows():
        # 获取特征
        testFeature = row[featureColumns].values
        # 预测标签
        predLabel = getPredictLabel(trainData, testFeature, predNum)
        # 实际标签
        trueLabel = row[specialColums].values
        # 输出计算
        # 如果真实值在预测值列表中，算作预测正确
        if str(trueLabel) in predLabel:
            accurateAccount = accurateAccount + 1
        print('标签预测值{}，真实值{}'.format(predLabel, trueLabel))

    # 输出准确率
    testSampleNumbers = testData.shape[0]
    print('预测准确率为：{:.2f}%'.format(accurateAccount / testSampleNumbers * 100))


# 测试
if __name__ == '__main__':
    predNum = 5
    main(predNum)