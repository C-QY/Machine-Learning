import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json


# 读取图片数据
def read_image_data(image_name):
    image = Image.open(image_name)
    
    # 将图片大小变为224*224的灰度图片
    new_x = 224
    new_y = 224
    image = image.resize((new_x, new_y), Image.ANTIALIAS).convert('L')
    
    # plt.imshow(image)
    
    # 将图片对象转为列表
    image_list = np.asarray(image)
    
    # 将列表变为一维
    image_list = image_list.flatten()
    
    # 返回处理好的一维列表
    return image_list


# 将所有图片转为列表，获取数据集
def get_images_list(num_per_kind):
    data = []
    label = []
    for i in range(num_per_kind):
        # 拼接狗图片名，并将数据集和标签集分别存储，狗标签为1
        dog_image_file_name = r'train/dog.{}.jpg'.format(i)
        data.append(read_image_data(dog_image_file_name))
        label.append(1)
        # 拼接猫图片名，并将数据集和标签集分别存储，猫标签为0
        cat_image_file_name = r'train/cat.{}.jpg'.format(i)
        data.append(read_image_data(cat_image_file_name))
        label.append(0)
    # 返回数据集和标签集
    return np.array(data), np.array(label)


# 分类网络
class ClassifierNetwork:
    # 前向传播函数
    # - x：包含输入数据的numpy数组，形状为（N，d_1，...，d_k）
    # - w：形状为（D，M）的一系列权重
    # - b：偏置，形状为（M，）
    def affine_forward(self, x, w, b):   
        out = None                       # 初始化返回值为None
        N = x.shape[0]                   # 重置输入参数X的形状
        x_row = x.reshape(N, -1)         # (N,D)
        out = np.dot(x_row, w) + b       # (N,M)
        cache = (x, w, b)                # 缓存值，反向传播时使用
        return out,cache

    # 反向传播函数
    # - x：包含输入数据的numpy数组，形状为（N，d_1，...，d_k）
    # - w：形状（D，M）的一系列权重
    # - b：偏置，形状为（M，）
    def affine_backward(self, dout, cache):   
        x, w, b = cache                              # 读取缓存
        dx, dw, db = None, None, None                # 返回值初始化
        dx = np.dot(dout, w.T)                       # (N,D)
        dx = np.reshape(dx, x.shape)                 # (N,d1,...,d_k)
        x_row = x.reshape(x.shape[0], -1)            # (N,D)
        dw = np.dot(x_row.T, dout)                   # (D,M)
        db = np.sum(dout, axis=0, keepdims=True)     # (1,M)
        return dx, dw, db

    # 初始化一些参数
    # 传入训练数据以及标签，可选传入本地保存的模型路径，就会加载本地保存的模型，否则将随机初始化
    def __init__(self, X, label, model_file_name=False):
        self.X = X
        self.label = label

        np.random.seed(1)         # 指定随机种子，固定随机数结果

        # 一些初始化参数  
        self.input_dim = self.X.shape[1]         # 输入参数的维度
        self.num_classes = self.label.shape[0]   # 输出参数的维度
        self.hidden_dim = 50            # 隐藏层维度，为可调参数
        self.reg = 0.001                # 正则化强度，为可调参数
        self.epsilon = 0.001            # 梯度下降的学习率，为可调参数
        # 初始化W1，W2，b1，b2
        # 判断是否需要从本地加载模型
        if not model_file_name:
            # 不需要加载模型，随机初始化参数
            self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
            self.W2 = np.random.randn(self.hidden_dim, self.num_classes)
            self.b1 = np.zeros((1, self.hidden_dim))
            self.b2 = np.zeros((1, self.num_classes))
        else:
            # 需要加载模型，通过加载模型方法来获取
            model_dict = self.load_model(model_file_name)
            self.W1 = np.array(model_dict['W1'])
            self.W2 = np.array(model_dict['W2'])
            self.b1 = np.array(model_dict['b1'])
            self.b2 = np.array(model_dict['b2'])



    # 训练方法
    def train(self):
        for epoch in range(10000):   #这里设置了训练的循环次数为10000
        # ①前向传播
            H, fc_cache = self.affine_forward(self.X, self.W1, self.b1)           # 第一层前向传播
            H = np.maximum(0, H)                                                  # 激活
            relu_cache = H                                                        # 缓存第一层激活后的结果
            Y, cachey = self.affine_forward(H, self.W2, self.b2)                  # 第二层前向传播
        # ②Softmax层计算
            probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))    
            probs /= np.sum(probs, axis=1, keepdims=True)                 # Softmax算法实现
        # ③计算loss值
            N = Y.shape[0]                                                # 值为4
            # print(probs[np.arange(N), t])                               # 打印各个数据的正确解标签对应的神经网络的输出
            loss = -np.sum(np.log(probs[np.arange(N), self.label])) / N   # 计算loss
            print(epoch, '-->', loss)                                     # 打印loss
        # ④反向传播
            dx = probs.copy()                                             # 以Softmax输出结果作为反向输出的起点
            dx[np.arange(N), self.label] -= 1
            dx /= N                                                       # 到这里是反向传播到softmax前
            dh1, dW2, db2 = self.affine_backward(dx, cachey)              # 反向传播至第二层前
            dh1[relu_cache <= 0] = 0                                      # 反向传播至激活层前
            dX, dW1, db1 = self.affine_backward(dh1, fc_cache)            # 反向传播至第一层前
        # ⑤参数更新
            dW2 += self.reg * self.W2
            dW1 += self.reg * self.W1
            self.W2 += -self.epsilon * dW2
            self.b2 += -self.epsilon * db2
            self.W1 += -self.epsilon * dW1
            self.b1 += -self.epsilon * db1
    
    # 预测
    def predict(self, test):
        H,fc_cache = self.affine_forward(test, self.W1, self.b1)               #仿射
        H = np.maximum(0, H)                                  #激活
        relu_cache = H
        Y,cachey = self.affine_forward(H, self.W2, self.b2)  #仿射
        # Softmax
        probs = np.exp(Y - np.max(Y, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)  # Softmax
        # print(probs)
        for k in probs:
            if np.argmax(k) == 1:
                print(np.argmax(k), '图片为狗，可能性为', max(k))
            else:
                print(np.argmax(k), '图片为猫，可能性为', max(k))
    
    # 保存模型
    def save_model(self, file_name='saved_model.json'):
        # 建立要保存的字典
        save_dict = {
            'W1': self.W1.tolist(),
            'W2': self.W2.tolist(),
            'b1': self.b1.tolist(),
            'b2': self.b2.tolist(),
        }
        with open(file_name, 'w') as fp:
            fp.write(json.dumps(save_dict))
    
    # 加载模型
    def load_model(self, file_name='saved_model.json'):
        with open(file_name, 'r') as fp:
            model_dict = json.loads(fp.read())
        return model_dict


def main():
    # 不加载模型过程
    # data, label = get_images_list(12500)
    # my_classifier = ClassifierNetwork(data, label)
    # my_classifier.train()
    # my_classifier.save_model()

    # 仅加载本地模型不训练过程
    data, label = get_images_list(1)
    my_classifier = ClassifierNetwork(data, label, 'saved_model.json')

    # 加载本地模型并继续训练过程
    # data, label = get_images_list(12500)
    # my_classifier = ClassifierNetwork(data, label, 'saved_model.json')
    # my_classifier.train()
    # my_classifier.save_model()

    
    # 预测十张图
    image_1 = r'test1/1.jpg'
    image_2 = r'test1/2.jpg'
    image_3 = r'test1/3.jpg'
    image_4 = r'test1/4.jpg'
    image_5 = r'test1/5.jpg'
    image_6 = r'test1/6.jpg'
    image_7 = r'test1/7.jpg'
    image_8 = r'test1/8.jpg'
    image_9 = r'test1/9.jpg'
    image_10 = r'test1/10.jpg'
    test_data = []
    test_data.append(read_image_data(image_1))
    test_data.append(read_image_data(image_2))
    test_data.append(read_image_data(image_3))
    test_data.append(read_image_data(image_4))
    test_data.append(read_image_data(image_5))
    test_data.append(read_image_data(image_6))
    test_data.append(read_image_data(image_7))
    test_data.append(read_image_data(image_8))
    test_data.append(read_image_data(image_9))
    test_data.append(read_image_data(image_10))
    test_data = np.array(test_data)

    my_classifier.predict(test_data)
    


if __name__ == '__main__':
    main()