import numpy as np
import pandas as pd


# sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid导函数
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


# mse损失函数
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


# 神经元对象
# class Neuron:
#     def __init__(self, weights, bias):
#         self.weights = weights
#         self.bias = bias

#     def feedforward(self, inputs):
#         total = np.dot(self.weights, inputs)
#         return sigmoid(total)

# # 网络对象
# class OurNeuralNetwork:
#     def __init__(self):
#         weights = np.array([0, 1])
#         bias = 0

#         # 定义三个神经元
#         self.h1 = Neuron(weights, bias)
#         self.h2 = Neuron(weights, bias)
#         self.o1 = Neuron(weights, bias)

#     def feedforward(self, x):
#         out_h1 = self.h1.feedforward(x)
#         out_h2 = self.h2.feedforward(x)

#         out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

#         return out_o1

class OurNeuralNetwork:
    def __init__(self):
        # 权重
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # 截距
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learning_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # o1神经元
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # h1神经元
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # h2神经元
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # 更新权重和截距
                # h1神经元
                self.w1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # h2神经元
                self.w3 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # o1神经元
                self.w5 -= learning_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learning_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learning_rate * d_L_d_ypred * d_ypred_d_b3

            # 计算损失
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print('Epoch {} loss: {:.3f}'.format(epoch, loss))


# 测试网络方法
def test_network():
    data = np.array([
        [-2, -1],
        [25, 6],
        [17, 4],
        [-15, -6],
    ])
    all_y_trues = np.array([
        1,
        0,
        0,
        1,
    ])

    network = OurNeuralNetwork()
    network.train(data, all_y_trues)

    # 做一些预测
    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    print("Emily: %.3f" % network.feedforward(emily))
    print("Frank: %.3f" % network.feedforward(frank))


def main():
    file_data = pd.read_csv('NeuralNetwork_Data1.csv')
    data = np.array(file_data[['x1', 'x2']])
    all_y_trues = np.array(file_data.y)

    network = OurNeuralNetwork()
    network.train(data, all_y_trues)


if __name__ == "__main__":
    main()
