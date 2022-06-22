#
# import numpy
#
# # 定义输入空间，每一个元组的前两个分量分别是x，y，最后一个分量标识正、负实例点
# input_space = [(3, 3, 1), (4, 3, 1), (1, 1, -1)]
#
# # 定义感知机模型 f(x) = sign(w.x+b)   if   w.x+b > 0 f(x)=1  elif   w.x+b<0 f(x)=-1
# def perceptron(x, w, b):
# 	# 实现w向量与x向量的点乘
#     if numpy.dot(w.T, x) + b > 0:
#         return 1
#     elif numpy.dot(w.T, x) + b < 0:
#         return -1
#     # 落在超平面上也是未分类
#     else:
#         return 0
#
# # num记录输入哪个点
# num = 0
#
# # 初始化w向量，偏置b，学习率yi_ta
# # 定义w向量
# w = numpy.array([0, 0])
# b = 0
# yi_ta = 1
#
# while True:
#     # 如果是误分类点，就更新w向量和偏置b，并重新开始检查
#     if input_space[num][2] * perceptron(numpy.array(input_space[num][0:2]), w, b) <= 0:
#         # 向量相加可以通过numpy实现
#         # 因为每一个元组只有前两个分量是坐标，所以用切片取前两个坐标
#         # 采用梯度下降法更新w向量和b			w <- w + y.x     b <- b + y
#         w = w + yi_ta * numpy.array(input_space[num][0:2]) * input_space[num][2]
#         b = b + input_space[num][2]
#         num = 0
#         print("代入(%d,%d)之后更新的w1=%d,w2=%d,b=%d" % (input_space[num][0], input_space[num][1], w[0], w[1], b))
#         continue
#     num += 1
#     if num == len(input_space):
#         break
#
#
# # 假设输入空间是线性可分的
# # 可以得到超平面和感知机模型
#
# print("超平面：%dx1+%dx2+%d=0" % (w[0], w[1], b))
# print("感知机模型：f(x) = sign(%dx1+%dx2+%d)" % (w[0], w[1], b))
#
#
# """
# # @Author: coderdz
# # @File: Perceptron.py
# # @Site: github.com/dingzhen7186
# # @Ended Time : 2020/9/21
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.datasets import load_iris
#
#
# class Perceptron():
#     def __init__(self):
#         # w初始化为全1数组
#         self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
#         self.b = 0
#         self.rate = 0.5  # 初始化学习率
#
#     # 感知机训练, 找出最合适的w, b
#     def fit(self, x_train, y_train):
#         while True:
#             flag = True  # 标记是否存在误分类数据
#             for i in range(len(x_train)):  # 遍历训练数据
#                 xi = x_train[i]
#                 yi = y_train[i]
#                 # 判断 yi * (wx + b) <= 0
#                 if yi * (np.inner(self.w, xi) + self.b) <= 0:
#                     flag = False  # 找到误分类数据, flag标记为False
#                     # 更新w, b值
#                     self.w += self.rate * np.dot(xi, yi)
#                     self.b += self.rate * yi
#             if flag:
#                 break
#         # 输出w = ? , b = ?
#         print('w = ' + str(self.w) + ', b = ' + str(self.b))
#
#     # 图形显示结果
#     def show(self, data):
#         x_ = np.linspace(4, 7, 10)
#         y_ = -(self.w[0] * x_ + self.b) / self.w[1]
#         # 画出这条直线
#         plt.plot(x_, y_)
#         # 画出数据集的散点图
#         plt.plot(data[:50, 0], data[:50, 1], 'bo', c='blue', label='0')
#         plt.plot(data[50:100, 0], data[50:100, 1], 'bo', c='orange', label='1')
#         plt.xlabel('sepal length')
#         plt.ylabel('sepal width')
#         plt.legend()
#         plt.show()
#
#
# iris = load_iris()
# # 通过DataFrmae对象获取iris数据集对象, 列名为该数据集样本的特征名
# df = pd.DataFrame(iris.data, columns=iris.feature_names)
# # 增加label列为它们的分类标签
# df['label'] = iris.target
# df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# # print(df.label.value_counts()) 不同标签的样本数量
# # 2:50 1:50 0:50
#
# # 画出散点图查看特征点的分布
# plt.scatter(df[:50]['sepal length'].values, df[:50]['sepal width'].values, label='0')
# plt.scatter(df[50:100]['sepal length'].values, df[50:100]['sepal width'].values, label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()
#
# # 选择数据集
# data = np.array(df.iloc[:100, [0, 1, -1]])
# # 数据集划分
# x, y = data[:, :-1], data[:, -1]
# # 将y数据集值变为1和-1
# y = np.array([1 if i == 1 else -1 for i in y])
#
# # 开始训练
# p = Perceptron()
# p.fit(x, y)
# p.show(data)
#
# """
#
#
# # """
# # name : demo1.py
# # time : 2021/7/14 18:25
# # author : yiyi
# # desc : please input some description here
# # """
# # import random
# #
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib
# # font = {
# #     "family" : "MicroSoft YaHei",
# #     "size" : "10"
# # }
# # matplotlib.rc("font", **font)
# #
# # # 定义learning rate学习率和初始化w和b
# # yi_ta = 0.1
# # w = np.array([0, 0])
# # b = 0
# # num_positive = 0
# # num_negative = 0
# # x = list()
# # y = list()
# # flag = list()
# #
# # # 定义符号函数
# # def sign(num):
# #     if num > 0:
# #         return 1
# #     elif num < 0:
# #         return -1
# #     else:
# #         return 0
# #
# #
# # # 定义感知机模型
# # def perceptron(w, x, b):
# #     res = w @ x + b
# #     return sign(res)
# #
# #
# # # 定义梯度下降算法
# # def gradient_descent(w, pos, b, flag):
# #     new_w = w + yi_ta * pos * flag
# #     new_b = b + yi_ta * flag
# #     return new_w, new_b
# #
# #
# # def get_input_space():
# #     global num_positive
# #     global num_negative
# #     global x
# #     global y
# #     global flag
# #     while True:
# #         str = input("请输入(x,y)坐标和实例点类型(-1 or 1),输入exit退出！例：2,3,-1")
# #         if str == "exit":
# #             break
# #         data_tmp = str.split(',')
# #         if int(data_tmp[2]) == 1:
# #             x.insert(num_positive, int(data_tmp[0]))
# #             y.insert(num_positive, int(data_tmp[1]))
# #             flag.insert(num_positive, int(data_tmp[2]))
# #             num_positive += 1
# #             continue
# #         x.append(int(data_tmp[0]))
# #         y.append(int(data_tmp[1]))
# #         flag.append(int(data_tmp[2]))
# #
# #
# # if __name__ == "__main__":
# #     get_input_space()
# #
# #     x = np.array(x)
# #     y = np.array(y)
# #     flag = np.array(flag)
# #
# #     print(x)
# #     print(y)
# #     print(flag)
# #     # 第一类，实例点
# #     # x = np.array([3, 4, 1, 0, 0])
# #     # y = np.array([3, 3, 1, 1, 2])
# #     # 定义每个点的索引
# #     # flag = np.array([1, 1, -1, -1, -1])
# #     fig, ax = plt.subplots()
# #
# #
# #     # 检查点的索引
# #     num = 0
# #     while True:
# #         print("num : ", num)
# #         print(w)
# #         print(b)
# #         ax.scatter(x[:num_positive], y[:num_positive], label="正实例点")
# #         ax.scatter(x[num_positive:], y[num_positive:], label="负实例点")
# #         plt.legend()
# #         x_linear = np.arange(-2, 8)
# #         if w[1] == 0:
# #             w[1] = random.randint(2,5)
# #         y_linear = (w[0] * x_linear + b) / w[1] * -1
# #         ax.plot(x_linear, y_linear)
# #         plt.xticks(x_linear)
# #         plt.pause(0.1)
# #         plt.cla()
# #         if perceptron(w, np.array([x[num], y[num]]), b) * flag[num] <= 0:
# #             res = gradient_descent(w, np.array([x[num], y[num]]), b, flag[num])
# #             w = res[0]
# #             b = res[1]
# #             num = 0
# #             continue
# #         num += 1
# #         if num == x.shape[0]:
# #             break
# #     ax.scatter(x[:num_positive], y[:num_positive], label="正实例点")
# #     ax.scatter(x[num_positive:], y[num_positive:], label="负实例点")
# #     y_linear = (w[0] * x_linear + b) / w[1] * -1
# #     ax.plot(x_linear, y_linear)
# #     plt.pause(10)
#
#
#
#
#
#
#


def add(x, y):
    return x + y


def sub(x, y):
    return x - y
