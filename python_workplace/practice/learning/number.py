# import tensorflow as tf
# from tensorflow import keras
# print("Tensorflow : " + tf.__version__)
# print("Keras : " + keras.__version__)

from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 全连接

# # print(train_images.shape, test_images.shape)
# # print(train_images[0])
# # print(train_labels[0])
# # plt.imshow(train_images[0])
# # plt.show()
#
train_images = train_images.reshape((60000, 28*28)).astype('float')
test_images = test_images.reshape((10000, 28*28)).astype('float')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# print(train_labels[0])


network = models.Sequential()
network.add(layers.Dense(units=128, activation='relu', input_shape=(28*28, ),
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=32, activation='relu',
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=10, activation='softmax'))

# 编译步骤
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
network.fit(train_images, train_labels, epochs=30, batch_size=128, verbose=2)


# 来在测试集上测试一下模型的性能吧
# y_pre = network.predict(test_images[:5])
# print(y_pre)
# print(test_labels[:5])

test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)


# 卷积网络

# # 加载数据集
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
#
# # 搭建LeNet网络
# def LeNet():
#     network = models.Sequential()
#     network.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#     network.add(layers.AveragePooling2D((2, 2)))
#     network.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
#     network.add(layers.AveragePooling2D((2, 2)))
#     network.add(layers.Conv2D(filters=120, kernel_size=(3, 3), activation='relu'))
#     network.add(layers.Flatten())
#     network.add(layers.Dense(84, activation='relu'))
#     network.add(layers.Dense(10, activation='softmax'))
#     return network
#
#
# network = LeNet()
# network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#
# train_images = train_images.reshape((60000, 28, 28, 1)).astype('float') / 255
# test_images = test_images.reshape((10000, 28, 28, 1)).astype('float') / 255
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# # 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
# network.fit(train_images, train_labels, epochs=10, batch_size=128, verbose=2)
# test_loss, test_accuracy = network.evaluate(test_images, test_labels)
# print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
