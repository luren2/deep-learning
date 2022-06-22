# print("hello world")

# import random                 ##引入随机数
# res = random.randint(1,10)
#
# temp = input("猜一下我心里想的是哪个数字: ")
# guess = int(temp)
# while guess != res:
#     temp = input("猜错了，重新猜一下我心里想的是哪个数字: ")
#     guess = int(temp)
#     if guess > res:
#         print("大了")
#     if guess < res:
#         print("小了")
# print("恭喜你，猜对了！！！")
# print("猜中了又能怎么样！！！")
# print("游戏结束，不想玩了！！！")

# temp = input("请输入您的姓名：")
# print("您好,",temp)

# print('let\'s go!!!') ##转义字符
# print("let's go!!!") ##也可使用双引号
# print(1>3)


####  数据类型

# print(type(1))   ##使用type查看数据类型
# print(type('1'))
#
# print(isinstance(1,int))  ##使用isinstance验证数据类型
# print(isinstance(1.0,float))
# print(isinstance("123",str))

####  优先级从高到低:
# 幂运算: **
# 正负号: + -
# 算术操作符: * / // + -              ## /号是取精确的商 //则是向下取整 类似C、C++、java中的除法操作
# 比较操作符: < <= > >= == !=
# 逻辑运算符: not and or

#### 列表(数组)的定义与遍历
x = [0, 1, 2, 3, '你好']
# for i in x:
#     print(i)         ##遍历数组(换行)
# print(x[4])          ##打印出下标为4的数据
# print(x)               ##打印数组
# print(x[0:5])        ##打印下标从0到4的数据(不换行)
# print(x[:])          ##打印下标从0到4的数据(不换行) 与上面的方式一样
# print(x[0:5:2])      ##打印x[0] x[2] x[4]
# print(x[-1])         ##打印最后一个数据即x[4]
# print(x[::-1])       ##倒序输出
# print(x[:3])         ##打印下标为0、1、2的数据
# print(x[3:])         ##打印下标为3、4的数据


#### 数组的增删改查

## 增
x = [0, 1, 2, 3]
# x.append("hi")           ##在x数组里面增加元素 但是每次只能增加一个
# print(x)
# x.extend([3,5,6,"你好"]) ##在x末尾增加元素 可增加多个
# print(x)
# x[len(x):]=[7,8,9]       ##用切片的方法往x里面增加元素 可增加多个
# print(x)
# x.insert(4,3)            ##向x下标为4的位置插入数据3
# print(x)

## 删
# x.remove(0)              ##删除x数组中的0 如果有多个0只会删除第一个 如果不存在0 那么程序会报错
# del x[0]                 ## 删除x[0]
# print(x)
# x.pop()                  ## 删除最后一个
# x.pop(1)                 ##删除x数组中下标为1的数据 pop是根据下标删除
# print(x)
# x.clear()                  ##清空数组
# print(x)

## 改
# x[1]=6                     ##将x数组下标为1的数据改为6
# print(x)
# x[2:]=[1]                  ##采用切片的方式修改元素
# print(x)
# x.sort()                   ##排序 从小到大
# sorted(x)                  ## 不改变x原来值
# sorted(x, reverse=1)       ## 从大到小
# print(x)
# x.reverse()
# print(x)                   ##翻转数组
# x.sort(reverse=1)          ##从大到小排序
# print(x)
# x.reverse()                ## 翻转数组 改变愿列表顺序


## 查
# print(x.count(1))         ##查找1出现的次数
# print(x.index(2))         ##查找数据2的索引 如果找不到会报错 如果有多个2则只会输出第一个索引值(下标值)
# x[x.index(2)]=4           ##找到数组中2的索引 并把2替换成4
# print(x)
# print(x.index(2,1,3))   ##从下标1-3中找到2的索引值

# a = ' hello world   '
# b = a.rstrip()  # 只去掉后面的空格 不改变原来的值
# c = a.strip()  # 去掉前后的空格
# d = a.lstrip() # 只去掉前面的空格
# print(a, '\n', b, '\n', c, '\n', d)
# print(len(a), len(b), len(c), len(d))


# print('let\'s \"go"!')

## 最大、最小、总和
# num = [1, 2, 3, 4]
# print(min(num))
# print(max(num))
# print(sum(num))

## 元组(不可变, 可以重新整体赋值)
# x1 = (1, 2, 3)
# print(x1[1])
# x1 = (5, 6)
# print(x1)

# x2 = [n**3 for n in range(1, 11)]
# print(x2)

# x = [
#     {
#         'a': 'b',
#         'c': 'd',
#     },
#     {
#         'a': 'b',
#         'c': 'd',
#     },
# ]


# for a, b in x.items():
#     print(a, b)
# for a in x.keys():
#     print(a)
# for a in x.values():
#     print(a)

# for item in x:
#     for i in item.items():
#         print(i)
#
# m, n, k = map(int, input().split(' '))  # m*n*k维数组
# graph = [[[0 for j in range(k)] for i in range(n)] for a in range(m)]
# print(graph)

# print('hello', end=' ')
# print('world', end='')


# def printStr(*args): #*args元组 如果是字典则是**
#     a = 'hh ';
#
#     for i in args:
#         a += i + ' '
#     return a

# b=['1','2','3']
# print('.'.join(b)) # 用.隔开 1.2.3

# from Learnnumpy import add, sub  # 模块化导入多个
# print(add(3, 4))
# print(sub(2, 1))

# class Cat:
#     def __init__(self, name):
#         self.name = name
#
#     def getName(self):
#         print(self.name)
#
#
# car = Cat('hh')
# car.getName()


# from random import randint
# x = randint(1, 6)  # 闭区间 都能取到
# print(x)

# from random import random
# x = random()  # 没有参数 取0-1的浮点数
# print(x)


## 读取文件
# with open("C:/Users/pc/Desktop/新建文本文档.txt", encoding='utf-8') as f:
#     print(f.read().strip())
# for line in f:
#     # print(line, end='')
#     print(line.rstrip())

## 按行读取
# res=[]
# with open("C:/Users/pc/Desktop/新建文本文档.txt", encoding='utf-8') as f:
#     lines = f.readlines()
# for line in lines:
#     res.append(line.strip())
# # print(lines)
# print(res)

## 追加a(只能写） 或a+(可读可写）
# with open("C:/Users/pc/Desktop/新建文本文档.txt", 'a+', encoding='utf-8') as f:
#     f.write("\n1111")
#     f.flush()
#     lines = f.read()
#     print(lines)

# # 读取txt文件并删除文中空行保存到新文件里
# file1 = open("C:/Users/pc/Desktop/新建文本文档.txt", 'r', encoding='utf-8')
# file2 = open("C:/Users/pc/Desktop/新建文本文档temp.dox", 'w', encoding='utf-8')
#
# for line in file1.readlines():
#     if line == '\n':
#         line = line.strip('\n')
#     file2.write(line)


# a ='1 2 3 4'
# print(a.split())  # 默认分割空格和换行 有两个参数一个是规则 一个是分割的长度

## 画图
import matplotlib.pyplot as plt

plt.rc("font", family="Microsoft YaHei")
# data = [1, 4, 9, 16, 25]

# plt.plot(data)
# plt.show()
# font = {'family': 'serif',
#     'color': 'darkred',
#     'weight': 'normal',
#     'size': 16,
#     }

# x = [1, 2, 3, 4, 5]
# y = [1, 4, 8, 16, 25]
# plt.plot(x, y, linewidth=1)  # 线的宽度
# plt.title('数据', fontsize=18)
# plt.xlabel('x轴数据', fontsize=14, labelpad=5)
# plt.ylabel('y轴数据', fontsize=14)
# plt.tick_params(axis='both', labelsize=15)
# plt.show()

## 保存
# plt.savefig('C:/Users/pc/Desktop/x.png', bbox_inches='tight',
#             pad_inches=1,
#             transparent=True,
#             facecolor='w',
#             orientation='landscape')

# x = ['a', 'b', 'c', 'd']
# for i, s in enumerate(x ):
#     print(i, s)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(3,2),facecolor="white")  #创建画布
# plt.plot()                                   #绘制图形
# plt.show()

import numpy as np

# x = 7 / 9 * np.log2(9 / 7) + 2 / 9 * np.log2(9 / 2)
# print(x)
# y1 = 3 / 4 * np.log2(4 / 3) + 1 / 4 * np.log2(4 / 1)
# y2 = 3 / 4 * np.log2(4 / 3) + 1 / 4 * np.log2(4 / 1)
# y3 = 1 / 1 * np.log2(1) + 1 / 1 * np.log2(1 / 1)
# y = 4 / 9 * y1 + 4 / 9 * y2 + 1 / 9 * y3
# print(np.round(x-y, 3))

rmse = np.sqrt(np.log2(131000)-np.log2(3767.5732))
print(rmse)

import torch
print(torch.__version__)

