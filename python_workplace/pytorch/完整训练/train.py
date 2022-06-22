import torch
import torchvision
from torchvision import models
from torchvision.transforms import transforms
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
# from model import Net

# 定义训练的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 处理数据
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "vaild": transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

train_data = torchvision.datasets.CIFAR10('../CIFAR10', train=True, download=False,
                                          transform=data_transform["train"])
test_data = torchvision.datasets.CIFAR10('../CIFAR10', train=False, download=False,
                                         transform=data_transform["vaild"])


train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
vaild_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# print(f'训练集的长度为：{len(train_data)}')
# print(f'验证集集的长度为：{len(test_data)}')

net = models.resnet18(pretrained=True)
net.add_module("add_linear", nn.Linear(1000, 10))

# net = Net()  # 创建网络模型
net.to(device)
loss_fn = nn.CrossEntropyLoss()  # 损失函数
loss_fn.to(device)
learning_rate = 0.0001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)  # 优化器


epoch = 10

# 训练
train_loss_all, vaild_loss_all = [], []
train_accur_all, vaild_accur_all = [], []

for i in range(epoch):
    net.train()
    train_loss, vaild_loss = 0, 0
    train_accuracy, vaild_accuracy = 0.0, 0.0
    total_train_step, total_vaild_step = 0, 0
    train_num, vaild_num = 0, 0
    print(f'------第{i + 1}轮训练开始------')

    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        img, label = data
        img = img.to(device)
        label = label.to(device)
        output = net(img)
        loss = loss_fn(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        accuracy = (output.argmax(1) == label).sum()
        train_accuracy += accuracy
        train_num += img.size(0)

        total_train_step += 1
        # if total_train_step % 100 == 0:
        #     print(f'训练次数：{total_train_step}, loss: {loss.item()}')

    train_loss_all.append(train_loss / total_train_step)
    train_accur_all.append(train_accuracy.item() / train_num)
    print(f'epoch: {i + 1}, train_loss: {train_loss / total_train_step:f}, train_accyracy: {train_accuracy / train_num * 100:f}%')

    # 验证
    net.eval()
    with torch.no_grad():
        vaild_bar = tqdm(vaild_loader)
        for data in vaild_bar:
            img, label = data
            img = img.to(device)
            label = label.to(device)
            output = net(img)
            loss = loss_fn(output, label)
            vaild_loss += loss.item()
            accuracy = (output.argmax(1) == label).sum()
            vaild_accuracy += accuracy
            vaild_num += img.size(0)

            total_vaild_step += 1

    vaild_loss_all.append(vaild_loss / total_vaild_step)
    vaild_accur_all.append(vaild_accuracy.item() / vaild_num)
    print(f'epoch: {i + 1}, vaild_loss: {vaild_loss / total_vaild_step:f}, vaild_accuracy: {vaild_accuracy / vaild_num * 100:f}%')

# 画图
plt.figure(figsize=(12, 4), dpi=300)
plt.subplot(1, 2, 1)
plt.plot(range(1, epoch + 1), train_loss_all, 'ro-', label='train loss')
plt.plot(range(1, epoch + 1), vaild_loss_all, 'bs-', label='vaild loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim([1, epoch])

plt.subplot(1, 2, 2)
plt.plot(range(1, epoch + 1), train_accur_all, 'b', label='train accuracy')
plt.plot(range(1, epoch + 1), vaild_accur_all, 'r--', label='vaild accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()