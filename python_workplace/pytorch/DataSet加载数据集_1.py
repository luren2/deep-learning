import json

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from tqdm import tqdm

# 定义训练的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_data(root):
    assert os.path.exists(root), f"dataset root: {root} does not exist"

    labels = [label for label in os.listdir(root) if os.path.isdir((os.path.join(root, label)))]
    labels.sort()
    labels_dict = dict((k, v) for v, k in enumerate(labels))
    json_str = json.dumps(dict((val, key) for key, val in labels_dict.items()), indent=4)
    if root.split('/')[-1] == 'train':
        with open('./dataset/labels.json', 'w') as json_file:
            json_file.write(json_str)

    images_path, images_label = [], []
    for label in labels:
        label_path = os.path.join(root, label)
        single_images = [os.path.join(root, label, i) for i in os.listdir(label_path)]
        single_images_label = labels_dict[label]
        for img_path in single_images:
            images_path.append(img_path)
            images_label.append(single_images_label)
    return images_path, images_label


class MyData(Dataset):

    def __init__(self, root, transform):
        super(MyData, self).__init__()
        self.images_path, self.images_label = read_data(root)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        label = self.images_label[idx]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images_path)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(8192, 1024),
            nn.Linear(1024, 10),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    train_dir = './dataset/train'
    vaild_dir = './dataset/val'

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),  # 将图像随意裁剪，宽高均为224
        transforms.RandomHorizontalFlip(),  # 以 0.5 的概率左右翻转图像
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
        # transforms.RandomRotation(degrees=5, expand=False, fill=None),
        transforms.ToTensor(),  # 将 PIL 图像转为 Tensor，并且进行归一化
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化
    ])
    val_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(64),
        transforms.ToTensor(),  # 将 PIL 图像转为 Tensor，并且进行归一化
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化
    ])
    train_data = MyData(train_dir, transform=train_transform)
    val_data = MyData(vaild_dir, transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    vaild_loader = DataLoader(val_data, batch_size=8, shuffle=True)

    # img1, label1 = train_dataset[0]
    # img2, label2 = val_dataset[0]
    # image = transforms.ToPILImage()(img1)  # tensor转化为PIL
    # # x = img1.permute(1, 2, 0).numpy()    # tensor转化为numpy
    # # x = Image.fromarray(x)
    # # plt.imshow(x)
    # # plt.show()
    # image.show()


def train():
    net = Net()  # 创建网络模型
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()  # 损失函数
    loss_fn.to(device)
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # 优化器

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
        print(
            f'epoch: {i + 1}, train_loss: {train_loss / total_train_step:f}, train_accyracy: {train_accuracy / train_num * 100:f}%')

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
        print(
            f'epoch: {i + 1}, vaild_loss: {vaild_loss / total_vaild_step:f}, vaild_accuracy: {vaild_accuracy / vaild_num * 100:f}%')

    # 画图
    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), train_loss_all, 'ro-', label='train loss')
    plt.plot(range(1, epoch + 1), vaild_loss_all, 'bs-', label='vaild loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), train_accur_all, 'b', label='train accuracy')
    plt.plot(range(1, epoch + 1), vaild_accur_all, 'r--', label='vaild accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

    # 保存模型
    torch.save(net, 'net.pth')


def im_convert(tensor):
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    # image = image.clip(0, 1)
    return image


def predict():
    with open('./dataset/labels.json') as f:
        cat_to_name = json.load(f)
    net = torch.load('net.pth')
    print(cat_to_name['1'])
    net.eval()
    with torch.no_grad():
        img, label = next(iter(vaild_loader))
        img = img.to(device)
        label = label.to(device)
        output = net(img)
    print(output)
    x = torch.argmax(output, dim=1)
    print(x)
    true_label = [cat_to_name[str(int(i))] for i in label]
    pred_label = [cat_to_name[str(int(i))] for i in x]

    fig = plt.figure(figsize=(20, 12))
    for idx in range(2 * 4):
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
        ax.set_title(f'true: {true_label[idx]} pred: {pred_label[idx]}',
                     color=('green' if true_label[idx] == pred_label[idx] else 'red'))
        plt.imshow(im_convert(img[idx]))
    plt.show()


# train()
predict()
