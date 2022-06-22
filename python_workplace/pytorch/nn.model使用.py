import torch
import torchvision
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10('./CIFAR10', train=True, download=False,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('./CIFAR10', train=False, download=False,
                                         transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# img, target = test_data[0]
# print(img.shape)
# print(test_data.classes[target])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding='same'),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding='same'),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding='same'),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = Net()
print(net)
input = torch.ones((64, 3, 32, 32))
output = net(input)
print(output.shape)

writer = SummaryWriter('logs')
writer.add_graph(net, input)
writer.close()


# for data in train_loader:
#     imgs, target = data
#     output = net(imgs)
#     print(output.shape)

