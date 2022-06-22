import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print('ok')

train_data = torchvision.datasets.CIFAR10('./CIFAR10', train=True, download=False,
                                          transform=torchvision.transforms.ToTensor())

vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 添加层
print(vgg16_true)
vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 修改层
print(vgg16_false)
vgg16_false.classifier[6] = nn.Sequential()  # 删除层
vgg16_false.features = \
    nn.Sequential(*list(vgg16_false.features.children())[:-4])  # 批量删除后四层

# 冻结指定层的预训练参数：
vgg16_false.feature[26].weight.requires_grad = False
