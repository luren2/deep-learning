import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存模型
torch.save(vgg16, 'vgg16_method1.pth')
model = torch.load('vgg16_method1.pth')
# print(model)

# 保存参数

torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

vgg16_temp = torchvision.models.vgg16(pretrained=False)
vgg16_temp.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16_temp)
