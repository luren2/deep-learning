from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

'''
Dataset：
能把数据进行编号
提供一种方式，获取数据，及其label，实现两个功能：
1、如何获取每一个数据，及其label
2、告诉我们总共有多少个数据

数据集的组织形式，有两种方式：
1、文件夹的名字，就是数据的label
2、文件名和label，分别处在两个文件夹中，label可以用txt的格式进行存储

在jupyter中，可以查看，help，两个方式：
1、help（Dataset）
2、Dataset？？

Dataloader：
为网络提供不同的数据形式，比如将0、1、2、3进行打包

这一节内容很重要
'''


# writer = SummaryWriter("logs")

class MyData(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform):
        #  初始化，为这个函数用来设置在类中的全局变量
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为 label 和 Image里面文件名字相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.image_path, img_name)
        label_item_path = os.path.join(self.label_path, label_name)
        img = Image.open(img_item_path)

        with open(label_item_path, 'r') as f:
            label = f.readline()

        # img = np.array(img)
        img = self.transform(img)
        sample = {'img': img, 'label': label}
        return sample

    def __len__(self):
        # assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    root_dir = "dataset/train"
    image_ants = "ants_image"
    label_ants = "ants_label"
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"
    label_bees = "bees_label"
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    train_dataset = ants_dataset + bees_dataset

    # transforms = transforms.Compose([transforms.Resize(256, 256)])
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    # writer.add_image('error', train_dataset[119]['img'])
    # writer.close()
    # for i, j in enumerate(dataloader):
    #     # imgs, labels = j
    #     print(type(j))
    #     print(i, j['img'].shape)
    #     # writer.add_image("train_data_b2", make_grid(j['img']), i)

    # writer.close()


