import cv2
import os
import glob
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from MyUnet import *
import albumentations as album
from torch import optim


def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        #np.equal实现把label每个像素的RGB值与某个class的RGB值进行比对，变成RGB bool值。
        equality = np.equal(label, colour)
        #np.all 把RGB bool值，变成一个bool值，即实现某个class 的label mask
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    #np.stack实现所有class的label mask的堆叠。最终depth size 为num_classes的数量。
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

class MyDataSet(Dataset):
    def __init__(self, datapath):
        super(MyDataSet, self).__init__()
        # 初始化函数，读取所有data_path下的图片
        self.data_path = datapath
        self.img_path = glob.glob(os.path.join(datapath, 'images/*.jpg'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, idx):
        #根据idx获得图片
        image_path = self.img_path[idx]
        #根据imgpath生产labelpath
        label_path = image_path.replace('images', 'masks')

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        label = one_hot_encode(label, [[255, 255, 255], [0, 0, 0]]).astype('float32')

        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('t', image)
        # cv2.waitKey(0)
        image = album.resize(image, 512, 512, cv2.INTER_CUBIC)
        label = album.resize(label, 512, 512, cv2.INTER_CUBIC)
        # image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        # label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_CUBIC)
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        # label = np.transpose(label, (1, 2, 0))
        # label = colour_code_segmentation(reverse_one_hot(label), [[255, 255, 255], [0, 0, 0]])
        # visualize(img=label)

        return image, label

    def __len__(self):
        return len(self.img_path)

def train(epoch):
    running_loss=0.0
    for batch_idx, data in enumerate(train_loader, 0):
        #0是表示从0开始
        image, label=data
        image, label=image.to(device), label.to(device)            #数据放进GPU里
        optimizer.zero_grad()                  #优化器参数清零

        #forword+backward+update
        image=image.type(torch.FloatTensor)        #转化数据类型,不转则会报错
        image=image.to(device)

        outputs=model(image)

        loss=criterion(outputs, label)        #进行loss计算

        lll=label.long().cpu().numpy()             #把label从GPU放进CPU

        loss.backward(retain_graph=True)                  #反向传播(求导)
        optimizer.step()            #优化器更新model权重
        running_loss+=loss.item()       #收集loss的值
        if batch_idx % 100 ==99:
            print('[epoch: %d,idex: %2d] loss:%.3f' % (epoch+1, batch_idx+1, running_loss/322))         ##训练集的数量,可根据数据集调整
            runing_loss = 0.0# 收集的loss值清零
        torch.save(model, f='./my_best_model.pth') #保存权重


if __name__ == "__main__":
    LR = 0.0001
    EPOCH = 20
    train_dataset = MyDataSet("E:\Pytest\input\kvasirseg")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    model = Unet(3, 2)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有GPU加速
    model.to(device)  # 网络放入GPU里加速
    # model.load_state_dict(torch.load('my_best_model.pth'))

    for epoch in range(EPOCH):  # 迭代次数
        train(epoch)

