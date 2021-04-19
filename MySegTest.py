import torch
from torchvision import models
from torchvision import transforms
from torchsummary import summary
import torch.nn as nn
import torch.utils.data as Data
import cv2, copy
import segmentation_models_pytorch as smp
from utils import *
import torch.nn.functional as F
#使用预训练好的VGG16
vgg16 = models.vgg16(pretrained=True)
#获取vgg16的特征提取层
vgg = vgg16.features


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['background', 'polyp']
mask_value = [[0, 0, 0], [255, 255, 255]]
root= 'E:\Pytest\input\kvasirseg_f\\'
image_id = "image_id"
image_path = "image_path"
mask_path = "mask_path"
for para in vgg.parameters():
    para.requires_grad_(False)


def load_data(csv_path):
    image = pd.read_csv(root+csv_path)
    image = image[[image_id, image_path, mask_path]]
    image[image_path] = root + image[image_path]
    image[mask_path] = root +image[mask_path]

    valid_df = image.sample(frac=0.1, random_state=42)
    train_df = image.drop(valid_df.index)
    return train_df, valid_df

def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map

class Conv3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.conv(input)

class ConvTwice(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvTwice, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.feature1 = ConvTwice(in_channel, 64)

        self.feature2 = ConvTwice(64, 128)

        self.feature3 = Conv3(128, 256)

        self.feature4 = Conv3(256, 512)

        self.feature5 = Conv3(512, 1024)

        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.up4_conv = ConvTwice(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2 ,stride=2)

        self.up3_conv = ConvTwice(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.up2_conv = ConvTwice(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up1_conv = ConvTwice(128, 64)
        self.up1 = nn.Conv2d(64, out_channel, kernel_size=1)

    def forward(self, input):
        f1 = self.feature1(input)
        temp = self.pool(f1)

        f2 = self.feature2(temp)
        temp = self.pool(f2)

        f3 = self.feature3(temp)
        temp = self.pool(f3)

        f4 = self.feature4(temp)
        temp = self.pool(f4)

        f5 = self.feature5(temp)#1024通道

        temp = self.up5(f5)#512通道
        temp = torch.cat([temp, f4], dim=1)#1024
        temp = self.up4_conv(temp)

        temp = self.up4(temp)
        temp = torch.cat([temp, f3],dim=1)
        temp = self.up3_conv(temp)

        temp = self.up3(temp)
        temp = torch.cat([temp, f2], dim=1)
        temp = self.up2_conv(temp)

        temp = self.up2(temp)
        temp = torch.cat([temp, f1], dim=1)
        temp = self.up1_conv(temp)

        temp = self.up1(temp)
        temp = nn.Sigmoid()(temp)
        return temp

def train_model(model, lossfunc, optimizer, traindataloader, valdataloader, num_epochs = 20):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0
        model.train()
        for step, (x, y) in enumerate(traindataloader):
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.long().to(device)
            out = model(x)
            loss = lossfunc(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(y)
            train_num += len(y)
        train_loss_all.append(train_loss/train_num)
        print('{} Train Loss: {:.4f}'.format(epoch, train_loss_all[-1]))

        model.eval()
        for step, (x, y) in enumerate(valdataloader):
            x = x.float().to(device)
            y = y.long().to(device)
            out = model(x)


            loss = lossfunc(out, y)
            train_loss += loss.item()*len(y)
            train_num += len(y)
        val_loss_all.append(val_loss/val_num)
        print('{} Val Loss: {:.4f}'.format(epoch, val_loss_all[-1]))
        if val_loss_all[-1]<best_loss:
            best_loss = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all})
    model.load_state_dict(best_model_wts)
    return model, train_process


if __name__=='__main__':
    LR = 0.0001
    model = Unet(3, 3).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train, val = load_data('metadata.csv')
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50', 'imagenet')
    train_dataset = MyDataSet(train, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    val_dataset = MyDataSet(val, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = Data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = Data.DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=2)

    model, train_process = train_model(model, loss, optimizer, train_loader, val_loader)
    torch.save(model, "mybest.pth")
