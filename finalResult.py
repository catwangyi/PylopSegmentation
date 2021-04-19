import sys
from PyQt5.QtGui import QPixmap
import pandas as pd
import numpy as np
import cv2,os
import albumentations as album
import torch
import torch.jit
#打包时要用到
def script_method(fn, _rcb=None):
    return fn
def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj
torch.jit.script_method = script_method
torch.jit.script = script
import segmentation_models_pytorch as smp
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(804, 537)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/wang/Desktop/jn.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.selectPicPushButton = QtWidgets.QPushButton(Dialog)
        self.selectPicPushButton.setGeometry(QtCore.QRect(170, 460, 211, 61))
        self.selectPicPushButton.setAutoDefault(False)
        self.selectPicPushButton.setObjectName("selectPicPushButton")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(160, 10, 72, 15))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(590, 10, 72, 15))
        self.label_2.setObjectName("label_2")
        self.scrollBar = QtWidgets.QScrollBar(Dialog)
        self.scrollBar.setGeometry(QtCore.QRect(20, 419, 761, 31))
        self.scrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.scrollBar.setObjectName("scrollBar")
        self.originPic = QtWidgets.QLabel(Dialog)
        self.originPic.setGeometry(QtCore.QRect(40, 40, 311, 321))
        self.originPic.setObjectName("originPic")
        self.predPic = QtWidgets.QLabel(Dialog)
        self.predPic.setGeometry(QtCore.QRect(450, 40, 311, 321))
        self.predPic.setObjectName("predPic")
        self.predPicPushButton = QtWidgets.QPushButton(Dialog)
        self.predPicPushButton.setGeometry(QtCore.QRect(440, 460, 211, 61))
        self.predPicPushButton.setAutoDefault(False)
        self.predPicPushButton.setObjectName("predPicPushButton")
        self.label_num = QtWidgets.QLabel(Dialog)
        self.label_num.setGeometry(QtCore.QRect(390, 380, 72, 15))
        self.label_num.setObjectName("label_num")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "息肉检测系统"))
        self.selectPicPushButton.setText(_translate("Dialog", "选择图片"))
        self.label.setText(_translate("Dialog", "原始图像"))
        self.label_2.setText(_translate("Dialog", "预测图像"))
        self.originPic.setText(_translate("Dialog", "originPic"))
        self.predPic.setText(_translate("Dialog", "predPic"))
        self.predPicPushButton.setText(_translate("Dialog", "开始检测"))
        self.label_num.setText(_translate("Dialog", "num"))

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = None
pred_file_path = None
masklist = None
polyp_num = 0
#原图中有息肉的图片下标
origin_num_list = []

if os.path.exists('./best_model.pth'):
    best_model = torch.load('./best_model.pth', map_location=DEVICE)
    # print('成功加载模型...')

def pred(data_dict):
    image_id = "frame_id"
    image_name = "image_path"
    mask_name = "mask_path"
    masklist = []
    #每检测一张图片循环一次
    for data_dir in data_dict:
        dict = {image_id: [0], image_name: data_dir, mask_name: data_dir}
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

        select_class_rgb_values = np.array([[1, 1, 1], [0, 0, 0]])
        # create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
        image_df = pd.DataFrame(dict)
        image_data = EndoscopyDataset(image_df,
                                      augmentation=get_validation_augmentation(),
                                      preprocessing=get_preprocessing(preprocessing_fn),
                                      class_rgb_values=select_class_rgb_values, img_pth=image_name, mask_pth=mask_name)

        image, mask = image_data[0]
        # unsqueeze添加维度
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        x_tensor = x_tensor.float()

        # Predict test image
        best_model.eval()
        with torch.no_grad():
            pred_mask = best_model(x_tensor)
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Get prediction channel corresponding to foreground
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)

        masklist.append(pred_mask)
        # plt.imshow(masklist[0])
        # plt.show()
    # print('预测结束')
    return masklist

def selectPicPushButton_click(window):
    ui.originPic.clear()
    ui.predPic.clear()
    path = QFileDialog.getOpenFileNames(window, "选择文件")

    if path:
        global file_path
        file_path = path
        # print(file_path)
        # print('选取文件数： ', len(file_path[0]))
        ui.scrollBar.setMaximum(len(file_path[0])-1)
        ui.scrollBar.setMinimum(0)
        pix = QPixmap(file_path[0][0])
        ui.scrollBar.setValue(0)
        ui.originPic.setPixmap(pix)
        ui.label_num.setText('{}/{}'.format(ui.scrollBar.value()+1, ui.scrollBar.maximum()+1))


def scrollBarChanged():
    # ui.originPic.repaint()
    #如果打开了文件
    global masklist
    if file_path:
        # print('scrollbarvalue:', ui.scrollBar.value())
        pix = QPixmap(file_path[0][ui.scrollBar.value()])
        ui.originPic.setPixmap(pix)

        if masklist:
            pix = QPixmap(file_path[0][origin_num_list[ui.scrollBar.value()]])
            ui.originPic.setPixmap(pix)
            pix = QPixmap('./cache/' + str(ui.scrollBar.value()) + '.jpg')
            ui.predPic.setPixmap(pix)
        ui.label_num.setText('{}/{}'.format(ui.scrollBar.value() + 1, ui.scrollBar.maximum() + 1))


def startPrediction():
    ui.scrollBar.setValue(0)
    # print("开始检测...")
    global masklist
    masklist = pred(file_path[0])
    num = 0
    global polyp_num
    for i in range(0, len(masklist)):
        # print(type(masklist[i]))
        if np.all(masklist[i]==0):
            # print(masklist[i])
            # print('全为0')
            pass
        else:
            # print('有息肉')
            #将检测出息肉的图片下标记录下
            origin_num_list.append(i)
            path = './cache/' + str(num) + '.jpg'
            num = num + 1
            # print("图片路径", file_path[0])
            originpic = cv2.imread(file_path[0][i])
            # print('读取图片成功')
            originpic = cv2.resize(originpic, (608, 512))
            img = np.array(masklist[i], dtype='uint8')
            img = cv2.multiply(img, originpic)
            cv2.imwrite(path, img)
            # print('保存相乘后的图片')
    polyp_num = num
    if polyp_num != 0:
        ui.scrollBar.setMaximum(num-1)
        pix = QPixmap('./cache/' + str(0) + '.jpg')
        ui.predPic.setPixmap(pix)

    else:#没有检测到息肉
        ui.scrollBar.setMaximum(0)
        ui.originPic.clear()
        ui.predPic.clear()
    ui.label_num.setText('{}/{}'.format(ui.scrollBar.value() + 1, ui.scrollBar.maximum() + 1))
    # print('finish')

#加**时，返回为字典，输入指定pred_mask = predmask，则该函数中images为字典。
def getImg(**images):
    imglist = []
    for idx, (name, image) in enumerate(images.items()):
        imglist.append(image)
    return imglist


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
def get_training_augmentation():
    train_transform = [
        album.Resize(height=512, width=608, interpolation=cv2.INTER_CUBIC, always_apply=True),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.Resize(height=512, width=608, interpolation=cv2.INTER_CUBIC, always_apply=True),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)
class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
            mask_pth=None,
            img_pth=None
    ):
        self.image_paths = df[img_pth].tolist()
        self.mask_paths = df[mask_pth].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)


        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)


if __name__ == "__main__":
    if os.path.exists('./cache'):
        pass
    else:
        os.makedirs('./cache')
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.originPic.setScaledContents(True)
    ui.predPic.setScaledContents(True)

    ui.selectPicPushButton.clicked.connect(lambda: selectPicPushButton_click(window=MainWindow))
    ui.scrollBar.valueChanged.connect(lambda: scrollBarChanged())
    ui.predPicPushButton.clicked.connect(lambda: startPrediction())
    sys.exit(app.exec_())
