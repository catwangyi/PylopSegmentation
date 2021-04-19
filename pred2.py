import torch
import pandas as pd
import cv2, os
from PIL import Image
import torch.utils.data as Data
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torchvision
from utils import *
import albumentations as album
from torchvision.transforms import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['background', 'polyp']
mask_value = [[0, 0, 0], [255, 255, 255]]
root= 'E:\Pytest\input\kvasirseg\\'
image_id = "image_id"
image_path = "image_path"
mask_path = "mask_path"


def load_data(csv_path):
    image = pd.read_csv(root+csv_path)
    image = image[[image_id, image_path, mask_path]]
    image[image_path] = root + image[image_path]
    image[mask_path] = root +image[mask_path]

    valid_df = image.sample(frac=0.1, random_state=42)
    train_df = image.drop(valid_df.index)
    return train_df, valid_df

class MyDataSet(Data.Dataset):
    def __init__(self, data, img_pth=None, mask_pth=None, augmentation=None, preprocessing=None):
        self.image_paths = data[img_pth].tolist()
        self.mask_paths = data[mask_pth].tolist()
        self.mask = mask_value
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[idx]), cv2.COLOR_BGR2RGB)

        mask = one_hot_encode(mask, mask_value).astype('float')

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
        return len(self.image_paths)
