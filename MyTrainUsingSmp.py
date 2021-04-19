# from PIL import Image
# import cv2, os
#
# file_dir = 'E:\Pytest\input\kvasirseg_my\masks\\'
#
# out_dir = 'E:\Pytest\input\kvasirseg_my\masks\\'
#
# img = os.listdir(file_dir)
#
# print(img)
# for i in img:
#     I = cv2.imread(file_dir+str(i))
#     I = cv2.cvtColor(I ,cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(out_dir+str(i), I)

from utils import *
DATA_DIR = '../input/kvasirseg'
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from torchsummary import summary


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


class EndoscopyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            df,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_paths = df['image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        # self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read images and masks
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float32')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        # one-hot-encode the mask


        # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']

        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1).astype('float32')

        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)

def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image ) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def get_training_augmentation():
    train_transform = [
        album.Resize(height=512, width=512, interpolation=cv2.INTER_CUBIC, always_apply=True),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.Resize(height=512, width=512, interpolation=cv2.INTER_CUBIC, always_apply=True),
    ]
    return album.Compose(test_transform)



# def get_preprocessing(preprocessing_fn=None):
# #     _transform = []
# #     if preprocessing_fn:
# #         _transform.append(album.Lambda(image=preprocessing_fn))
# #     _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
# #
# #     return album.Compose(_transform)


if __name__ =="__main__":
    TRAINING = False
    metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    metadata_df = metadata_df[['image_id', 'image_path', 'mask_path']]
    metadata_df['image_path'] = metadata_df['image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    # Shuffle DataFrame
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

    # Perform 90/10 split for train / val
    valid_df = metadata_df.sample(frac=0.1, random_state=42)
    train_df = metadata_df.drop(valid_df.index)

    # Get class RGB values
    select_class_rgb_values = [[255, 255, 255], [0, 0, 0]]
    select_classes = ['polyp', 'background']


    ENCODER = 'resnet50'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = select_classes
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

    # load best saved model checkpoint from the current run
    if os.path.exists('./best_model.pth'):
        model = torch.load('./best_model.pth', map_location=DEVICE)
        print('Loaded UNet model from this run.')
    else:
        # create segmentation model with pretrained encoder
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
        )
    # summary(model, (3, 512, 512))
    # with SummaryWriter(comment='U-Net' ) as w:
    #     w.add_graph(model, torch.rand(1, 3, 512, 512))
    # de = torch.device("cuda")
    # model.to(de)
    # summary(model, (3, 512, 512))

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # print(preprocessing_fn)

    train_dataset = EndoscopyDataset(
        train_df,
        augmentation=get_training_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )

    valid_dataset = EndoscopyDataset(
        valid_df,
        augmentation=get_validation_augmentation(),
        class_rgb_values=select_class_rgb_values,
    )
    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Set num of epochs
    EPOCHS = 20
    # define loss function
    loss = smp.utils.losses.DiceLoss()
    # define metrics
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    # define optimizer
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    if TRAINING:
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, './best_model.pth')
                print('Model saved!')
    else:
        model.eval()
        with torch.no_grad():
            # create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
            test_dataset = EndoscopyDataset(
                valid_df,
                augmentation=get_validation_augmentation(),
                class_rgb_values=select_class_rgb_values,
            )
            for idx in range(len(test_dataset)):
                image, gt_mask = test_dataset[idx]

                image_vis = image.astype('uint8')
                image_vis = np.transpose(image_vis, (1, 2, 0))

                image = np.expand_dims(image, axis=0)
                image = torch.from_numpy(image).to(device=DEVICE)
                # Predict test image

                pred_mask = model(image)
                pred_mask = pred_mask.detach().squeeze().cpu().numpy()

                # Convert pred_mask from `CHW` format to `HWC` format
                pred_mask = np.transpose(pred_mask, (1, 2, 0))
                pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)

                visualize(
                    original_image=image_vis,
                    predicted_mask=pred_mask)