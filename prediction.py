from utils import *
import numpy as np
import torch
import os
import segmentation_models_pytorch as smp

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prediction on Test Data
# load best saved model checkpoint from the current run
if os.path.exists('./best_model.pth'):
    best_model = torch.load('./best_model.pth', map_location=DEVICE)
    print('成功加载模型...')

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
                                      class_rgb_values=select_class_rgb_values)

        image, mask = image_data[0]
        # unsqueeze添加维度
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        x_tensor = x_tensor.float()

        # Predict test image
        pred_mask = best_model(x_tensor)

        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Get prediction channel corresponding to foreground
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
        masklist.append(pred_mask)
        # plt.imshow(masklist[0])
        # plt.show()
    print('预测结束')
    return masklist


if __name__ =="__main__":
    data_dict = []
    for i in range(1, 4):
         data_dict.append("../input/kvasirseg_f/images/"+str(i)+'.jpg')
    data_dict.append('jn.jpg')
    pred(data_dict)