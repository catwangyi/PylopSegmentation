import matplotlib.pyplot as plt
import torch, os, cv2
import numpy as np
from MyDataSet import MyDataSet
from utils import colour_code_segmentation, reverse_one_hot


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


if __name__ =="__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists('./my_best_model.pth'):
        # create test dataloader to be used with UNet model (with preprocessing operation: to_tensor(...))
        val_dataset = MyDataSet("E:\Pytest\input\kvasirseg")
        model = torch.load('./my_best_model.pth', map_location=DEVICE)
        cur_num = 1
        model.eval()
        with torch.no_grad():
            for data in val_dataset:
                image, label = data

                input_tensor = np.expand_dims(image, axis=0)
                input_tensor = torch.from_numpy(input_tensor)

                image_vis = np.transpose(image, (1, 2, 0))

                input_tensor = input_tensor.to(device=DEVICE, dtype=torch.float32)
                # Predict test image
                pred_mask = model(input_tensor)
                pred_mask = pred_mask.detach().squeeze().cpu().numpy()

                # Convert pred_mask from `CHW` format to `HWC` format
                pred_mask = np.transpose(pred_mask, (1, 2, 0))

                pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), [[255, 255, 255], [0, 0, 0]])

                visualize(
                    original_image=image_vis,
                    predicted_mask=pred_mask)
                if cur_num == 10:
                    break
                cur_num += 1

    else:
        print("no model!")