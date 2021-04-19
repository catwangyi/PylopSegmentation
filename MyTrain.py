from MyDataSet import *
from MyUnet import UNet
from torch import optim
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

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

def train_net(net, device, data_apth="E:\Pytest\input\kvasirseg", epochs=20, batch_size = 1, lr = 0.0001):
    #加载训练集
    train_dataset = MyDataSet(data_apth)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #定义optimizer
    optimizer = optim.Adam([dict(params=net.parameters(), lr=lr)])
    #loss算法
    # criterion = nn.CrossEntropyLoss()
    criterion = smp.utils.losses.DiceLoss()
    #初始化为正无穷
    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()#进入训练模式
        current_num = 1
        for image, label in train_loader:
            optimizer.zero_grad()
            #将数据传入device
            image = image.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.float)

            out = net(image)
            #test
            # out_test = out.detach().squeeze().cpu().numpy()
            # label_test = label.detach().squeeze().cpu().numpy()
            #
            # label_test = np.transpose(label_test, (1, 2, 0))
            # label_test = colour_code_segmentation(reverse_one_hot(label_test), [[255, 255, 255], [0, 0, 0]])
            #
            # out_test = np.transpose(out_test, (1, 2, 0))
            # out_test = colour_code_segmentation(reverse_one_hot(out_test), [[255, 255, 255], [0, 0, 0]])
            # visualize(label_test=label_test,
            #           out_test=out_test)

            loss =criterion(out, label)
            if loss<best_loss:
                best_loss = loss
                torch.save(net, 'my_best_model.pth')
            print("epoch:", epoch+1,
                  "{}/{}".format(current_num, len(train_dataset)),
                  ' best_loss:%.5f' % best_loss.item(),
                  " loss:%.5f" % loss.item()
                  )
            current_num += 1
            #更新参数
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    if os.path.exists('./my_best_model.pth'):
        net = torch.load('./my_best_model.pth', map_location=device)
    else:
        net = UNet(n_channels=3, n_classes=2)
    net.to(device)
    # 将网络拷贝到deivce中
    # 指定训练集地址，开始训练
    train_net(net, device, data_apth="E:\Pytest\input\kvasirseg_f", epochs=20, batch_size=1, lr=0.0001)
