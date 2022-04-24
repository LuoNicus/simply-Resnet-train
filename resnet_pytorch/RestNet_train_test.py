import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import random
import os
from PIL import Image
import numpy as np
import cv2
import time
import yaml
import torch.utils.data as Data
from load_data_test import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"             #  防止出错





# ************************ 下面代码用于处理文件夹里的图片与标签 *********

def load_data_doc(file_name, file_num_one, file_num_two):
    x_out, y_out = [], []  # 设两个空列表，用来存储  [图片矩阵信息] [标签信息]
    for i in range(file_num_one, file_num_two):
        # docs_list = os.listdir(f"{file_name}\\" + f"{i}")
        # print(docs_list)
        for k, j in enumerate(os.listdir(f"{file_name}\\" + f"{i}\\images")):  # 遍历文件夹 images 中的每个文件名，i表示每个文件名
            # print(f"{file_name}\\" + f"{i}\\images")                                                                    # print(i)   调试使用
            f_name = j[:-4]  # 取文件名前面的数字  将  .png 去掉
            # print(f"{file_name}\\{i}\\labels\\{f_name}.yaml")
            if j[-3:] in ["jpg", "png"]:  # 判断文件名后三位  是不是 jpg or png
                try:
                    with open(f"{file_name}\\{i}\\labels\\{f_name}.yaml") as f:  # 构建一个代码块，用来批处理
                        # print(f"{file_name}\\{i}\\labels\\{f_name}.yaml")
                        yaml_load = yaml.load(f, Loader=yaml.SafeLoader)  # 下载yaml数据
                        y_out.append(yaml_load)  # 将下载好的数据 append 到 列表 y_out
                except:
                    print(f"缺失标签：{f_name}.yaml")  # 否者数据缺失
                    continue

            img = cv2.imread(f"{file_name}\\{i}\\images\\{j}")  # 图片读取
            x_out.append(img)  # 将读取的图片数据 append 到 x_out

            # x = np.array(x_out)                                         #  转换成 np.array 类型，用于后续的  Q_learning
            # x = torch.from_numpy()                                     # 将numpy数组转换成tensor

    return x_out, y_out


def data_packing(x, y, batch_size=5, shuffle=False):                      #  该函数用于批量打包
    # x = np.array(x)
    x = x.swapaxes(1, 3)                                            #  这里进行换轴操作
    print(x.shape)
    # y = np.array(y)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y/5)
    if (use_gpu):
        x, y = x.cuda(), y.cuda()
    y = y.long()      # 转长整型
    x = 2 * x / 255 - 1  # 图像归一化
    # 构建分批数据集
    dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(       #使训练分批
        dataset=dataset,            # torch TensorDataset format
        batch_size=batch_size,             # mini batch size
        shuffle=shuffle,               # 训练时随机打乱数据再抽样
        num_workers=0,              # subprocesses for loading data
    )

    return loader










#   *********************该代码部分用于进行测试******************************

def load_MINST(labelsPath, imagesPath):
    import numpy as np
    import struct
    # labels：'train-labels.idx1-ubyte'
    # image：'train-images.idx3-ubyte'

    with open(labelsPath, 'rb') as lb_file:
        magic, n = struct.unpack('>II', lb_file.read(8))
        # 第一参数：二进制文件的格式（大端存储，整形，整形）；第二参数：要解码的对象（文件的前8个字节）
        labels = np.fromfile(lb_file, dtype=np.uint8)

    with open(imagesPath, 'rb') as img_file:
        magic, num, rows, cols = struct.unpack('>IIII', img_file.read(16))
        images = np.fromfile(img_file, dtype=np.uint8).reshape(num, 784)
    return images, labels



#   *********************该代码部分用于进行测试******************************










class Residual(nn.Module):            #  实现Residual block结构
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# '''下面实现ResNet34'''
#
#
# class ResidualBlock(nn.Module):      #  实现残差块方法2
#     # 实现子module: Residual Block
#     def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.right = shortcut
#
#     def forward(self, x):
#         out = self.left(x)
#         residual = x if self.right is None else self.right(x)
#         out += residual
#         return F.relu(out)




# class ResNet34(nn.Module):
#     # 实现主module:ResNet34
#     # ResNet34包含多个layer，每个layer又包含多个residual block
#     # 用子module实现residual block，用_make_layer函数实现layer
#
#     def __init__(self, num_classes=4):      #  类别有  4
#         super(ResNet34, self).__init__()
#         # 前几层图像转换
#         self.pre = nn.Sequential(
#             nn.Conv2d(3, 64, 7, 2, 3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3, 2, 1)
#         )
#         # 重复的layer，分别有3,4,6,3个residual block
#         self.layer1 = self._make_layer(64, 128, 3)
#         self.layer2 = self._make_layer(128, 256, 4, stride=2)
#         self.layer3 = self._make_layer(256, 512, 6, stride=2)
#         self.layer4 = self._make_layer(512, 512, 3, stride=2)
#
#         # 分类用的全连接
#         self.fc = nn.Linear(512, num_classes)
#
#     def _make_layer(self, inchannel, outchannel, block_num, stride=1):
#         # 构造layer，包含多个residual block
#         shortcut = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#
#         layers = []
#         layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
#
#         for i in range(1, block_num):
#             layers.append(ResidualBlock(outchannel, outchannel))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.pre(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = F.avg_pool2d(x, 7)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)






b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,     #  定义一个残差模块类
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))


net_resnet18 = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),    #  定义好的模型
                    nn.Flatten(), nn.Linear(512, 256), nn.Linear(256, 4))                    #  输出类别数4







def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            # BERT微调所需的（之后将介绍）
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]








def check_correct(net, loader_val):  # 计算正确率
    total_correct = 0
    total_num = 0
    net.eval()  # 【好习惯】将模型设置为测试模式

    for k, (bx, by) in enumerate(loader_val()):
        bx = torch.from_numpy(bx).type(torch.FloatTensor)
        by = torch.tensor(by).long()
        p = torch.argmax(F.softmax(net(bx), dim=0), dim=1)
        correct_num = torch.sum(p == by)
        total_correct += correct_num
        total_num += by.shape[0]
    # print(f"total: {float(total_correct / total_num)} {float(total_correct / total_num)})")

    return total_correct / total_num


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train(net, train_iter1, test_iter1, num_epochs, lr):     #  训练网络
    loss = F.cross_entropy
    if (use_gpu):
        print("train on GPU")
        net = net.cuda()
    net.apply(init_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_temp = 0
    batch_num = 0
    for k, (bx, by) in enumerate(train_iter1()):
        bx = torch.from_numpy(bx).type(torch.float)
        by = torch.tensor(by).long()
        p = net(bx)
        loss_1 = loss(p, by)
        loss_temp += float(loss_1)
        batch_num = k + 1


    losses = []
    losses.append(float(loss_temp / batch_num))  # 记录初始损失函数
    print(f"初始loss计算完成：{float(loss_temp / batch_num)}")
    correct_rates_test = []
    correct_rates_train = []
    temp1 = float(check_correct(net, train_iter1))
    temp2 = float(check_correct(net, test_iter1))
    print(f"epo:{0}\n loss:{loss_1}\tacc_train:{temp1}\tacc_val:{temp2}")
    correct_rates_train.append(temp1)  # 记录初始正确率
    correct_rates_test.append(temp2)  # 记录初始正确率

    max_acc = temp2
    for i in range(num_epochs):
        net.train()                                   # 【好习惯】将模型设置为训练模式
        loss_temp = 0
        batch_num = 0
        for k, (bx, by) in enumerate(train_iter1()):

            bx = torch.from_numpy(bx).type(torch.FloatTensor)
            by = torch.tensor(by).long()

            optimizer.zero_grad()                             # 清空梯度累计【重要】
            p = net(bx)  # 向前计算
            loss_ = loss(p, by)  # 损失函数计算 l(output, target)
            loss_.backward()  # 反向传播
            optimizer.step()  # 更新梯度
            loss_temp += float(loss_)
            batch_num = k + 1

        losses.append(float(loss_temp / batch_num))  # 每循环记录损失函数

        acc1 = float(check_correct(net, train_iter1))
        acc2 = float(check_correct(net, test_iter1))
        correct_rates_test.append(acc1)  # 每循环记录正确率
        correct_rates_train.append(acc2)  # 每循环记录正确率
        print(f"epo:{i}\n loss:{loss_} \t acc_train:{acc1}\tacc_val:{acc2}")

        if acc2 >= max_acc:
            import time
            now = time.localtime(time.time())
            torch.save(net.state_dict(), f".\\torch_weights\\best{now.tm_mon}月{now.tm_mday}日{now.tm_hour}时.pt")
            print("已保存模型权重")
            max_acc = acc2


    with open('./torch_train_info.yaml', 'w') as f:
        yaml.dump({"loss": losses, "loss_train": correct_rates_train, "loss_val": correct_rates_test}, f)


    import  matplotlib.pyplot as plt
    import time

    now = time.localtime(time.time())
    plt.figure()
    plt.plot(losses)
    plt.savefig(f".\\plots\\loss{now.tm_mon}月{now.tm_mday}日{now.tm_hour}时.svg")
    plt.show()

    plt.figure()
    plt.plot(correct_rates_train)
    plt.plot(correct_rates_test)
    plt.savefig(f".\\plots\\acc{now.tm_mon}月{now.tm_mday}日{now.tm_hour}时.svg")
    plt.show()



if __name__ == '__main__':
    import load_data

    # use_gpu = torch.cuda.is_available()
    use_gpu = 0
    lr, num_epochs, batch_size = 0.0001, 100, 20    # 训练参数


    main_dict = load_data.get_datas("train_data", 0.5)
    main_dict[2] = main_dict[4].copy()
    main_dict[3] = main_dict[5].copy()
    del main_dict[4], main_dict[5]


    val_dict = load_data.get_datas("val_data", 1.0)
    val_dict[2] = val_dict[4].copy()
    val_dict[3] = val_dict[5].copy()
    del val_dict[4], val_dict[5]

    train_pack = load_data.data_iter(main_dict, batch_size, shuffle=True)
    val_pack = load_data.data_iter(val_dict, batch_size)
    print(f"num_train:{len(train_pack.data_pool)}\tnum_val:{len(val_pack.data_pool)}")



#    datas_ = get_datas("train_data")
#    index_0, index_1, index_4, index_5 = shuffle_datas(datas_)
#   train_features, train_labels = connect_data(index_1, index_5, 1, 60, 1, 5)  # 生成训练数据
#    test_features, test_labels = connect_data(index_1, index_5, 3, 20, 1, 5)  # 生成测试数据

    # train_data = data_packing(train_features, train_labels, batch_size=5, shuffle=True)
    # test_data = data_packing(test_features, test_labels, batch_size=5, shuffle=False)
    # train(net, train_data, test_data, num_epochs, lr)  # d2l.try_gpu()  try_gpu  torch.device('cpu')
    # train_data = data_packing(train_features, train_labels, batch_size=5, shuffle=True)

    # test_data = data_packing(test_features, test_labels, batch_size=5, shuffle=False)
    train(net_resnet18, train_pack.main_iter, val_pack.main_iter, num_epochs, lr)  # d2l.try_gpu()  try_gpu  torch.device('cpu')
