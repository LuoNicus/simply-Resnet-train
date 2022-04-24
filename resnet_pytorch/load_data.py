import numpy as np
import os
import random
import yaml
import cv2
# import torch


def get_datas(data_path, drop_rate=0.1):  # 填写文件夹名
    datas = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }
    for k, i in enumerate(os.listdir(f".\\{data_path}")):
        # 大文件夹
        for k1, i1 in enumerate(os.listdir(f".\\{data_path}\\{i}\\images")):
            f_name = i1[:-4]
            if i1[-3:] in ["png", 'jpg']:
                try:
                    with open(f".\\{data_path}\\{i}\\labels\\{f_name}.yaml") as f:
                        label = yaml.load(f, Loader=yaml.SafeLoader)  # 读取yaml中存储的标签数据(0 -- 5,与字典值相对应)
                    datas[label].append(f".\\{data_path}\\{i}\\images\\{i1}")  # 按标签值来进行索引，并  append
                except:
                    print(f"缺失标签：{f_name}.yaml")  # 否者数据缺失
                    continue
    datas[0] = random.sample(datas[0], int(len(datas[0]) * drop_rate))  # 随机取样， 在标签为 0 的图片路径中，取出其数量的  1/10
    return datas


def load_data_from_hard_disk(path_list, labels):  #
    # path_list 一个装满了路径的大列表
    # labels 一个装满了标签的大列表
    arr = None
    for k, i in enumerate(path_list):
        img = cv2.imread(i)
        if arr is None:  # 判断是不是第一张图片
            arr = np.array(img)
            arr = arr.reshape(1, arr.shape[0], arr.shape[1], 3)
        else:
            temp = np.array(img)
            temp = temp.reshape(1, temp.shape[0], temp.shape[1], 3)
            arr = np.concatenate((arr, temp), axis=0)  # 按照第一个轴进行拼接 （  例：（1， 288， 384， 3） 在 1 上拼接     ）
            del temp
    labels = np.array(labels)
    return arr, labels


class data_iter:
    def __init__(self, data_dict, batch_size=128, shuffle=False):
        # data_dict: 代表图片路径的字典
        # batch_size: 代表打包的批量数
        self.data_dict = data_dict
        self.data_pool = []                   # 保存了所有的数据，包含：[标签, 路径]
        self.batch_size = batch_size
        for i in data_dict:
            for j in data_dict[i]:
                self.data_pool.append([i, j])
        random.shuffle(self.data_pool)        # 打乱数据池
        self.rest_pool = self.data_pool.copy()
        self.shuffle = shuffle
   
    def main_iter(self):
        self.rest_pool = self.data_pool.copy()
        if self.shuffle:  # 这样的话每次都会打乱数据来采样
            random.shuffle(self.data_pool)
        
        while True:
            if len(self.rest_pool) >= self.batch_size:
                batch_pool = random.sample(self.rest_pool, self.batch_size).copy()
            else:
                batch_pool = self.rest_pool.copy()
            
            for i in batch_pool:
                self.rest_pool.remove(i)
            
            x_list, y_list = [], []
            for y, x_path in batch_pool:
                x_list.append(x_path)
                y_list.append(y)
   
            batch_x, batch_y = load_data_from_hard_disk(x_list, y_list)
            batch_x = batch_x.swapaxes(1, 3)
            yield 2. * batch_x / 255 - 1, batch_y
            
            if len(self.rest_pool) == 0:
                break


if __name__ == '__main__':
    a = get_datas("mod_data", 0.2)
    a[2] = a[4].copy()
    a[3] = a[5].copy()
    del a[4], a[5]

    b = data_iter(a, 20)
    print(len(b.data_pool))
    
    data_loader = b.main_iter()
    for i in range(5):
        for k, (x, y) in enumerate(b.main_iter()):
            print(i, k, x.shape, y.shape)
    
    # for k, (x, y) in enumerate(data_loader):
    #     print(0, k, x.shape, y.shape)

