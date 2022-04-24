import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import load_data
import torch_agent_models



def check_correct(net, loader_val, device="cuda:0"):  # 计算正确率
    total_correct = 0
    total_num = 0
    net.eval()  # 【好习惯】将模型设置为测试模式

    for k, (bx, by) in enumerate(loader_val()):
        bx = torch.from_numpy(bx).type(torch.FloatTensor).to(device)
        by = torch.tensor(by).long().to(device)
        p, _ = net(bx)
        p = torch.argmax(F.softmax(p, dim=0), dim=1)
        correct_num = torch.sum(p == by)
        total_correct += correct_num
        total_num += by.shape[0]
    return total_correct / total_num

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train(net, train_iter, test_iter, num_epochs, lr, 
			device="cuda:0", weights_path=None):

    net = net.to(device)
    net.apply(init_weights)
    l = F.cross_entropy
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss_temp = 0
    batch_num = 0
    for k, (bx, by) in enumerate(train_iter()):
        bx = torch.from_numpy(bx).type(torch.float).to(device)
        by = torch.tensor(by).long().to(device)
        p, _ = net(bx)
        loss_temp += float(l(p, by))
        batch_num = k + 1
    losses = []
    losses.append(float(loss_temp / batch_num))  # 记录初始损失函数

    correct_rates_train = []
    correct_rates_val = []
    acc_train = float(check_correct(net, train_iter))
    acc_val = float(check_correct(net, test_iter))
    print(f"epo:{0}\n loss:{float(loss_temp / batch_num)}\tacc_train:{acc_train}\tacc_val:{acc_val}")
    correct_rates_train.append(acc_train)  # 记录初始正确率
    correct_rates_val.append(acc_val)  # 记录初始正确率

    max_acc = acc_val

    for i in range(num_epochs):
        net.train()                                   # 【好习惯】将模型设置为训练模式
        loss_temp = 0
        batch_num = 0
        for k, (bx, by) in enumerate(train_iter()):
            bx = torch.from_numpy(bx).type(torch.FloatTensor).to(device)
            by = torch.tensor(by).long().to(device)
            optimizer.zero_grad()                             # 清空梯度累计【重要】
            p, _ = net(bx)    # 向前计算
            loss = l(p, by)   # 损失函数计算 l(output, target)
            loss.backward()   # 反向传播
            optimizer.step()  # 更新梯度
            loss_temp += float(loss)
            batch_num = k + 1

        avg_loss = float(loss_temp / batch_num)
        losses.append(avg_loss)  # 每循环记录损失函数
        acc_train = float(check_correct(net, train_iter))
        acc_val = float(check_correct(net, test_iter))
        correct_rates_train.append(acc_train)  # 每循环记录正确率
        correct_rates_val.append(acc_val)  # 每循环记录正确率
        print(f"epo:{i}\n loss:{avg_loss} \t acc_train:{acc_train}\tacc_val:{acc_val}")

        if acc_val >= max_acc:
            import time
            now = time.localtime(time.time())
            torch.save(net.state_dict(), f".\\torch_weights\\best{now.tm_mon}月{now.tm_mday}日{now.tm_hour}时.pt")
            print("已保存模型权重")
            max_acc = acc_val

        with open('.\\torch_train_info.yaml', 'w') as f:
            yaml.dump({"loss": losses, "loss_train": correct_rates_train, "loss_val": correct_rates_val}, f)
        import time
        now = time.localtime(time.time())
        torch.save(net.state_dict(), f".\\torch_weights\\last{now.tm_mon}月{now.tm_mday}日{now.tm_hour}时.pt")



if __name__ == "__main__":

	lr, num_epochs, batch_size = 1e-4, 1000, 4    # 训练参数

	main_dict = load_data.get_datas(".\\USM_data\\train_data", 0.1)
	val_dict = load_data.get_datas(".\\USM_data\\val_data", 1.0)

	train_pack = load_data.data_iter(main_dict, batch_size, shuffle=True)
	val_pack = load_data.data_iter(val_dict, batch_size)
	print(f"num_train:{len(train_pack.data_pool)}\tnum_val:{len(val_pack.data_pool)}")

	# model = torch_agent_models.Resnet_18(act_num=6, softmax=False)
	model = torch_agent_models.Vgg(act_num=6, softmax=False)
	train(model, train_pack.main_iter, val_pack.main_iter, num_epochs, lr)
