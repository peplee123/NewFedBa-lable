#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.test import test_img
import torch.nn.functional as F
import copy
import numpy as np

def distillation_loss(outputs, teacher_outputs, temperature):
    soft_labels = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    log_probs = nn.functional.log_softmax(outputs, dim=1)
    loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean') * temperature ** 2
    return loss

def bhattacharyya_distance(vector1, vector2):
    # Avoid division by zero
    epsilon = 1e-10
    vector1 = torch.mean(vector1, dim=0).cpu().detach().numpy()
    vector2 = torch.mean(vector2, dim=0).cpu().detach().numpy()
    vector1 = np.clip(vector1, epsilon, 1.0 - epsilon)
    vector2 = np.clip(vector2, epsilon, 1.0 - epsilon)
    BC = np.sum(np.sqrt(vector1 * vector2))
    return -np.log(BC)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def copy_layers(old_net, new_net, n):
    """
    将 old_net 模型的前 n 层参数复制给 new_net 模型。

    参数：
    old_net (torch.nn.Module): 要复制参数的原始模型。
    new_net (torch.nn.Module): 要接受参数复制的新模型。
    n (int): 要复制的层数。

    注意：需要确保 old_net 和 new_net 模型具有相同的层结构，以便复制参数。
    """
    if n < 1:
        raise ValueError("n 必须大于等于 1")

    old_params = list(old_net.parameters())
    new_params = list(new_net.parameters())

    if len(old_params) < n or len(new_params) < n:
        raise ValueError("模型的参数数量少于 n")

    for i in range(n):
        new_params[i].data = copy.deepcopy(old_params[i].data)


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, test_idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=self.args.local_bs, shuffle=False)
        self.last_net = None

    def train(self, net):
        if self.last_net:
            copy_layers(net, self.last_net, 7)
            net = copy.deepcopy(self.last_net)
        global_w = copy.deepcopy(net)
        net.train()
        global_w.eval()
        # train and update
        # optimizer = torch.optim.Adam(net.parameters(),lr=self.args.lr,weight_decay=1e-3)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.args.lr_decay)
        epoch_loss = []
        everyclient_distributed =[]
        total_local_probs = 0
        temperature = 1
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                global_probs = global_w(images)

                local_probs = F.softmax(log_probs, dim=1)
                global_probs = F.softmax(global_probs, dim=1)
                # loss = self.loss_func(log_probs, labels)
                # prox的正则项
                # proximal_term = 0.0
                # for w, w_t in zip(net.parameters(), global_w.parameters()):
                #     proximal_term += ((1 / 2) * torch.norm((w - w_t)) ** 2)
                # loss = self.loss_func(log_probs, labels) + proximal_term
                proximal_term = bhattacharyya_distance(local_probs,global_probs)
                loss = self.loss_func(log_probs, labels)+0.005*proximal_term

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_local_probs += local_probs.sum(dim=0)

                batch_loss.append(loss.item())
                # print("loss: ", loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print("epoch_loss: ", epoch_loss)
        sum_ = sum(total_local_probs)
        total_local_probs = torch.tensor([p/sum_ for p in total_local_probs])
        everyclient_distributed.append(total_local_probs)
        accuracyafter, test_loss = test_img(net, self.ldr_test.dataset, self.args)
        # accuracybefor, test_loss1 = test_img(global_w, self.ldr_test.dataset, self.args)
        # print('accuracy',accuracybefor)
        # print('accuracy', accuracyafter)
        # print(f"batch_loss: {sum(batch_loss) / len(batch_loss)}, acc: {accuracy}, test_loss: {test_loss}", )
        # return net, sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed
        self.last_net = copy.deepcopy(net)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed, accuracyafter
