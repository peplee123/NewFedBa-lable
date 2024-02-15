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
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA

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


# class LocalUpdate(object):
#     def __init__(self, args, dataset=None, idxs=None, test_idxs=None):
#         self.args = args
#         self.loss_func = nn.CrossEntropyLoss()
#         self.selected_clients = []
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
#         self.ldr_test = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=self.args.local_bs, shuffle=False)
#         self.last_net = None
#
#     def train(self, net):
#         print("train")
#         # print(self.last_net)
#         if self.last_net:
#             n=len( list(net.parameters()))
#             print('========',n)
#             copy_layers(net, self.last_net, n-self.args.layers)
#             net = copy.deepcopy(self.last_net)
#         global_w = copy.deepcopy(net)
#         net.train()
#         global_w.eval()
#         # train and update
#         # optimizer = torch.optim.Adam(net.parameters(),lr=self.args.lr,weight_decay=1e-3)
#         optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=1e-3)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
#         # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.args.lr_decay)
#         epoch_loss = []
#         everyclient_distributed =[]
#         total_local_probs = 0
#         temperature = 1
#         for iter in range(self.args.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.ldr_train):
#                 images, labels = images.to(self.args.device), labels.to(self.args.device)
#                 net.zero_grad()
#                 log_probs = net(images)
#                 global_probs = global_w(images)
#
#                 local_probs = F.softmax(log_probs, dim=1)
#                 global_probs = F.softmax(global_probs, dim=1)
#                 # loss = self.loss_func(log_probs, labels)
#                 # prox的正则项
#                 # proximal_term = 0.0
#                 # for w, w_t in zip(net.parameters(), global_w.parameters()):
#                 #     proximal_term += ((1 / 2) * torch.norm((w - w_t)) ** 2)
#                 # loss = self.loss_func(log_probs, labels) + proximal_term
#                 proximal_term = bhattacharyya_distance(local_probs,global_probs)
#                 loss = self.loss_func(log_probs, labels)+self.args.hy*proximal_term
#                 loss.backward()
#                 optimizer.step()
#                 scheduler.step()
#                 total_local_probs += local_probs.sum(dim=0)
#
#                 batch_loss.append(loss.item())
#                 # print("loss: ", loss)
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#         # 差分
#         # DPnet = copy.deepcopy(net)
#         # if self.args.dp_type != 0:
#         #     with torch.no_grad():  # Ensure gradients are not computed for this operation
#         #         for param in DPnet.parameters():
#         #             param.data = add_noise(param.data, self.args.dp_type, self.args)
#         # print('训练后加密结束')
#         #
#         # print("epoch_loss: ", epoch_loss)
#         sum_ = sum(total_local_probs)
#         total_local_probs = torch.tensor([p/sum_ for p in total_local_probs])
#         everyclient_distributed.append(total_local_probs)
#         # accuracyafter, test_loss = test_img(net, self.ldr_test.dataset, self.args)
#         accuracybefor, test_loss1 = test_img(global_w, self.ldr_test.dataset, self.args)
#         # print('accuracy',accuracybefor)
#         # print('accuracy', accuracyafter)
#         # print(f"batch_loss: {sum(batch_loss) / len(batch_loss)}, acc: {accuracy}, test_loss: {test_loss}", )
#         # return net, sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed
#         # print(copy.deepcopy(net))
#
#         # 改了self.last_net = copy.deepcopy(w)
#         self.last_net = copy.deepcopy(net)
#         return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed, accuracybefor
class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None,test_idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(self.args.dp_sample * len(idxs)), replace=False)
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=len(self.idxs_sample),
                                    shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=len(self.idxs_sample), shuffle=False)
        self.idxs = idxs
        self.times = self.args.epochs * self.args.frac
        self.lr = args.lr
        self.noise_scale = self.calculate_noise_scale()
        self.last_net = None


    def calculate_noise_scale(self):
        if self.args.dp_mechanism == 'Laplace':
            epsilon_single_query = self.args.dp_epsilon / self.times
            return Laplace(epsilon=epsilon_single_query)
        elif self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        elif self.args.dp_mechanism == 'MA':
            return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample, epoch=self.times)

    def train(self, net):
        if self.last_net:
            n = len(list(net.parameters()))
            print('========', n)
            copy_layers(net, self.last_net, n - self.args.layers)
            net = copy.deepcopy(self.last_net)
        global_w = copy.deepcopy(net)
        net.train()
        global_w.eval()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_client = 0
        everyclient_distributed = []
        total_local_probs = 0
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            log_probs = net(images)
            global_probs = global_w(images)

            local_probs = F.softmax(log_probs, dim=1)
            global_probs = F.softmax(global_probs, dim=1)

            proximal_term = bhattacharyya_distance(local_probs, global_probs)
            loss = self.loss_func(log_probs, labels) + self.args.hy * proximal_term
            loss.backward()
            if self.args.dp_mechanism != 'no_dp':
                self.clip_gradients(net)
            optimizer.step()
            scheduler.step()
            # add noises to parameters
            if self.args.dp_mechanism != 'no_dp':
                self.add_noise(net)
            loss_client = loss.item()
            total_local_probs += local_probs.sum(dim=0)
        self.lr = scheduler.get_last_lr()[0]
        sum_ = sum(total_local_probs)
        total_local_probs = torch.tensor([p / sum_ for p in total_local_probs])
        # print(total_local_probs)
        everyclient_distributed.append(total_local_probs)
        # accuracyafter, test_loss = test_img(net, self.ldr_test.dataset, self.args)
        accuracybefor, test_loss1 = test_img(global_w, self.ldr_test.dataset, self.args)
        # print(average_loss)
        self.last_net = copy.deepcopy(net)
        return net.state_dict(), loss_client, scheduler.get_last_lr()[0], everyclient_distributed, accuracybefor

    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=1)
        elif self.args.dp_mechanism == 'Gaussian' or self.args.dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, len(self.idxs_sample))
        state_dict = net.state_dict()
        if self.args.dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'Gaussian':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'MA':
            sensitivity = cal_sensitivity_MA(self.args.lr, self.args.dp_clip, len(self.idxs_sample))
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        net.load_state_dict(state_dict)


class LocalUpdateDPSerial(LocalUpdateDP):
    def __init__(self, args, dataset=None, idxs=None,test_idxs=None):
        super().__init__(args, dataset, idxs, test_idxs)

    def train(self, net):
        if self.last_net:
            n = len(list(net.parameters()))
            print('========', n)
            copy_layers(net, self.last_net, n - self.args.layers)
            net = copy.deepcopy(self.last_net)
        global_w = copy.deepcopy(net)
        net.train()
        global_w.eval()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        losses = 0
        total_local_probs = 0
        everyclient_distributed = []
        for images, labels in self.ldr_train:
            net.zero_grad()
            index = int(len(images) / self.args.serial_bs)
            total_grads = [torch.zeros(size=param.shape).to(self.args.device) for param in net.parameters()]
            for i in range(0, index + 1):
                net.zero_grad()
                start = i * self.args.serial_bs
                end = (i+1) * self.args.serial_bs if (i+1) * self.args.serial_bs < len(images) else len(images)
                # print(end - start)
                if start == end:
                    break
                image_serial_batch, labels_serial_batch \
                    = images[start:end].to(self.args.device), labels[start:end].to(self.args.device)
                log_probs = net(image_serial_batch)
                global_probs = global_w(image_serial_batch)

                local_probs = F.softmax(log_probs, dim=1)
                global_probs = F.softmax(global_probs, dim=1)

                proximal_term = bhattacharyya_distance(local_probs, global_probs)
                loss = self.loss_func(log_probs, labels_serial_batch) + self.args.hy * proximal_term
                loss.backward()
                total_local_probs += local_probs.sum(dim=0)
                if self.args.dp_mechanism != 'no_dp':
                    self.clip_gradients(net)
                grads = [param.grad.detach().clone() for param in net.parameters()]
                for idx, grad in enumerate(grads):
                    total_grads[idx] += torch.mul(torch.div((end - start), len(images)), grad)
                losses += loss.item() * (end - start)
            for i, param in enumerate(net.parameters()):
                param.grad = total_grads[i]
            optimizer.step()
            scheduler.step()
            # add noises to parameters
            if self.args.dp_mechanism != 'no_dp':
                self.add_noise(net)
            self.lr = scheduler.get_last_lr()[0]
            sum_ = sum(total_local_probs)
            total_local_probs = torch.tensor([p / sum_ for p in total_local_probs])
            everyclient_distributed.append(total_local_probs)
            # accuracyafter, test_loss = test_img(net, self.ldr_test.dataset, self.args)
            accuracybefor, test_loss1 = test_img(global_w, self.ldr_test.dataset, self.args)
            self.last_net = copy.deepcopy(net)
        return net.state_dict(), losses / len(self.idxs_sample),scheduler.get_last_lr()[0], everyclient_distributed, accuracybefor