#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.test import test_img
import torch.nn.functional as F



def distillation_loss(outputs, teacher_outputs, temperature):
    soft_labels = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    log_probs = nn.functional.log_softmax(outputs, dim=1)
    loss = nn.functional.kl_div(log_probs, soft_labels, reduction='batchmean') * temperature ** 2
    return loss


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, test_idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset, test_idxs), batch_size=self.args.local_bs, shuffle=False)

    def train(self, net):
        net.train()
        # global_w = copy.deepcopy(net)
        # global_w.eval()
        # train and update
        # optimizer = torch.optim.Adam(net.parameters(),lr=self.args.lr,weight_decay=1e-3)
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=self.args.lr_decay)
        epoch_loss = []
        everyclient_distributed =[]
        total_local_probs = 0
        temperature =1
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                # output_teacher = global_w(images)
                # output_student = net(images)
                # loss = distillation_loss(output_student, output_teacher, temperature)

                # print(log_probs)

                loss = self.loss_func(log_probs, labels)
                # global_output = global_w(images)
                local_probs = F.softmax(log_probs, dim=1)
                # global_probs = F.softmax(global_output, dim=1)
                #
                # local_probs = torch.clamp(local_probs, min=1e-10)
                # global_probs = torch.clamp(global_probs, min=1e-10)
                # print(global_probs)
                # kl_div = F.kl_div(local_probs.log(), global_probs.detach(), reduction='batchmean')
                # print("kl_div: ", kl_div)
                # loss =loss+0.5*kl_div
                # -----------
                # params = weight_flatten(net)
                # params_ = weight_flatten(global_w)
                # sub = params - params_
                # loss = 0.8*loss + 0.2 * torch.dot(sub, sub)
                # -----------
                # KL = F.kl_div(log_probs.softmax(dim=-1).log(), labels.softmax(dim=-1), reduction='sum')
                # labels = labels.reshape(labels.shape[0], 1)
                # # labels = torch.tensor(labels, dtype=torch.int64).cuda()
                #
                # KL = kl_loss(Softmax(log_probs), Softmax(labels))

                # print(KL.cpu().detach().numpy())

                # loss = 0.5*loss+0.5*KL
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
        accuracy, test_loss = test_img(net, self.ldr_test.dataset, self.args)
        print(accuracy)

        # print(f"batch_loss: {sum(batch_loss) / len(batch_loss)}, acc: {accuracy}, test_loss: {test_loss}", )
        # return net, sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0],everyclient_distributed
