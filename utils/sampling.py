#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def build_noniid_pfl(dataset, num_users, alpha, train_ratio=0.8):
    train_labels = np.array(dataset.targets)
    n_classes = np.max(train_labels) + 1
    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)

    class_idxs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_train_idxs = [[] for _ in range(num_users)]
    client_test_idxs = [[] for _ in range(num_users)]

    for c, fracs in zip(class_idxs, label_distribution):
        num_samples = len(c)
        num_train_samples = int(train_ratio * num_samples)
        train_frac = fracs * train_ratio
        test_frac = fracs * (1 - train_ratio)

        train_sizes = (train_frac * num_train_samples).astype(int)
        test_sizes = (test_frac * (num_samples - num_train_samples)).astype(int)

        train_idx = 0
        test_idx = 0
        for i in range(num_users):
            train_end = train_idx + train_sizes[i]
            test_end = test_idx + test_sizes[i]
            client_train_idxs[i] += list(c[train_idx:train_end])
            client_test_idxs[i] += list(c[test_idx:test_end])
            train_idx = train_end
            test_idx = test_end

    dict_users_train = {i: client_train_idxs[i] for i in range(len(client_train_idxs))}
    dict_users_test = {i: client_test_idxs[i] for i in range(len(client_test_idxs))}

    draw_data_distribution(dict_users_train, dataset, n_classes)
    draw_data_distribution(dict_users_test, dataset, n_classes)

    return dict_users_train, dict_users_test


def noniid(args,dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    n_class =args.bingtai
    num_shards, num_imgs = num_users * n_class, int(len(dataset) / (num_users * n_class))
    idx_shard = [i for i in range(num_shards)]
    train_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array([], dtype="int64")
    for d in dataset.datasets:
        labels = np.append(labels, np.array(d.targets, dtype='int64'))
    # labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))

    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, n_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            data = np.concatenate((train_dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            train_dict_users[i], test_dict_users[i] = train_test_split(data, train_size=0.8, shuffle=True)
            # train_dict_users[i], test_dict_users[i] = data[:int(0.8*len(data))], data[int(0.8*len(data)):]
    return train_dict_users, test_dict_users



def bingtai_mnist(dataset,num_clients, num_classes_per_client, num_samples_per_class):
    # 加载MNIST数据集
    # train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    # 创建一个包含所有数据索引的列表
    all_indices = list(range(len(dataset)))
    # 随机打乱索引列表
    random.shuffle(all_indices)
    dict_users = {}  # 存储划分结果的字典
    # 划分数据集
    for i in range(num_clients):
        selected_indices = []  # 当前客户端的数据索引
        # 从每个类别中选择两类
        selected_classes = random.sample(range(10), num_classes_per_client)
        # 选择每个类别的数据样本
        for class_label in selected_classes:
            indices = [index for index in all_indices if dataset.targets[index] == class_label]
            selected_indices.extend(random.sample(indices, num_samples_per_class))
        dict_users[i] = selected_indices
    draw_data_distribution(dict_users, dataset, 10)
    return dict_users


def build_noniid(dataset, num_users, alpha):
    print("DDDD1")
    train_labels = np.array([], dtype="int64")
    for d in dataset.datasets:
        train_labels = np.append(train_labels, np.array(d.targets, dtype='int64'))
    # train_labels = np.array(dataset.targets)
    n_classes = np.max(train_labels) + 1
    label_distribution = np.random.dirichlet([alpha] * num_users, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idxs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    # 记录每个K类别对应的样本下标

    client_idxs = [[] for _ in range(num_users)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idxs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idxs 为遍历第i个client对应样本集合的索引
        for i, idxs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idxs[i] += [idxs]

    client_idxs = [np.concatenate(idxs) for idxs in client_idxs]
    #
    train_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for i in range(len(client_idxs)):
        data = client_idxs[i]
        train_dict_users[i], test_dict_users[i] = train_test_split(data, train_size=0.8, shuffle=True)

    # draw_data_distribution(train_dict_users, dataset, n_classes)
    # draw_data_distribution(test_dict_users, dataset, n_classes)
    return train_dict_users, test_dict_users


def draw_data_distribution(dict_users, dataset, num_class):
    import matplotlib.pyplot as plt
    targets = dataset.targets

    # plt.figure(figsize=(20, 3))
    plt.hist([np.array(targets)[idc] for idc in dict_users.values()], stacked=True,
             bins=np.arange(min(targets) - 0.5, max(targets) + 1.5, 1),
             label=["C{}".format(i) for i in range(len(dict_users))], rwidth=0.5)
    plt.xticks(np.arange(num_class), rotation=70)
    plt.legend(loc=(0.95, -0.1))
    plt.savefig("2.jpg")
    plt.show()


if __name__ == '__main__':

    trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download=True,
                                          transform=trans_fashion_mnist)

    train_dict_users, test_dict_users = noniid(dataset_train, 10)
    draw_data_distribution(train_dict_users, dataset_train, 10)
    draw_data_distribution(test_dict_users, dataset_train, 10)
    # num = 100
    # d = mnist_iid(dataset_train, num)
    # path = '../data/fashion_iid_100clients.dat'
    # file = open(path, 'w')
    # for idx in range(num):
    #     for i in d[idx]:
    #         file.write(str(i))
    #         file.write(',')
    #     file.write('\n')
    # file.close()
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    # print(fashion_iid(dataset_train, 1000)[0])


